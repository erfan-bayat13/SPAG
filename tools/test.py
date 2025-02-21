import json
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from arguments import CustomTrainingArguments
from utils import DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN
from dataloaders import batch_padding

def load_keyword_list(data_path):
    with open(data_path, 'r') as f:
        keywords = f.read().strip().split('\n')
    return keywords

def query_data_collactor(args, batch, tokenizer):
    input_ids, attention_mask, labels = [], [], []
    text = [item['query'] for item in batch]
    query_ids = [item['query_id'] for item in batch]

    for sent in text:
        input_query_ids = [tokenizer.bos_token_id] + tokenizer.encode(sent, add_special_tokens=False)            
        input_ids.append(input_query_ids)

    outputs = batch_padding(
        input_ids,
        tokenizer,
        max_length=tokenizer.model_max_length - args.max_new_tokens
    )
    
    outputs['query_ids'] = query_ids
    outputs['text'] = text
    return outputs

def create_direct_game_prompt(history, target_word, max_turns, role):
    base_prompt = f"""You are playing a word guessing game with two roles:
- Attacker: Knows a secret word and tries to make the defender say it without using it the attacker will give out hints to play the game
- Defender: Tries to figure out the secret word and can win by saying "I know the word! It is [word]"
- Game ends if: defender guesses correctly, defender says target word unknowingly, attacker uses target word, or after {max_turns} turns
"""
    if role == "attacker":
        base_prompt += f"\nYou are the ATTACKER. The secret word is '{target_word}'.\nYour goal: Make the defender say '{target_word}' without using it yourself.\n"
    else:
        base_prompt += "\nYou are the DEFENDER. Try to figure out the secret word. If you know it, say 'I know the word! It is [your guess]'.\n"

    if history:
        base_prompt += "\nConversation history:\n"
        for msg in history:
            base_prompt += f"{msg['role'].upper()}: {msg['content']}\n"
    
    base_prompt += f"\n{role.upper()}: "
    return base_prompt

def load_model_and_tokenizer(model_name_or_path, args):
    print(f"Loading model from {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_cache=True,
        torch_dtype=torch.float16 if args.bf16 else torch.float32,
        device_map='auto'
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="left",
        truncation_side='left',
        model_max_length=args.max_length,
        trust_remote_code=True
    )

    # Handle special tokens
    special_tokens = {
        "pad_token": DEFAULT_PAD_TOKEN,
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
    }
    
    # Add missing special tokens
    for token_name, token_value in special_tokens.items():
        if getattr(tokenizer, token_name) is None:
            tokenizer.add_special_tokens({token_name: token_value})
            
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    
    return {"model": model, "tokenizer": tokenizer}

def main():
    parser = transformers.HfArgumentParser(CustomTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    eval_dataset = load_keyword_list(args.data_path)
    
    # Setup models
    players = {}
    players['attacker'] = load_model_and_tokenizer(args.attacker_model_name_or_path or args.model_name_or_path, args)
    if args.defender_model_name_or_path and args.defender_model_name_or_path != args.attacker_model_name_or_path:
        players['defender'] = load_model_and_tokenizer(args.defender_model_name_or_path, args)
    else:
        players['defender'] = players['attacker']

    dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        batch_size=args.per_device_eval_batch_size
    )

    all_outputs = []
    
    for step, batch_words in enumerate(dataloader):
        print(f"Processing batch {step+1}/{len(dataloader)}")
        
        batch_games = [
            {"history": [], "target_word": keyword, "max_turns": args.taboo_max_turns}
            for keyword in batch_words
        ]
        
        for taboo_turn in range(2 * args.taboo_max_turns):            
            next_player = "attacker" if taboo_turn % 2 == 0 else "defender"
            model, tokenizer = players[next_player]['model'], players[next_player]['tokenizer']

            generation_config = GenerationConfig(
                max_new_tokens=args.max_new_tokens,
                temperature=1.2 if args.task_type == "sampling" else 1.0,
                do_sample=args.task_type == "sampling",
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=1,
            )

            batch_queries = [{
                "query": create_direct_game_prompt(
                    game['history'],
                    target_word=game['target_word'],
                    max_turns=game['max_turns'],
                    role=next_player
                ),
                "query_id": game['target_word']
            } for game in batch_games]

            batch = query_data_collactor(args, batch_queries, tokenizer)           
        
            input_ids = torch.tensor(batch['input_ids']).to(model.device)
            attention_mask = torch.tensor(batch['attention_mask']).to(model.device)
            
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                )
                
            output_seq = generation_output.sequences
            inputs_string = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            finished_ids = []
            for idx, input_str in enumerate(inputs_string):
                output_response = tokenizer.decode(output_seq[idx], skip_special_tokens=True)
                response_sample = output_response.replace(input_str, '').split(tokenizer.eos_token)[0].strip()
                
                if args.debug_mode:
                    print(f"\nTurn {taboo_turn + 1}, {next_player.upper()}:")
                    print(f"Response: {response_sample}")
                
                batch_games[idx]['history'].append({
                    'role': next_player,
                    'content': response_sample
                })
                
                if "i know the word" in response_sample.lower() and next_player == 'defender':
                    all_outputs.append(batch_games[idx])
                    finished_ids.append(idx)
                    
            batch_games = [game for idx, game in enumerate(batch_games) if idx not in finished_ids]
            if len(batch_games) == 0:
                break
                
        all_outputs.extend(batch_games)
        
        if step % args.logging_steps == 0:
            print(f"Finished {step + 1} of {len(dataloader)} batches")
            if all_outputs:
                print("Latest game:", json.dumps(all_outputs[-1], indent=2))

    # Save results
    output_file = f"{args.output_dir}/{args.model_prefix}_{args.task_type}_{args.data_suffix}_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()