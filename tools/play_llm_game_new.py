import glob
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

import tqdm
import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from arguments import CustomTrainingArguments

from utils import print_rank_0, read_json_or_jsonl_data
from utils import DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN
from utils import convert_game_history_to_query, set_special_tokens

from dataloaders import batch_padding


def load_keyword_list(args, data_path):
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

def create_direct_game_prompt(history,target_word,max_turns, role):
    base_prompt = f"""You are playing a word guessing game with two roles:
- Attacker: Knows a secret word and tries to make the defender say it without using it
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

def load_model_and_tokenizer(args, model_name_or_path):
    print_rank_0(f"start loading model from {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_cache=True,
        torch_dtype=torch.float16,
        # device_map='auto'
    )
    if hasattr(model, 'ref_model'):
        del model.ref_model
        
    print_rank_0(model)
    
    device = torch.cuda.current_device()
    model.to(device)
    model.eval()
   
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="left",  # for batch decode
        truncation_side='left',
        model_max_length=args.max_length,
        trust_remote_code=True
    )

    model, tokenizer = set_special_tokens(model, tokenizer)
    
    return {"model": model, "tokenizer": tokenizer}

def main():
    parser = transformers.HfArgumentParser(CustomTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    eval_dataset = load_keyword_list(args, args.data_path)

    # Setup models - now using base models directly
    players = dict()
    players['attacker'] = load_model_and_tokenizer(args, args.model_name_or_path)
    
    if args.defender_model_name_or_path:
        players['defender'] = load_model_and_tokenizer(args, args.defender_model_name_or_path)
    else:
        players['defender'] = players['attacker']  # Use same model for both roles
    
    sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=True)
    dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        batch_size=args.per_device_eval_batch_size,
        sampler=sampler,
    )

    all_outputs = []
    progress_bar = tqdm(range(len(dataloader)), disable=(dist.get_rank() != 0))
    
    for step, batch_words in enumerate(dataloader):
        progress_bar.update(1)

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
        
            input_ids = torch.Tensor(batch['input_ids']).long().to(model.device)        
            attention_mask = torch.Tensor(batch['attention_mask']).float().to(model.device)
            
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
        if dist.get_rank() == 0 and (step % args.logging_steps == 0):
            print_rank_0(f"finished {step} of {len(dataloader)}")
            print_rank_0(all_outputs[-1])

    # Save results
    output_file_prefix = f"{args.output_dir}/{args.model_prefix}_{args.task_type}_{args.data_suffix}"
    with open(f"{output_file_prefix}_rank{dist.get_rank()}.json", 'w') as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)

    # Cleanup and merge results
    if 'model' in players['attacker']:
        del players['attacker']['model']
    if 'model' in players['defender']:
        del players['defender']['model']
        
    torch.cuda.empty_cache() 
    dist.barrier()
    
    if dist.get_rank() == 0:
        result_paths = glob.glob(f"{output_file_prefix}_rank*.json")
        all_results = []
        for res_path in result_paths:
            new_results = read_json_or_jsonl_data(res_path)
            all_results.extend(new_results)

        print(f"totally loaded {len(all_results)} results")
        with open(f"{output_file_prefix}_results.json", 'w') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()