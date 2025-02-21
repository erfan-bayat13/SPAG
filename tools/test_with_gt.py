import os
import argparse
from copy import deepcopy
import json
import glob
from dataclasses import dataclass
import random
from typing import Dict, List, Sequence
from tqdm import tqdm


import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

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

# Define action cards with their effects
ACTION_CARDS = {
    "Block": "Prevents the opponent from using their next action card",
    "Double Guess": "Defender can make two guesses this turn without losing",
    "Hint Shield": "Attacker must be more direct in their next hint",
    "Word Ban": "Defender can ban a word (except the target) from being used",
    "Topic Switch": "Force the conversation to shift to a new topic",
    "Reverse Psychology": "Next hint must be about what the word is NOT",
    "Time Extension": "Adds one additional turn to the game",
    "Clarity Check": "Force opponent to rephrase their last statement"
}

def initialize_player_cards(num_cards=3):
    """Give each player a random hand of action cards"""
    available_cards = list(ACTION_CARDS.keys())
    return random.sample(available_cards, num_cards)

def create_enhanced_game_prompt(history, target_word, max_turns, role, player_cards, active_effects):
    """Creates a prompt that includes action card information"""
    base_prompt = f"""You are playing an enhanced word guessing game with special action cards.

BASIC RULES:
- Attacker: Knows a secret word and tries to make the defender say it without using it
- Defender: Tries to figure out the secret word and can win by saying "I know the word! It is [word]"
- Game ends if: defender guesses correctly, defender says target word unknowingly, attacker uses target word, or after {max_turns} turns

YOUR ACTION CARDS:
{', '.join(f'"{card}": {ACTION_CARDS[card]}' for card in player_cards)}

To use a card, start your response with "USE CARD: [card name] - " followed by your message.

ACTIVE EFFECTS: {', '.join(active_effects) if active_effects else 'None'}
"""
    
    if role == "attacker":
        base_prompt += f"\nYou are the ATTACKER. The secret word is '{target_word}'.\n"
    else:
        base_prompt += "\nYou are the DEFENDER.\n"

    if history:
        base_prompt += "\nConversation history:\n"
        for msg in history:
            base_prompt += f"{msg['role'].upper()}: {msg['content']}\n"
            if msg.get('card_used'):
                base_prompt += f"(Used card: {msg['card_used']})\n"
    
    base_prompt += f"\n{role.upper()}: "
    return base_prompt

def process_response(response: str, player_cards: List[str]) -> Dict:
    """Process a response to extract card usage and message"""
    if response.upper().startswith("USE CARD:"):
        try:
            card_part, message = response.split("-", 1)
            card_name = card_part.replace("USE CARD:", "").strip()
            if card_name in player_cards:
                return {
                    "card_used": card_name,
                    "content": message.strip(),
                    "cards_remaining": [c for c in player_cards if c != card_name]
                }
        except ValueError:
            pass
    
    return {
        "card_used": None,
        "content": response.strip(),
        "cards_remaining": player_cards
    }

def apply_card_effects(card_name: str, game_state: Dict) -> Dict:
    """Apply effects of used cards to the game state"""
    effects = game_state.get("active_effects", [])
    
    if card_name == "Time Extension":
        game_state["max_turns"] += 1
    elif card_name == "Block":
        effects.append("Next card blocked")
    elif card_name == "Double Guess":
        effects.append("Double guess allowed")
    elif card_name == "Hint Shield":
        effects.append("Direct hint required")
    # ... implement other card effects
    
    game_state["active_effects"] = effects
    return game_state

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
    # setup model
    #---------------------------------------------------------------------------------
    players = dict()
    players['attacker'] = load_model_and_tokenizer(args, args.attacker_model_name_or_path)
    
    if args.attacker_model_name_or_path == args.defender_model_name_or_path:
        players['defender'] = players['attacker']
    else:
        players['defender'] = load_model_and_tokenizer(args, args.defender_model_name_or_path)
    
    
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

        # Initialize games with cards for each player
        batch_games = []
        for keyword in batch_words:
            game = {
                "history": [],
                "target_word": keyword,
                "max_turns": args.taboo_max_turns,
                "player_cards": {
                    "attacker": initialize_player_cards(),
                    "defender": initialize_player_cards()
                },
                "active_effects": []
            }
            batch_games.append(game)
        
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

            # Create prompts with card information
            batch_queries = [{
                "query": create_enhanced_game_prompt(
                    game['history'],
                    target_word=game['target_word'],
                    max_turns=game['max_turns'],
                    role=next_player,
                    player_cards=game['player_cards'][next_player],
                    active_effects=game['active_effects']
                ),
                "query_id": game['target_word']
            } for game in batch_games]

            # Generate responses
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

            # Process responses and update game states
            finished_ids = []
            for idx, input_str in enumerate(inputs_string):
                output_response = tokenizer.decode(output_seq[idx], skip_special_tokens=True)
                response_sample = output_response.replace(input_str, '').split(tokenizer.eos_token)[0].strip()
                
                # Process the response for card usage
                response_info = process_response(
                    response_sample,
                    batch_games[idx]['player_cards'][next_player]
                )
                
                # Update game state with response and card effects
                if response_info['card_used']:
                    batch_games[idx] = apply_card_effects(
                        response_info['card_used'],
                        batch_games[idx]
                    )
                    batch_games[idx]['player_cards'][next_player] = response_info['cards_remaining']
                
                batch_games[idx]['history'].append({
                    'role': next_player,
                    'content': response_info['content'],
                    'card_used': response_info['card_used']
                })
                
                # Check for game end conditions
                if "i know the word" in response_info['content'].lower() and next_player == 'defender':
                    all_outputs.append(batch_games[idx])
                    finished_ids.append(idx)
            
            # Remove finished games
            batch_games = [game for idx, game in enumerate(batch_games) if idx not in finished_ids]
            if len(batch_games) == 0:
                break
                
        all_outputs.extend(batch_games)
        if dist.get_rank() == 0 and (step % args.logging_steps == 0):
            print_rank_0(f"finished {step} of {len(dataloader)}")
            print_rank_0(all_outputs[-1])

    output_file_prefix = f"{args.output_dir}/{args.model_prefix}_{args.task_type}_{args.data_suffix}"
    with open(f"{output_file_prefix}_rank{dist.get_rank()}.json", 'w') as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)
    print(f"rank {dist.get_rank()} finishs inference.")

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
        print(f"finished inference results merge.")


if __name__ == "__main__":
    main()