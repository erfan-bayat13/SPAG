import os
import argparse
from copy import deepcopy
import json
import glob
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Sequence
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from arguments import CustomTrainingArguments
from utils import print_rank_0, read_json_or_jsonl_data, set_special_tokens
from utils import convert_game_history_to_query, check_defender_response
from dataloaders import batch_padding

class HSDataset(Dataset):
    def __init__(self, words, explanations):
        self.words = words
        self.explanations = explanations
    
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, idx):
        return {
            'word': self.words[idx],
            'explanation': self.explanations[idx]
        }

def load_hs_dataset(args, data_path):
    """Load dataset containing words and their explanations"""
    # Assuming CSV format with 'word' and 'explanation' columns
    df = pd.read_csv(data_path)
    return HSDataset(df['word'].tolist(), df['explanation'].tolist())

def query_data_collactor(args, batch, tokenizer):
    input_ids, attention_mask = [], []
    text = [item['query'] for item in batch]
    query_ids = [f"{item['query_id']}|{item['explanation']}" for item in batch]

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

def load_model_and_tokenizer(args, model_name_or_path):
    print_rank_0(f"start loading model from {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_cache=True,
        torch_dtype=torch.float16,
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
    
    # Load dataset with words and explanations
    eval_dataset = load_hs_dataset(args, args.data_path)

    # Setup models
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
    
    for step, batch_data in enumerate(dataloader):
        progress_bar.update(1)

        batch_games = [
            {
                "history": [], 
                "target_word": word,
                "explanation": explanation,
                "max_turns": args.taboo_max_turns
            }
            for word, explanation in zip(batch_data['word'], batch_data['explanation'])
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
                "query": convert_game_history_to_query(
                    game['history'],
                    target_word=f"{game['target_word']} - {game['explanation']}",
                    max_turns=game['max_turns']
                ),
                "query_id": game['target_word'],
                "explanation": game['explanation']
            } for game in batch_games]

            batch = query_data_collactor(args, batch_queries, tokenizer)           
        
            input_ids = torch.Tensor(batch['input_ids']).long().to(model.device)        
            attention_mask = torch.Tensor(batch['attention_mask']).float().to(model.device)
            query_ids = batch['query_ids']
            text = batch['text']
            batch_size = input_ids.shape[0]
        
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                )
                
            output_seq = generation_output.sequences.reshape(batch_size, generation_config.num_return_sequences, -1)        
            inputs_string = tokenizer.batch_decode(input_ids.reshape(batch_size, -1), skip_special_tokens=True)

            finished_ids = []
            for idx in range(batch_size):
                output_response = tokenizer.batch_decode(output_seq[idx], skip_special_tokens=True)[0]
                response_sample = output_response.replace(inputs_string[idx], '').split(tokenizer.eos_token)[0]
                batch_games[idx]['history'].append({'role': next_player, 'content': response_sample})
                
                if next_player == 'defender':
                    has_format, guess, counterspeech = check_defender_response(response_sample)
                    if has_format:
                        # Store the original word/explanation, defender's guess, and counterspeech
                        all_outputs.append({
                            'target_word': batch_games[idx]['target_word'],
                            'target_explanation': batch_games[idx]['explanation'],
                            'defender_guess': guess,
                            'defender_counterspeech': counterspeech,
                            'full_history': batch_games[idx]['history']
                        })
                        finished_ids.append(idx)
                    
            batch_games = [game for idx, game in enumerate(batch_games) if idx not in finished_ids]
            if len(batch_games) == 0:
                break            
                
        # Add any unfinished games with their current state
        for game in batch_games:
            all_outputs.append({
                'target_word': game['target_word'],
                'target_explanation': game['explanation'],
                'defender_guess': 'NO_GUESS',
                'defender_counterspeech': 'NO_COUNTERSPEECH',
                'full_history': game['history']
            })

        if dist.get_rank() == 0 and (step % args.logging_steps == 0):
            print_rank_0(f"finished {step} of {len(dataloader)}")
            print_rank_0(all_outputs[-1])

    # Save results
    output_file_prefix = f"{args.output_dir}/{args.model_prefix}_{args.task_type}_{args.data_suffix}"
    with open(f"{output_file_prefix}_rank{dist.get_rank()}.json", 'w') as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)
    print(f"rank {dist.get_rank()} finishes inference.")

    # Cleanup
    if 'model' in players['attacker']:
        del players['attacker']['model']
    if 'model' in players['defender']:
        del players['defender']['model']
        
    torch.cuda.empty_cache() 
    dist.barrier()

    # Merge results from all ranks
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