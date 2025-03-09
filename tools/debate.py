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
from utils import convert_debate_history_to_query, check_defender_response
from dataloaders import batch_padding

class DebateDataset(Dataset):
    def __init__(self, topics, positions):
        """
        Dataset for debate topics and positions
        
        Args:
            topics: List of general topic areas (e.g. "immigration", "religion", etc.)
            positions: List of specific problematic positions on those topics
        """
        self.topics = topics
        self.positions = positions
    
    def __len__(self):
        return len(self.topics)
    
    def __getitem__(self, idx):
        return {
            'topic': self.topics[idx],
            'position': self.positions[idx]
        }

def load_debate_dataset(args, data_path):
    """Load dataset containing debate topics and positions"""
    # Assuming CSV format with 'topic' and 'position' columns
    df = pd.read_csv(data_path)
    return DebateDataset(df['topic'].tolist(), df['position'].tolist())

def query_data_collactor(args, batch, tokenizer):
    input_ids, attention_mask = [], []
    text = [item['query'] for item in batch]
    query_ids = [f"{item['query_id']}|{item['position']}" for item in batch]

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
    
    # Load dataset with topics and positions
    eval_dataset = load_debate_dataset(args, args.data_path)

    # Setup models
    players = dict()
    players['proponent'] = load_model_and_tokenizer(args, args.attacker_model_name_or_path)
    
    if args.attacker_model_name_or_path == args.defender_model_name_or_path:
        players['opponent'] = players['proponent']
    else:
        players['opponent'] = load_model_and_tokenizer(args, args.defender_model_name_or_path)
    
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

        batch_debates = [
            {
                "history": [], 
                "topic": topic,
                "position": position,
                "max_turns": args.taboo_max_turns if args.taboo_max_turns <= 4 else 4
            }
            for topic, position in zip(batch_data['topic'], batch_data['position'])
        ]
        
        # Each turn consists of a proponent and opponent exchange (2 messages)
        for debate_turn in range(2 * max([debate['max_turns'] for debate in batch_debates])):            
            next_player = "proponent" if debate_turn % 2 == 0 else "opponent"
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
                "query": convert_debate_history_to_query(
                    debate['history'],
                    topic=debate['topic'],
                    position=debate['position'],
                    max_turns=debate['max_turns']
                ),
                "query_id": debate['topic'],
                "position": debate['position']
            } for debate in batch_debates]

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
                
                # Check if this debate has exceeded its max turns
                current_turn = (len(batch_debates[idx]['history']) // 2) + 1
                if current_turn > batch_debates[idx]['max_turns'] and next_player == "opponent":
                    # This debate is complete - both sides have had their final turn
                    all_outputs.append({
                        'topic': batch_debates[idx]['topic'],
                        'position': batch_debates[idx]['position'],
                        'full_debate': batch_debates[idx]['history'] + [{'role': next_player, 'content': response_sample}]
                    })
                    finished_ids.append(idx)
                else:
                    # Add this response to the debate history
                    batch_debates[idx]['history'].append({'role': next_player, 'content': response_sample})
                    
            # Remove completed debates
            batch_debates = [debate for idx, debate in enumerate(batch_debates) if idx not in finished_ids]
            if len(batch_debates) == 0:
                break            

        # Add any remaining debates that haven't finished their turns
        for debate in batch_debates:
            all_outputs.append({
                'topic': debate['topic'],
                'position': debate['position'],
                'full_debate': debate['history'],
                'status': 'incomplete'
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
    if 'model' in players['proponent']:
        del players['proponent']['model']
    if 'model' in players['opponent']:
        del players['opponent']['model']
        
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