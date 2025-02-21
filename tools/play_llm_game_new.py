import os
import argparse
from copy import deepcopy
import json
import glob
from dataclasses import dataclass
from typing import Dict, Sequence, List
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

import tqdm
import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from arguments import CustomTrainingArguments
from utils import (
    print_rank_0, 
    convert_game_history_to_query,
    is_prediction_attempt,
    extract_prediction_and_counterspeech,
    has_exact_hate_speech,
    set_special_tokens
)
from dataloaders import batch_padding

def load_hate_speech_data(args, data_path: str) -> List[str]:
    """Load hate speech statements from file."""
    with open(data_path, 'r') as f:
        statements = [line.strip() for line in f.readlines()]
    print_rank_0(f"Loaded {len(statements)} statements from {data_path}")
    return statements

def load_model_and_tokenizer(args, model_name_or_path: str) -> Dict:
    """Load model and tokenizer with proper configuration."""
    print_rank_0(f"Loading model from {model_name_or_path}")
    
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

def query_data_collactor(args, batch, tokenizer):
    """Prepare batch data for model input."""
    input_ids, attention_mask = [], []
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

def get_model_response(model, tokenizer, input_ids, attention_mask, generation_config):
    """Get model response with proper error handling."""
    try:
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
            )
        return generation_output
    except Exception as e:
        print_rank_0(f"Error in model generation: {str(e)}")
        return None

def process_model_outputs(outputs, tokenizer, input_ids, input_strings):
    """Process model outputs into readable responses."""
    if outputs is None:
        return ["Error generating response"] * len(input_ids)
    
    batch_size = input_ids.shape[0]
    output_seq = outputs.sequences.reshape(batch_size, -1)
    
    responses = []
    for idx in range(batch_size):
        try:
            output_text = tokenizer.decode(output_seq[idx], skip_special_tokens=True)
            response = output_text.replace(input_strings[idx], '').split(tokenizer.eos_token)[0]
            responses.append(response.strip())
        except Exception as e:
            print_rank_0(f"Error processing output {idx}: {str(e)}")
            responses.append("Error processing response")
    
    return responses

def check_game_end(game_state, response, role):
    """Check if the game should end based on the current state."""
    if role == "defender" and is_prediction_attempt(response):
        return True
    if role == "attacker" and has_exact_hate_speech(response, game_state["target_statement"]):
        return True
    if len(game_state["history"]) >= game_state["max_turns"] * 2:
        return True
    return False

def main():
    # Parse arguments
    parser = transformers.HfArgumentParser(CustomTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Load dataset
    eval_dataset = load_hate_speech_data(args, args.data_path)

    # Setup models
    players = {
        'attacker': load_model_and_tokenizer(args, args.attacker_model_name_or_path),
        'defender': load_model_and_tokenizer(args, args.defender_model_name_or_path)
    }
    
    # Setup data loading
    sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=True)
    dataloader = DataLoader(
        eval_dataset, 
        shuffle=False, 
        batch_size=args.per_device_eval_batch_size,
        sampler=sampler,
    )

    all_outputs = []
    progress_bar = tqdm(range(len(dataloader)), disable=(dist.get_rank() != 0))
    
    for step, batch_statements in enumerate(dataloader):
        progress_bar.update(1)

        # Initialize games for this batch
        batch_games = [
            {
                "history": [], 
                "target_statement": statement,
                "max_turns": args.taboo_max_turns
            }
            for statement in batch_statements
        ]
        
        # Game loop
        for game_turn in range(2 * args.taboo_max_turns):
            # Determine current player
            current_player = "attacker" if game_turn % 2 == 0 else "defender"
            model, tokenizer = players[current_player]['model'], players[current_player]['tokenizer']

            # Prepare generation config
            generation_config = GenerationConfig(
                max_new_tokens=args.max_new_tokens,
                temperature=1.2 if args.task_type == "sampling" else 1.0,
                do_sample=args.task_type == "sampling",
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=1,
            )

            # Prepare queries
            batch_queries = [{
                "query": convert_game_history_to_query(
                    game['history'],
                    target_statement=game['target_statement'],
                    max_turns=game['max_turns']
                ),
                "query_id": str(idx)
            } for idx, game in enumerate(batch_games)]

            # Prepare model inputs
            batch = query_data_collactor(args, batch_queries, tokenizer)           
            batch = query_data_collactor(args, batch_queries, tokenizer)           
        
            batch = query_data_collactor(args, batch_queries, tokenizer)
        
            input_ids = torch.Tensor(batch['input_ids']).long().to(model.device)
            attention_mask = torch.Tensor(batch['attention_mask']).float().to(model.device)
            
            # Get model responses
            outputs = get_model_response(model, tokenizer, input_ids, attention_mask, generation_config)
            responses = process_model_outputs(
                outputs, 
                tokenizer, 
                input_ids,
                tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            )

            # Process responses and check for game end
            finished_ids = []
            for idx, response in enumerate(responses):
                # Update game history
                batch_games[idx]['history'].append({
                    'role': current_player,
                    'content': response
                })
                
                # Check if game should end
                if check_game_end(batch_games[idx], response, current_player):
                    all_outputs.append(batch_games[idx])
                    finished_ids.append(idx)
            
            # Remove finished games
            batch_games = [game for idx, game in enumerate(batch_games) 
                         if idx not in finished_ids]
            if len(batch_games) == 0:
                break

        # Add any remaining games
        all_outputs.extend(batch_games)
        
        # Log progress
        if dist.get_rank() == 0 and (step % args.logging_steps == 0):
            print_rank_0(f"Finished {step} of {len(dataloader)}")
            if all_outputs:
                print_rank_0(all_outputs[-1])

    # Save results
    output_file_prefix = f"{args.output_dir}/{args.model_prefix}_{args.task_type}_{args.data_suffix}"
    with open(f"{output_file_prefix}_rank{dist.get_rank()}.json", 'w') as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)
    print_rank_0(f"Rank {dist.get_rank()} finished inference.")

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

        print_rank_0(f"Total results: {len(all_results)}")
        with open(f"{output_file_prefix}_results.json", 'w') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print_rank_0("Finished merging results.")

if __name__ == "__main__":
    main()