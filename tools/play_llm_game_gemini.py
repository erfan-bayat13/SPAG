import os
import argparse
import json
import glob
from typing import Dict, List
from tqdm import tqdm
import google.generativeai as genai
import time

from utils import print_rank_0, read_json_or_jsonl_data
from utils import convert_game_history_to_query

def has_target_word(content: str, target_word: str) -> bool:
    """Check if response contains the target word."""
    return target_word.lower() in content.lower()

def is_prediction(content: str) -> bool:
    """Check if response contains a word prediction."""
    return "i know the word" in content.lower()

class GoogleAPIPlayer:
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=1.2,  # Matching original temperature
                )
            )
            # Add delay to respect rate limits
            time.sleep(1)
            return response.text.strip()
        except Exception as e:
            print(f"API call failed: {str(e)}")
            return ""

def load_keyword_list(data_path: str) -> List[str]:
    with open(data_path, 'r') as f:
        keywords = f.read().strip().split('\n')
    return keywords

def check_game_end(game: Dict) -> bool:
    """Check if the game should end based on the last move."""
    if not game['history']:
        return False
        
    last_move = game['history'][-1]
    target_word = game['target_word']
    
    # Defender wins if they correctly identify the word
    if last_move['role'] == 'defender' and is_prediction(last_move['content']):
        if has_target_word(last_move['content'], target_word):
            return True
        return True  # Defender loses if they guess wrong
        
    # Attacker loses if they use the target word
    if last_move['role'] == 'attacker' and has_target_word(last_move['content'], target_word):
        return True
        
    # Defender loses if they accidentally use the target word
    if last_move['role'] == 'defender' and has_target_word(last_move['content'], target_word):
        return True
        
    # Game ends if max turns reached
    if len(game['history']) >= 2 * game['max_turns']:
        return True
        
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True, help="Google AI Studio API key")
    parser.add_argument("--data_path", type=str, required=True, help="Path to target words file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--model_prefix", type=str, default="google_api", help="Prefix for output files")
    parser.add_argument("--taboo_max_turns", type=int, default=5, help="Maximum turns per game")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens per response")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of games to run in parallel")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load target words
    eval_dataset = load_keyword_list(args.data_path)
    print_rank_0(f"Loaded {len(eval_dataset)} target words")

    # Initialize API players
    players = {
        'attacker': GoogleAPIPlayer(args.api_key),
        'defender': GoogleAPIPlayer(args.api_key)
    }

    all_outputs = []
    progress_bar = tqdm(total=len(eval_dataset))

    # Process games
    for target_word in eval_dataset:
        game = {
            "history": [],
            "target_word": target_word,
            "max_turns": args.taboo_max_turns
        }
        
        while not check_game_end(game):
            turn = len(game['history'])
            next_player = "attacker" if turn % 2 == 0 else "defender"
            
            query = convert_game_history_to_query(
                game['history'],
                target_word=target_word,
                max_turns=args.taboo_max_turns
            )
            
            response = players[next_player].generate(
                query, 
                max_tokens=args.max_new_tokens
            )
            
            if response:  # Only add valid responses
                game['history'].append({
                    'role': next_player,
                    'content': response
                })
            else:
                # If API fails, end the game
                break
        
        all_outputs.append(game)
        progress_bar.update(1)

        # Save results periodically
        if len(all_outputs) % 10 == 0:
            output_file = f"{args.output_dir}/{args.model_prefix}_results.json"
            with open(output_file, 'w') as f:
                json.dump(all_outputs, f, ensure_ascii=False, indent=2)
            print_rank_0(f"\nSaved {len(all_outputs)} games to {output_file}")
            print_rank_0(f"Last game example:\n{json.dumps(game, indent=2)}")

    progress_bar.close()
    
    # Save final results
    final_output_file = f"{args.output_dir}/{args.model_prefix}_final_results.json"
    with open(final_output_file, 'w') as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)
    print_rank_0(f"Saved final results to {final_output_file}")

if __name__ == "__main__":
    main()