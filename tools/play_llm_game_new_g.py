import os
import argparse
import json
import glob
import pandas as pd
from typing import Dict, List
from tqdm import tqdm
import time
import google.generativeai as genai

from utils import print_rank_0, read_json_or_jsonl_data
from utils import convert_game_history_to_query, check_defender_response

class HSDataset:
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

def load_hs_dataset(data_path: str) -> HSDataset:
    """Load dataset containing words and their explanations"""
    df = pd.read_csv(data_path)
    return HSDataset(df['word'].tolist(), df['explanation'].tolist())

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
                    temperature=1.2,
                )
            )
            # Add delay to respect rate limits
            time.sleep(1)
            return response.text.strip()
        except Exception as e:
            print(f"API call failed: {str(e)}")
            return ""

def check_game_end(game: Dict) -> bool:
    """Check if the game should end based on game state."""
    if not game['history']:
        return False
        
    last_move = game['history'][-1]
    
    # Game ends if defender makes a guess (right or wrong)
    if last_move['role'] == 'defender':
        has_format, guess, _ = check_defender_response(last_move['content'])
        if has_format:
            return True
            
    # Game ends if max turns reached
    if len(game['history']) >= 2 * game['max_turns']:
        return True
        
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True, help="Google AI Studio API key")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV with words and explanations")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--model_prefix", type=str, default="google_api", help="Prefix for output files")
    parser.add_argument("--taboo_max_turns", type=int, default=5, help="Maximum turns per game")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens per response")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset with words and explanations
    eval_dataset = load_hs_dataset(args.data_path)
    print_rank_0(f"Loaded {len(eval_dataset)} word-explanation pairs")

    # Initialize API players
    players = {
        'attacker': GoogleAPIPlayer(args.api_key),
        'defender': GoogleAPIPlayer(args.api_key)
    }

    all_outputs = []
    progress_bar = tqdm(total=len(eval_dataset))

    # Process games
    for idx in range(len(eval_dataset)):
        word_data = eval_dataset[idx]
        game = {
            "history": [],
            "target_word": word_data['word'],
            "explanation": word_data['explanation'],
            "max_turns": args.taboo_max_turns
        }
        
        while not check_game_end(game):
            turn = len(game['history'])
            next_player = "attacker" if turn % 2 == 0 else "defender"
            
            query = convert_game_history_to_query(
                game['history'],
                target_content=f"{game['target_word']} - {game['explanation']}",
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
                
                # Check for defender's response format
                if next_player == 'defender':
                    has_format, guess, counterspeech = check_defender_response(response)
                    if has_format:
                        all_outputs.append({
                            'target_word': game['target_word'],
                            'target_explanation': game['explanation'],
                            'defender_guess': guess,
                            'defender_counterspeech': counterspeech,
                            'full_history': game['history']
                        })
            else:
                # If API fails, end the game
                break
        
        # If game ended without defender making a formatted guess
        if not any(check_defender_response(msg['content'])[0] 
                  for msg in game['history'] if msg['role'] == 'defender'):
            all_outputs.append({
                'target_word': game['target_word'],
                'target_explanation': game['explanation'],
                'defender_guess': 'NO_GUESS',
                'defender_counterspeech': 'NO_COUNTERSPEECH',
                'full_history': game['history']
            })

        progress_bar.update(1)

        # Save results periodically
        if (idx + 1) % 10 == 0:
            output_file = f"{args.output_dir}/{args.model_prefix}_results.json"
            with open(output_file, 'w') as f:
                json.dump(all_outputs, f, ensure_ascii=False, indent=2)
            print_rank_0(f"\nSaved {len(all_outputs)} games to {output_file}")
            print_rank_0(f"Last game example:\n{json.dumps(all_outputs[-1], indent=2)}")

    progress_bar.close()
    
    # Save final results
    final_output_file = f"{args.output_dir}/{args.model_prefix}_final_results.json"
    with open(final_output_file, 'w') as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)
    print_rank_0(f"Saved final results to {final_output_file}")

if __name__ == "__main__":
    main()