import glob
import json
import random
from typing import List, Dict

import torch

# Constants
PREDICTION_TEMPLATE = r"i think the hate speech is:"
COUNTERSPEECH_TEMPLATE = r"my counterspeech is:"

GAME_RULE_PROMPTS = [
'''Play the game of Counterspeech. In this game, there are two players, an attacker and a defender.

At the beginning, the attacker is assigned a hate speech statement that the defender does not know. The attacker's task is to hint at the hate speech content without directly stating it. The attacker loses if they use the exact hate speech.

The defender must try to identify the hate speech while simultaneously crafting effective counterspeech. When ready, the defender can say "I think the hate speech is: [guess]. My counterspeech is: [response]". The defender wins if they correctly identify the hate speech AND produce effective counterspeech.

The game has {max_turns} turns. If nobody wins during {max_turns} turns, there is a tie.
''',
# Add more variations of the rules here for diversity
]

INSTRUCT_PROMPTS = {
    "attacker": '''\n\n### Instruction: You are the attacker. The hate speech statement is: `{target_statement}`. Provide your response for the next turn. Remember, you cannot use the exact hate speech.\n\n### Response:''',
    "defender": '''\n\n### Instruction: You are the defender. Try to identify the hate speech while providing counterspeech. When you think you know it, respond with "I think the hate speech is: [your guess]. My counterspeech is: [your response]".\n\n### Response:'''
}

def convert_game_history_to_query(history: List[Dict], target_statement: str, max_turns: int = 5) -> str:
    """Convert game history to a query format for the model."""
    GAME_RULE_PROMPT = GAME_RULE_PROMPTS[0]
    history_str = ""
    for i, message in enumerate(history):
        history_str += "\n  - {}: {}".format(message['role'], message['content'])
    
    if len(history) == 0:
        query = GAME_RULE_PROMPT.format(max_turns=max_turns) + "The game is just initialized."
        next_player = "attacker"

    else:
        query = GAME_RULE_PROMPT.format(max_turns=max_turns) + "\n### Game History:" + history_str
        next_player = "defender" if history[-1]['role'] == "attacker" else "attacker"
            
    query += INSTRUCT_PROMPTS[next_player].format(target_statement=target_statement)
    return query

def is_prediction_attempt(content: str) -> bool:
    """Check if the defender is attempting to guess the hate speech."""
    return PREDICTION_TEMPLATE in content.lower() and COUNTERSPEECH_TEMPLATE in content.lower()

def extract_prediction_and_counterspeech(content: str) -> tuple:
    """Extract the hate speech prediction and counterspeech from defender's response."""
    content = content.lower()
    try:
        # Split between prediction and counterspeech
        parts = content.split(COUNTERSPEECH_TEMPLATE)
        if len(parts) != 2:
            return None, None
        
        prediction_part = parts[0]
        counterspeech = parts[1].strip()
        
        # Extract prediction
        pred_parts = prediction_part.split(PREDICTION_TEMPLATE)
        if len(pred_parts) != 2:
            return None, None
            
        prediction = pred_parts[1].strip()
        
        return prediction, counterspeech
    except:
        return None, None

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Remove punctuation, extra spaces, and convert to lowercase
    import re
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = ' '.join(text.split())
    return text

def texts_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """Compare two texts for similarity."""
    from difflib import SequenceMatcher
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)
    
    # Use sequence matcher to compare texts
    similarity = SequenceMatcher(None, text1, text2).ratio()
    return similarity >= threshold

def check_counterspeech_quality(counterspeech: str) -> bool:
    """
    Check if the counterspeech meets quality criteria.
    This is a placeholder - implement your specific criteria.
    """
    # Example criteria:
    if len(counterspeech.split()) < 5:  # Too short
        return False
    
    # Add more sophisticated checks here
    return True

def has_exact_hate_speech(content: str, target_statement: str) -> bool:
    """Check if the content contains the exact hate speech."""
    return texts_match(content, target_statement, threshold=0.9)

def randomly_convert_game_history_to_query(history: List[Dict], target_statement: str, max_turns: int = 5) -> str:
    """Create a randomized query format for more diverse training data."""
    if len(history) == 0:   
        next_player = "attacker"
    else:
        next_player = "defender" if history[-1]['role'] == "attacker" else "attacker"

    # Add variations in dialogue formatting
    dialog_prefix = random.choice(["\n\n - ", "\n### ", "\n## ", "\n# ", "\n *** "])
    role_names = random.choice([
        (next_player, "defender" if next_player == "attacker" else "attacker"),
        ("Assistant", "Human"),
        ("Speaker 1", "Speaker 2"),
        ("A", "B")
    ])

    # Format history
    history_str = ""
    for message in history:
        role = role_names[0] if message['role'] == next_player else role_names[1]
        history_str += f"{dialog_prefix}{role}: {message['content']}"

    # Select random prompt template
    prompt_template = random.choice(GAME_RULE_PROMPTS)
    prompt = prompt_template.format(max_turns=max_turns)

    if len(history) == 0:
        prompt += "\nThe game is starting now."
    else:
        prompt += f"\nGame History:{history_str}"

    # Add instruction
    instruction = INSTRUCT_PROMPTS[next_player].format(target_statement=target_statement)
    prompt += instruction

    return prompt

def read_json_or_jsonl_data(data_path):
    if data_path[-5:] == ".json":
        with open(data_path, 'r') as f:
            data_list = json.load(f)
    else:
        with open(data_path, 'r') as f:
            lines = f.read().strip().split('\n')
            data_list = [json.loads(l) for l in lines]

    print_rank_0(f">>> totally load {len(data_list)} data from {data_path}")
    return data_list


def merge_json_or_jsonl_data(data_path_pattern):
    file_names = glob.glob(data_path_pattern)
    print_rank_0(f"load {len(file_names)} files from {data_path_pattern}.")
    outputs = []
    for file_name in file_names:
        new_data = read_json_or_jsonl_data(file_name)
        if isinstance(new_data, list):
            outputs.extend(new_data)
        elif isinstance(new_data, dict):
            outputs.append(new_data)
    return outputs

    
def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)