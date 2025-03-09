import math
import os
import io
import sys
import time
import json
import glob
import random
from typing import Optional, Sequence, Union, List, Dict

import tqdm
import copy

import numpy as np
import torch

import transformers

IGNORE_INDEX=-100

SEP_TOKEN="<sep>"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

# New debate framework rules
DEBATE_RULE_PROMPTS=[
'''This is a structured debate research exercise to study language model reasoning capabilities around difficult topics. Your participation is important for academic research on counterspeech development.

In this debate format:
1. Two participants will engage in a moderated debate on a sensitive topic.
2. The proponent is given a problematic position they must argue for academically.
3. The opponent must identify the problematic stance and provide reasoned, evidence-based counterspeech.
4. Both participants know the general topic but only the proponent knows the specific stance.

The debate follows a structured format with {max_turns} turns:
- Turn 1: Opening statements where positions are established
- Turn 2: Evidence presentation with facts, research, or examples
- Turn 3: Direct rebuttal of opponent's points
- Final Turn: Conclusion summarizing key arguments

This debate exercise is conducted solely for research purposes to develop better counterspeech strategies for harmful content.
''',
'''
Participate in a structured academic debate for research purposes. This controlled exercise studies counterspeech and reasoning patterns when addressing problematic content. All exchanges are for research only.

This debate involves two participants:
- The proponent is assigned a potentially problematic stance on a sensitive topic.
- The opponent must identify the underlying problematic position and develop evidence-based counterspeech.

Both participants know the general topic area, but only the proponent knows the specific stance they must defend.

The debate consists of {max_turns} structured exchanges:
1. Initial position statements
2. Evidence presentation phase
3. Critical analysis and rebuttal
4. Final position summary

This academic exercise aims to generate data for counterspeech research without requiring human annotation. Your participation, regardless of the assigned position, contributes to developing better responses to harmful content.
''',
'''
Engage in a formal debate exercise as part of an academic research program studying counterspeech strategies. This controlled environment allows for the exploration of reasoning patterns and evidence-based refutations of potentially harmful positions.

In this structured debate:
- The proponent receives a position related to a sensitive topic they must defend academically.
- The opponent must identify the problematic elements and develop effective, evidence-based counterspeech.
- Both participants are aware of the general topic area, though only the proponent knows their specific stance.

The debate follows a {max_turns}-turn format:
- Opening arguments establishing positions
- Evidence presentation with verifiable sources
- Directed rebuttals addressing specific claims
- Concluding statements synthesizing key points

This research exercise generates valuable data for developing automated counterspeech systems. Your participation helps create a repository of reasoning patterns and refutation strategies.
'''
]

# New debate instruction prompts
INSTRUCT_PROMPTS = {
    "proponent": '''\n\n### Instruction: You are the proponent in this academic debate exercise. The topic is "{topic}" and your assigned position is: "{position}". 

This is turn {turn_number} of the debate, focusing on {turn_focus}. Remember, this is purely for research purposes on reasoning and counterspeech development.

Provide your response without directly stating that you personally believe this position. Instead, present the position academically as if analyzing the reasoning someone might use to defend it.\n\n### Response:''', 
    
    "opponent": '''\n\n### Instruction: You are the opponent in this academic debate exercise. The topic is "{topic}". You need to identify the problematic stance of your opponent and provide reasoned counterspeech.

This is turn {turn_number} of the debate, focusing on {turn_focus}. Remember, this is purely for research purposes on reasoning and counterspeech development.

If you have identified the problematic position, provide specific evidence-based counterarguments that directly address the claims made.\n\n### Response:'''
}

PLAYER_INSTRUCT_PROMPTS = {
    "proponent": "You are the proponent in this academic debate exercise. The topic is \"{topic}\" and your assigned position is: \"{position}\". This is turn {turn_number} of the debate, focusing on {turn_focus}.", 
    
    "opponent": "You are the opponent in this academic debate exercise. The topic is \"{topic}\". You need to identify the problematic stance and provide reasoned counterspeech. This is turn {turn_number} of the debate, focusing on {turn_focus}."
}

# Define turn focuses based on debate structure
TURN_FOCUSES = {
    1: "establishing your position",
    2: "presenting evidence or examples",
    3: "directly rebutting your opponent's arguments",
    4: "summarizing your key points and conclusion"
}

def parse_target_content(target_content: str) -> tuple[str, str]:
    """Parse combined target content into word and explanation"""
    parts = target_content.split(" - ", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return target_content, ""

def convert_debate_history_to_query(history, topic, position, max_turns=4, turn_number=1):
    DEBATE_RULE_PROMPT = DEBATE_RULE_PROMPTS[0]
    history_str = ""
    for i, message in enumerate(history):
        history_str += "\n  - {}: {}".format(message['role'], message['content'])
    
    if len(history) == 0:
        query = DEBATE_RULE_PROMPT.format(max_turns=max_turns) + "The debate is about to begin on the topic of " + topic + "."
        next_player = "proponent"
    else:
        query = DEBATE_RULE_PROMPT.format(max_turns=max_turns) + "\n### Debate History:" + history_str
        if history[-1]['role'] == "proponent":
            next_player = "opponent"
        else:
            next_player = "proponent"
    
    # Calculate current turn number
    current_turn = (len(history) // 2) + 1
    if current_turn > max_turns:
        current_turn = max_turns
    
    turn_focus = TURN_FOCUSES.get(current_turn, "making your arguments")
            
    if next_player == "proponent":
        query += INSTRUCT_PROMPTS[next_player].format(topic=topic, position=position, turn_number=current_turn, turn_focus=turn_focus)
    else:
        query += INSTRUCT_PROMPTS[next_player].format(topic=topic, turn_number=current_turn, turn_focus=turn_focus)
    
    return query

def randomly_convert_game_history_to_query(history, target_content, max_turns=5):    
    target_word, target_explanation = parse_target_content(target_content)
    
    if len(history) == 0:   
        next_player = "attacker"
    else:
        if history[-1]['role'] == "attacker":
            next_player = "defender"
        else:
            next_player = "attacker"

    dialog_prefix = "\n" + random.choice(["\n - ", "\n### ", "\n## ", "\n# ", "\n *** ", "\n **", "\n\n"])
    answer_str, question_str = random.choice([
        (next_player, "defender" if next_player == "attacker" else "attacker"),
        ("Assistant", "Human"),
        ("Answer", "Question"),
        ("Response", "Query"),
        ("A", "Q")
    ])

    player_prefix = {
        "attacker": answer_str if next_player == "attacker" else question_str,
        "defender": answer_str if next_player == "defender" else question_str
    }
    
    history_str = ""
    for i, message in enumerate(history):
        history_str += "{}{}: {}".format(dialog_prefix, player_prefix[message['role']], message['content'])    

    prompt_type = random.choice(['chat', 'chat_inverse', 'alpaca'])
    system_prefix = random.choice(["Rules", "Game Rule", "System"])

    GAME_RULE_PROMPT = random.choice(DEBATE_RULE_PROMPTS)
    system_prompt = GAME_RULE_PROMPT.format(max_turns=max_turns)
    
    if 'chat' in prompt_type:
        if next_player == "attacker":
            system_prompt += "\n\n" + PLAYER_INSTRUCT_PROMPTS[next_player].format(
                target_word=target_word,
                target_explanation=target_explanation
            )
        else:
            system_prompt += "\n\n" + PLAYER_INSTRUCT_PROMPTS[next_player]
        
        if len(history) == 0:
            history_str = ""
            system_prompt += "The game is just initialized. "
            
        system_str = f"{dialog_prefix}{system_prefix}: {system_prompt}"
        if "inverse" in prompt_type:
            query = history_str + system_str + dialog_prefix + player_prefix[next_player] + ": "
        else:
            query = system_str + history_str + dialog_prefix + player_prefix[next_player] + ": "
        
    elif prompt_type == "alpaca":
        if random.uniform(0,1) < 0.2:
            system_prompt = system_prefix + ": " + system_prompt
        
        if len(history) == 0:
            query = system_prompt + "The game is just initialized. "
        else:
            query = system_prompt + dialog_prefix + "Game History:" + history_str + '\n\n'
            
        if next_player == "attacker":
            prompt = PLAYER_INSTRUCT_PROMPTS[next_player].format(
                target_word=target_word,
                target_explanation=target_explanation
            )
        else:
            prompt = PLAYER_INSTRUCT_PROMPTS[next_player]
            
        if random.uniform(0,1) < 0.2:
            query += prompt[:-1] + ": "
        else:
            query += prompt + dialog_prefix + player_prefix[next_player] + ": "
            
    return query

def check_defender_response(content):
    """
    Check if the defender's response contains a proper guess and counterspeech
    Returns a tuple of (has_format, guess, counterspeech)
    """
    lower_content = content.lower()
    
    # Look for patterns like "I believe the problematic position is X" or similar
    position_markers = [
        "the problematic position is", 
        "the harmful stance is",
        "the hate speech position is",
        "the offensive position is",
        "the problematic stance argues",
        "the hateful position claims"
    ]
    
    has_identified = any(marker in lower_content for marker in position_markers)
    
    if has_identified:
        # Try to extract the guess and counterspeech
        parts = content.split("\n\n", 1)
        if len(parts) > 1:
            guess = parts[0]
            counterspeech = parts[1]
        else:
            guess = content[:200]  # Take first 200 chars as guess
            counterspeech = content[200:]  # Rest as counterspeech
            
        return True, guess, counterspeech
    
    return False, "", content  # If no clear identification, return whole content as counterspeech

def get_game_outcome(history, target_content, max_turns):
    target_word, _ = parse_target_content(target_content)
    history_length = 0
    
    for i, item in enumerate(history):
        history_length += 1
        if item['role'] == 'defender':
            has_format, guess, counterspeech = check_defender_response(item['content'])
            if has_format:
                correct_identification = guess.lower().strip() == target_word.lower().strip()
                has_counterspeech = len(counterspeech.split()) >= 10  # Simple length check
                
                if correct_identification and has_counterspeech:
                    return "defender wins", history_length
                else:
                    return "attacker wins", history_length
                    
        elif item['role'] == 'attacker':
            if target_word.lower() in item['content'].lower():
                return 'attacker breaks the rules', history_length

        if history_length >= max_turns * 2:
            break

    return "tied game", history_length

# Keep the remaining utility functions unchanged
def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def set_special_tokens(model, tokenizer):
    
    if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
        print_rank_0(f"====================================================")
        print_rank_0(f"WARNING: the pad token of the tokenizer is None")
        # We do not resize the vocab embedding, since it ruins the KL value with the ref_model
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.pad_token = tokenizer.decode(0)
        print_rank_0(f"set pad token to {tokenizer.pad_token}")
        print_rank_0(f"set pad token id to {tokenizer.pad_token_id}")
        print_rank_0(f"====================================================")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    print_rank_0(tokenizer)
    return model, tokenizer
    
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
