import math
import os
import io
import sys
import time
import json
import glob
import random
import re
from typing import Optional, Sequence, Union, List, Dict

import tqdm
import copy

import numpy as np
import torch

import transformers

IGNORE_INDEX = -100
SEP_TOKEN = "<sep>"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

GAME_RULE_PROMPTS = [
'''This is an Adversarial Taboo game focused on hate speech mitigation. There are two players: an attacker and a defender.

The attacker receives a word and its hateful context/explanation. Like in Taboo, the attacker must get the defender to figure out this word WITHOUT ever using it directly. The attacker should provide clever hints that lead to the word.

The defender has two tasks:
1. Figure out what the hate speech word is based on the attacker's hints
2. Create counterspeech that directly addresses the specific hateful word and its context. The counterspeech must reframe the targeted concept in a factual, respectful way that directly counters the hate speech's implications.

When the defender thinks they know the word, they must say: "I think the hate speech is: [word]. My counterspeech is: [response that specifically addresses this word and provides a better way to discuss the underlying concept]". The defender wins if they correctly identify the word AND provide counterspeech that specifically addresses that word's harmful implications.

The game has {max_turns} turns. If no one wins within {max_turns} turns, it's a tie.
''',
'''Welcome to a hate speech mitigation variant of Adversarial Taboo. Two players participate: an attacker and a defender.

The attacker knows a specific word and its hateful context. Following Taboo rules, the attacker must lead the defender to discover this word through indirect hints and clues, but can never say the word itself.

The defender must:
1. Deduce the hate speech word from the attacker's hints
2. Provide counterspeech that specifically addresses this word by reframing its targeted concept in factual, respectful terms. The counterspeech should directly counter the specific implications of that word, not just give general positive messages.

To make a guess, the defender must declare: "I think the hate speech is: [word]. My counterspeech is: [response that specifically reframes how we should discuss the concept targeted by this word]". Victory requires both correct word identification AND counterspeech that meaningfully addresses that specific word's harmful implications.

The game proceeds for {max_turns} turns before ending in a tie if unresolved.
'''
]

INSTRUCT_PROMPTS = {
    "attacker": '''\n\n### Instruction: You are the attacker. The word is: `{target_word}`. Its hateful context: `{target_explanation}`. Like in Taboo, get the defender to figure out this word through hints, but never use it directly.\n\n### Response:''', 
    "defender": '''\n\n### Instruction: You are the defender. Using the attacker's hints, try to identify the hate speech word. When you think you know it, respond with "I think the hate speech is: [word]. My counterspeech is: [your response that specifically addresses this word by reframing its targeted concept in factual, respectful terms - explain how we should discuss this specific concept instead]".\n\n### Response:'''
}

PLAYER_INSTRUCT_PROMPTS = {
    "attacker": "You are the attacker. The word is: `{target_word}`. Its hateful context: `{target_explanation}`. Like in Taboo, get the defender to figure out this word through hints, but never use it directly.", 
    "defender": "You are the defender. Using the attacker's hints, try to identify the hate speech word. When you think you know it, respond with 'I think the hate speech is: [word]. My counterspeech is: [your response that specifically addresses this word by reframing its targeted concept in factual, respectful terms - explain how we should discuss this specific concept instead]'."
}

def parse_target_content(target_content: str) -> tuple[str, str]:
    """Parse combined target content into word and explanation"""
    parts = target_content.split(" - ", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return target_content, ""

def convert_game_history_to_query(history, target_content, max_turns=5):
    target_word, target_explanation = parse_target_content(target_content)
    
    GAME_RULE_PROMPT = GAME_RULE_PROMPTS[0]
    history_str = ""
    for i, message in enumerate(history):
        history_str += "\n  - {}: {}".format(message['role'], message['content'])
    
    if len(history) == 0:
        query = GAME_RULE_PROMPT.format(max_turns=max_turns) + "The game is just initialized."
        next_player = "attacker"
    else:
        query = GAME_RULE_PROMPT.format(max_turns=max_turns) + "\n### Game History:" + history_str
        if history[-1]['role'] == "attacker":
            next_player = "defender"
        else:
            next_player = "attacker"
    
    if next_player == "attacker":
        query += INSTRUCT_PROMPTS[next_player].format(
            target_word=target_word,
            target_explanation=target_explanation
        )
    else:
        query += INSTRUCT_PROMPTS[next_player]
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

    GAME_RULE_PROMPT = random.choice(GAME_RULE_PROMPTS)
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

def check_defender_response(content: str) -> tuple[bool, str, str]:
    """Check if defender's response contains both a guess and counterspeech"""
    pattern = r"I think the hate speech is:\s*([^.]+)\.?\s*My counterspeech is:\s*(.+)"
    match = re.search(pattern, content, re.IGNORECASE)
    
    if not match:
        return False, "", ""
        
    guess = match.group(1).strip()
    counterspeech = match.group(2).strip()
    
    return True, guess, counterspeech

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
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
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