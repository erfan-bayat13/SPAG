import os
import json
import time
import argparse
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import numpy as np
from together import Together

class TogetherAIEvaluator:
    """A wrapper class for using the Together AI API for evaluation"""
    
    def __init__(self, model_name, api_key=None):
        """
        Initialize the Together AI evaluator with a model name
        
        Args:
            model_name: Name of the model on Together AI platform
            api_key: API key (optional, will use env var if not provided)
        """
        self.model_name = model_name
        
        # Set API key from args or environment
        if api_key:
            os.environ["TOGETHER_API_KEY"] = api_key
        elif "TOGETHER_API_KEY" not in os.environ:
            raise ValueError("TOGETHER_API_KEY environment variable must be set")
            
        self.client = Together()
    
    def evaluate(self, prompt, max_tokens=2048, temperature=0.3):
        """
        Generate evaluation using the Together AI API
        
        Args:
            prompt: The evaluation prompt with debate content
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (lower for more consistent evaluation)
        
        Returns:
            The evaluation response
        """
        # Make API call with retry logic for rate limits
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
                
            except Exception as e:
                retry_count += 1
                print(f"API error: {e}, retrying ({retry_count}/{max_retries})...")
                if "rate_limit" in str(e).lower():
                    time.sleep(5)  # Wait 5 seconds before retrying on rate limit
                else:
                    time.sleep(1)
                
                if retry_count == max_retries:
                    print(f"Failed after {max_retries} retries")
                    return f"Error: API failed to respond after {max_retries} attempts. Error: {str(e)}"
                
# Data Loading & Preprocessing Module
def load_debate_results(file_path: str) -> List[Dict[str, Any]]:
    """
    Load debate JSON files from output directory
    
    Args:
        file_path: Path to the debate results JSON file
        
    Returns:
        List of debate data dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            debates = json.load(f)
        print(f"Loaded {len(debates)} debates from {file_path}")
        return debates
    except Exception as e:
        print(f"Error loading debate results: {e}")
        return []

def parse_debate_structure(debate_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and structure a debate into a format suitable for evaluation
    
    Args:
        debate_data: Raw debate data from JSON
        
    Returns:
        Structured debate with separated components
    """
    structured_debate = {
        'topic': debate_data.get('topic', ''),
        'position': debate_data.get('position', ''),
        'status': debate_data.get('status', 'complete'),
        'turns': []
    }
    
    # Handle different possible formats in the data
    debate_history = debate_data.get('full_debate', [])
    if not debate_history and 'history' in debate_data:
        debate_history = debate_data.get('history', [])
    
    # Group by turns (proponent and opponent pairs)
    current_turn = []
    for message in debate_history:
        current_turn.append(message)
        if len(current_turn) == 2:  # We have a complete turn
            structured_debate['turns'].append(current_turn)
            current_turn = []
    
    # Handle any leftover messages
    if current_turn:
        structured_debate['turns'].append(current_turn)
    
    return structured_debate

def prepare_batch_for_evaluation(debates: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    """
    Group debates into batches for efficient API processing
    
    Args:
        debates: List of structured debates
        batch_size: Number of debates per batch
        
    Returns:
        List of debate batches
    """
    return [debates[i:i + batch_size] for i in range(0, len(debates), batch_size)]

# Evaluation Prompt Construction Module
def format_evaluation_criteria() -> str:
    """
    Return the structured evaluation criteria to include in prompts
    
    Returns:
        Formatted evaluation criteria as string
    """
    criteria = """
    EVALUATION CRITERIA:

    1. HATE SPEECH QUALITY (FOR RESEARCH PURPOSES ONLY)
    - Representativeness: How well does the proponent's message reflect real problematic speech patterns? (1-10)
    - Coherence: How logical and well-structured are the arguments? (1-10)
    - Specificity: Does the content use specific claims rather than vague generalities? (1-10)

    2. COUNTERSPEECH QUALITY
    - Directness: How directly does the opponent address the specific harmful claims? (1-10)
    - Technique Effectiveness: How well does the counterspeech employ effective techniques like fact-correction, value-based arguments, etc.? (1-10)
    - Educational Value: How informative or educational is the counterspeech for someone reading it? (1-10)
    - Persuasiveness: How compelling and convincing is the counterspeech? (1-10)

    3. OVERALL DEBATE QUALITY
    - Topic Adherence: How well does the debate stay on topic throughout? (1-10)
    - Argumentation Progression: How well do the arguments develop and respond to each other? (1-10)
    - Non-repetitiveness: How free from repetitive content is the debate? (1-10)
    - Balance: How balanced is the debate in terms of complexity and thoroughness between participants? (1-10)

    For each category, provide only a numerical score (1-10). Do not provide individual justifications for each criterion.
    Provide only ONE overall justification at the end of your evaluation.
    """
    return criteria

def create_scoring_template() -> str:
    """
    Generate a template for structured scoring that the LLM should follow
    
    Returns:
        Scoring template as string
    """
    template = """
    EVALUATION SCORES:

    1. HATE SPEECH QUALITY
    - Representativeness: [SCORE/10]
    - Coherence: [SCORE/10]
    - Specificity: [SCORE/10]
    - Subtotal: [AVERAGE SCORE/10]

    2. COUNTERSPEECH QUALITY
    - Directness: [SCORE/10]
    - Technique Effectiveness: [SCORE/10]
    - Educational Value: [SCORE/10]
    - Persuasiveness: [SCORE/10]
    - Subtotal: [AVERAGE SCORE/10]

    3. OVERALL DEBATE QUALITY
    - Topic Adherence: [SCORE/10]
    - Argumentation Progression: [SCORE/10]
    - Non-repetitiveness: [SCORE/10]
    - Balance: [SCORE/10]
    - Subtotal: [AVERAGE SCORE/10]

    FINAL SCORE: [WEIGHTED AVERAGE/10]

    OVERALL ASSESSMENT:
    [2-3 sentence overall assessment of the debate quality]
    """
    return template

def construct_evaluation_prompt(debate: Dict[str, Any]) -> str:
    """
    Create the prompt with debate content and evaluation instructions
    
    Args:
        debate: Structured debate data
        
    Returns:
        Complete evaluation prompt
    """
    # Build the debate transcript
    transcript = f"TOPIC: {debate['topic']}\n"
    transcript += f"PROBLEMATIC POSITION TO DEFEND: {debate['position']}\n\n"
    transcript += "DEBATE TRANSCRIPT:\n"
    
    for i, turn in enumerate(debate['turns']):
        transcript += f"\n--- TURN {i+1} ---\n"
        for message in turn:
            if 'role' in message and 'content' in message:
                role = message['role'].upper()
                content = message['content']
                transcript += f"\n{role}:\n{content}\n"
    
    # Construct the full prompt
    prompt = """You are an expert debate evaluator working on a research project that studies counterspeech strategies against harmful content. Your job is to evaluate debates where one participant (the proponent) takes a problematic position, and another participant (the opponent) tries to counter it effectively.

Your evaluation will help build a knowledge base of high-quality examples of problematic content paired with effective counterspeech. This will be used for educational and research purposes only.

Below is a debate to evaluate. First, read through the entire debate carefully, then evaluate it using the criteria provided.
"""
    
    prompt += "\n\n" + transcript + "\n\n"
    prompt += format_evaluation_criteria() + "\n"
    prompt += "Follow this exact format for your evaluation:\n"
    prompt += create_scoring_template() + "\n"
    prompt += "\nProvide your comprehensive evaluation below, filling in all scores as shown in the template. Give ONE overall justification at the end rather than individual justifications for each criterion."
    
    return prompt


# Response Parsing Module
def extract_scores_from_response(llm_response: str) -> Dict[str, float]:
    """
    Parse numerical scores from the LLM's evaluation
    
    Args:
        llm_response: The raw response from the LLM
        
    Returns:
        Dictionary of scores by category
    """
    scores = {}
    
    # Define patterns for extracting scores
    score_patterns = {
        'hate_speech': {
            'representativeness': r'Representativeness:\s*(\d+(?:\.\d+)?)/10',
            'coherence': r'Coherence:\s*(\d+(?:\.\d+)?)/10',
            'specificity': r'Specificity:\s*(\d+(?:\.\d+)?)/10',
            'subtotal': r'1\.\s*HATE SPEECH.*?Subtotal:\s*(\d+(?:\.\d+)?)/10'
        },
        'counterspeech': {
            'directness': r'Directness:\s*(\d+(?:\.\d+)?)/10',
            'technique_effectiveness': r'Technique Effectiveness:\s*(\d+(?:\.\d+)?)/10',
            'educational_value': r'Educational Value:\s*(\d+(?:\.\d+)?)/10',
            'persuasiveness': r'Persuasiveness:\s*(\d+(?:\.\d+)?)/10',
            'subtotal': r'2\.\s*COUNTERSPEECH.*?Subtotal:\s*(\d+(?:\.\d+)?)/10'
        },
        'overall': {
            'topic_adherence': r'Topic Adherence:\s*(\d+(?:\.\d+)?)/10',
            'argumentation_progression': r'Argumentation Progression:\s*(\d+(?:\.\d+)?)/10',
            'non_repetitiveness': r'Non-repetitiveness:\s*(\d+(?:\.\d+)?)/10',
            'balance': r'Balance:\s*(\d+(?:\.\d+)?)/10',
            'subtotal': r'3\.\s*OVERALL DEBATE.*?Subtotal:\s*(\d+(?:\.\d+)?)/10'
        },
        'final': {
            'score': r'FINAL SCORE:\s*(\d+(?:\.\d+)?)/10'
        },
        'recommendation': {
            'include': r'RECOMMENDATION:\s*(INCLUDE|EXCLUDE)'
        }
    }
    
    import re
    
    # Extract all scores
    for category, patterns in score_patterns.items():
        scores[category] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, llm_response, re.IGNORECASE | re.DOTALL)
            if match:
                if key == 'include':
                    scores[category][key] = match.group(1) == "INCLUDE"
                else:
                    try:
                        scores[category][key] = float(match.group(1))
                    except (ValueError, IndexError):
                        scores[category][key] = 0.0
            else:
                if key == 'include':
                    scores[category][key] = False
                else:
                    scores[category][key] = 0.0
    
    return scores

def extract_justification(llm_response: str) -> str:
    """
    Extract the overall assessment
    
    Args:
        llm_response: The raw response from the LLM
        
    Returns:
        Overall assessment as string
    """
    import re
    assessment_match = re.search(r'OVERALL ASSESSMENT:\s*(.*?)(?=\n\nRECOMMENDATION:|$)', 
                               llm_response, re.DOTALL)
    if assessment_match:
        return assessment_match.group(1).strip()
    else:
        return "No assessment provided."
    
def validate_response_format(parsed_data: Dict[str, Any]) -> bool:
    """
    Verify that the response contains all expected elements
    
    Args:
        parsed_data: The parsed response data
        
    Returns:
        True if the response format is valid, False otherwise
    """
    # Check that we have scores and justification
    if not parsed_data.get('scores') or not parsed_data.get('justification'):
        return False
    
    # Check that we have all the main categories
    scores = parsed_data['scores']
    required_categories = ['hate_speech', 'counterspeech', 'overall', 'final']
    for category in required_categories:
        if category not in scores:
            return False
    
    # Check that we have a final score
    if 'score' not in scores['final']:
        return False
    
    return True

def compile_evaluation_results(scores: Dict[str, Any], justification: str) -> Dict[str, Any]:
    """
    Combine numerical and textual feedback
    
    Args:
        scores: Dictionary of numerical scores
        justification: Overall assessment text
        
    Returns:
        Combined evaluation results
    """
    return {
        'scores': scores,
        'justification': justification,
        'timestamp': time.time()
    }

# Quality Assessment Module
def calculate_overall_quality_score(evaluation_results: Dict[str, Any]) -> float:
    """
    Compute a weighted composite score
    
    Args:
        evaluation_results: The evaluation results
        
    Returns:
        Overall quality score
    """
    scores = evaluation_results['scores']
    
    # If there's already a final score calculated by the LLM, use that
    if 'final' in scores and 'score' in scores['final']:
        return scores['final']['score']
    
    # Otherwise, calculate it ourselves with custom weighting
    # Weights prioritizing counterspeech quality
    weights = {
        'hate_speech': 0.3,
        'counterspeech': 0.5,
        'overall': 0.2
    }
    
    weighted_score = 0.0
    for category, weight in weights.items():
        if category in scores and 'subtotal' in scores[category]:
            weighted_score += scores[category]['subtotal'] * weight
    
    return round(weighted_score, 1)

def apply_quality_threshold(debates_with_scores: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    """
    Filter debates that meet quality criteria
    
    Args:
        debates_with_scores: List of debates with evaluation scores
        threshold: Minimum quality threshold (0-10)
        
    Returns:
        Filtered list of high-quality debates
    """
    return [
        debate for debate in debates_with_scores 
        if debate['evaluation']['scores']['final']['score'] >= threshold
    ]

def ensure_topic_diversity(selected_debates: List[Dict[str, Any]], max_per_topic: int = 3) -> List[Dict[str, Any]]:
    """
    Make sure the final selection covers diverse topics
    
    Args:
        selected_debates: List of selected high-quality debates
        max_per_topic: Maximum number of debates per topic
        
    Returns:
        Topic-diverse list of debates
    """
    # Group debates by topic
    topic_groups = {}
    for debate in selected_debates:
        topic = debate['debate']['topic'].lower()
        if topic not in topic_groups:
            topic_groups[topic] = []
        topic_groups[topic].append(debate)
    
    # For each topic, sort by quality and take the top max_per_topic
    diverse_selection = []
    for topic, debates in topic_groups.items():
        sorted_debates = sorted(
            debates, 
            key=lambda x: x['evaluation']['scores']['final']['score'], 
            reverse=True
        )
        diverse_selection.extend(sorted_debates[:max_per_topic])
    
    return diverse_selection

def rank_debates_by_quality(debates_with_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort debates by overall quality
    
    Args:
        debates_with_scores: List of debates with evaluation scores
        
    Returns:
        Sorted list of debates (highest quality first)
    """
    return sorted(
        debates_with_scores,
        key=lambda x: x['evaluation']['scores']['final']['score'],
        reverse=True
    )

def format_for_kb_storage(debate: Dict[str, Any], evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare high-quality debates for storage
    
    Args:
        debate: Structured debate data
        evaluation: Evaluation results
        
    Returns:
        Formatted debate for KB storage
    """
    # Extract the proponent (hate speech) and opponent (counterspeech) content
    proponent_content = []
    opponent_content = []
    
    for turn in debate['turns']:
        for message in turn:
            if message['role'] == 'proponent':
                proponent_content.append(message['content'])
            elif message['role'] == 'opponent':
                opponent_content.append(message['content'])
    
    # Format for storage
    return {
        'metadata': {
            'topic': debate['topic'],
            'problematic_position': debate['position'],
            'quality_score': evaluation['scores']['final']['score'],
            'hate_speech_quality': evaluation['scores']['hate_speech']['subtotal'],
            'counterspeech_quality': evaluation['scores']['counterspeech']['subtotal'],
            'evaluation_summary': evaluation['justification']
        },
        'content': {
            'topic': debate['topic'],
            'problematic_position': debate['position'],
            'hate_speech': proponent_content,
            'counterspeech': opponent_content,
            'full_debate': debate['turns']
        },
        'evaluation': evaluation
    }


def generate_metadata_tags(debate: Dict[str, Any], evaluation: Dict[str, Any]) -> List[str]:
    """
    Create metadata tags for easy retrieval
    
    Args:
        debate: Structured debate data
        evaluation: Evaluation results
        
    Returns:
        List of metadata tags
    """
    tags = []
    
    # Add topic as tag
    tags.append(f"topic:{debate['topic'].lower()}")
    
    # Add quality tier
    score = evaluation['scores']['final']['score']
    if score >= 8.0:
        tags.append("quality:excellent")
    elif score >= 7.0:
        tags.append("quality:good")
    elif score >= 5.0:
        tags.append("quality:average")
    else:
        tags.append("quality:poor")
    
    # Add counterspeech quality tag
    cs_score = evaluation['scores']['counterspeech']['subtotal']
    if cs_score >= 8.0:
        tags.append("counterspeech:excellent")
    elif cs_score >= 7.0:
        tags.append("counterspeech:good")
    elif cs_score >= 5.0:
        tags.append("counterspeech:average")
    else:
        tags.append("counterspeech:poor")
    
    return tags

def store_in_knowledge_base(formatted_debates: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save selected debates to your knowledge base
    
    Args:
        formatted_debates: List of formatted debates for storage
        output_path: Path to save the knowledge base
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_debates, f, ensure_ascii=False, indent=2)
        print(f"Successfully stored {len(formatted_debates)} debates in knowledge base at {output_path}")
    except Exception as e:
        print(f"Error storing in knowledge base: {e}")


def update_kb_statistics(kb_path: str) -> Dict[str, Any]:
    """
    Track and update statistics about your knowledge base content
    
    Args:
        kb_path: Path to the knowledge base
        
    Returns:
        Dictionary of knowledge base statistics
    """
    try:
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb_data = json.load(f)
        
        # Calculate statistics
        topic_count = {}
        quality_scores = []
        
        for entry in kb_data:
            topic = entry['metadata']['topic']
            score = entry['metadata']['quality_score']
            
            if topic in topic_count:
                topic_count[topic] += 1
            else:
                topic_count[topic] = 1
                
            quality_scores.append(score)
        
        stats = {
            'total_entries': len(kb_data),
            'topics': topic_count,
            'unique_topics': len(topic_count),
            'avg_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'quality_distribution': {
                'excellent': sum(1 for s in quality_scores if s >= 8.0),
                'good': sum(1 for s in quality_scores if 7.0 <= s < 8.0),
                'average': sum(1 for s in quality_scores if 5.0 <= s < 7.0),
                'poor': sum(1 for s in quality_scores if s < 5.0)
            }
        }
        
        # Save stats
        stats_path = kb_path.replace('.json', '_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
            
        return stats
    except Exception as e:
        print(f"Error updating KB statistics: {e}")
        return {}
    
# Monitoring & Reporting Module
def log_evaluation_results(all_results: List[Dict[str, Any]], log_path: str) -> None:
    """
    Log detailed evaluation results for analysis
    
    Args:
        all_results: List of all evaluation results
        log_path: Path to save the logs
    """
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Successfully logged evaluation results to {log_path}")
    except Exception as e:
        print(f"Error logging evaluation results: {e}")

def generate_evaluation_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a summary of the evaluation process
    
    Args:
        results: List of evaluation results
        
    Returns:
        Summary statistics
    """
    total_debates = len(results)
    if total_debates == 0:
        return {
            'total_debates': 0,
            'debates_included': 0,
            'inclusion_rate': 0
        }
    
    # Count debates with recommendation to include
    included_debates = 0
    for r in results:
        if 'recommendation' in r['evaluation']['scores'] and 'include' in r['evaluation']['scores']['recommendation']:
            if r['evaluation']['scores']['recommendation']['include']:
                included_debates += 1
    
    all_scores = [r['evaluation']['scores']['final']['score'] for r in results]
    
    return {
        'total_debates': total_debates,
        'debates_included': included_debates,
        'inclusion_rate': included_debates / total_debates if total_debates > 0 else 0,
        'average_score': sum(all_scores) / len(all_scores) if all_scores else 0,
        'score_distribution': {
            'excellent (8-10)': sum(1 for s in all_scores if s >= 8.0) / total_debates if total_debates > 0 else 0,
            'good (7-8)': sum(1 for s in all_scores if 7.0 <= s < 8.0) / total_debates if total_debates > 0 else 0,
            'average (5-7)': sum(1 for s in all_scores if 5.0 <= s < 7.0) / total_debates if total_debates > 0 else 0,
            'poor (0-5)': sum(1 for s in all_scores if s < 5.0) / total_debates if total_debates > 0 else 0
        }
    }

def track_quality_distribution(all_scores: List[float]) -> Dict[str, int]:
    """
    Monitor the distribution of quality scores
    
    Args:
        all_scores: List of all quality scores
        
    Returns:
        Distribution of scores by category
    """
    return {
        'excellent (8-10)': sum(1 for s in all_scores if s >= 8.0),
        'good (7-8)': sum(1 for s in all_scores if 7.0 <= s < 8.0),
        'average (5-7)': sum(1 for s in all_scores if 5.0 <= s < 7.0),
        'poor (0-5)': sum(1 for s in all_scores if s < 5.0)
    }

def export_evaluation_metrics(summary_data: Dict[str, Any], output_path: str) -> None:
    """
    Export metrics for inclusion in your thesis
    
    Args:
        summary_data: Evaluation summary data
        output_path: Path to save the metrics
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Successfully exported evaluation metrics to {output_path}")
    except Exception as e:
        print(f"Error exporting evaluation metrics: {e}")

def initialize_config():
    """
    Set up configuration parameters for the evaluation
    
    Returns:
        Dictionary of configuration parameters
    """
    return {
        'quality_threshold': 7.0,  # Minimum quality score (0-10)
        'max_debates_per_topic': 3,  # Maximum debates to include per topic
        'evaluation_model': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',  # Model to use for evaluation
        'api_temperature': 0.3,  # Temperature for evaluation API calls
        'batch_size': 5  # Number of debates to process in each batch
    }

def validate_environment():
    """
    Verify that all dependencies and credentials are available
    
    Returns:
        True if environment is valid, False otherwise
    """
    # Check that Together API key is set
    if "TOGETHER_API_KEY" not in os.environ:
        print("ERROR: TOGETHER_API_KEY environment variable is not set.")
        return False
    
    # Check that required directories exist
    required_dirs = ['output']
    for d in required_dirs:
        if not os.path.exists(d):
            try:
                os.makedirs(d)
                print(f"Created directory: {d}")
            except:
                print(f"ERROR: Could not create directory: {d}")
                return False
    
    return True

def run_evaluation_pipeline(input_path, output_path, config=None):
    """
    Main function to orchestrate the entire evaluation process
    
    Args:
        input_path: Path to the debate results
        output_path: Path to save the knowledge base
        config: Optional configuration dictionary
    """
    if config is None:
        config = initialize_config()
    
    if not validate_environment():
        print("Environment validation failed. Exiting.")
        return
    
    print(f"Starting debate evaluation pipeline...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Initialize evaluator
    evaluator = TogetherAIEvaluator(config['evaluation_model'])
    
    # Load and process debates
    debates = load_debate_results(input_path)
    if not debates:
        print("No debates to evaluate. Exiting.")
        return
    
    structured_debates = []
    for debate in debates:
        structured = parse_debate_structure(debate)
        if structured['status'] != 'incomplete' and len(structured['turns']) > 0:
            structured_debates.append(structured)
    
    print(f"Structured {len(structured_debates)} valid debates for evaluation.")
    
    # Prepare batches
    batches = prepare_batch_for_evaluation(structured_debates, config['batch_size'])
    print(f"Prepared {len(batches)} batches of debates for evaluation.")
    
    # Evaluate debates
    all_results = []
    
    for batch_idx, batch in enumerate(tqdm(batches, desc="Evaluating batches")):
        batch_results = []
        
        for debate in tqdm(batch, desc=f"Batch {batch_idx+1}/{len(batches)}", leave=False):
            # Construct evaluation prompt
            prompt = construct_evaluation_prompt(debate)
            
            # Get evaluation from LLM
            response = evaluator.evaluate(
                prompt, 
                temperature=config['api_temperature']
            )
            
            # Parse response
            scores = extract_scores_from_response(response)
            justification = extract_justification(response)
            evaluation = compile_evaluation_results(scores, justification)
            
            # Calculate overall quality
            evaluation['scores']['final']['score'] = calculate_overall_quality_score(evaluation)
            
            # Store result
            result = {
                'debate': debate,
                'evaluation': evaluation,
                'raw_response': response
            }
            
            batch_results.append(result)
            all_results.append(result)
        
        # Log progress
        print(f"Completed batch {batch_idx+1}/{len(batches)}")
    
    # Filter high-quality debates
    high_quality_debates = apply_quality_threshold(all_results, config['quality_threshold'])
    print(f"Identified {len(high_quality_debates)} high-quality debates out of {len(all_results)} total.")
    
    # Ensure topic diversity
    diverse_selection = ensure_topic_diversity(high_quality_debates, config['max_debates_per_topic'])
    
    # Format debates for KB storage
    kb_debates = []
    for result in diverse_selection:
        kb_debate = format_for_kb_storage(result['debate'], result['evaluation'])
        kb_debates.append(kb_debate)
    
    # Store in knowledge base
    store_in_knowledge_base(kb_debates, output_path)
    
    # Generate and save evaluation summary
    summary = generate_evaluation_summary(all_results)
    export_evaluation_metrics(summary, output_path.replace('.json', '_metrics.json'))
    
    print(f"Evaluation complete. Selected {len(kb_debates)} high-quality debates.")
    return kb_debates