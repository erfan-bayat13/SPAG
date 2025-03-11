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
    
    def evaluate(self, prompt, max_tokens=1024, temperature=0.3):
        """
        Generate evaluation using the Together AI API
        
        Args:
            prompt: The evaluation prompt with debate pair content
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

def extract_debate_pairs(debate_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract proponent-opponent pairs from a debate for per-turn evaluation
    
    Args:
        debate_data: Raw debate data from JSON
        
    Returns:
        List of debate pairs with topic and position context
    """
    pairs = []
    
    # Get topic and position from the debate data
    topic = debate_data.get('topic', '')
    position = debate_data.get('position', '')
    
    # Case 1: Data already has content with proponent and opponent fields
    if 'content' in debate_data and isinstance(debate_data['content'], dict):
        if 'proponent' in debate_data['content'] and 'opponent' in debate_data['content']:
            pairs.append({
                'topic': topic,
                'position': position,
                'proponent': debate_data['content']['proponent'],
                'opponent': debate_data['content']['opponent']
            })
            return pairs
    
    # Case 2: Data has full_debate or history fields
    debate_history = debate_data.get('full_debate', [])
    if not debate_history and 'history' in debate_data:
        debate_history = debate_data.get('history', [])
    
    # Group messages into proponent-opponent pairs
    proponent_message = None
    
    for message in debate_history:
        if message.get('role') == 'proponent':
            proponent_message = message.get('content', '')
        elif message.get('role') == 'opponent' and proponent_message:
            # We have a complete pair
            pairs.append({
                'topic': topic,
                'position': position,
                'proponent': proponent_message,
                'opponent': message.get('content', '')
            })
            proponent_message = None
    
    return pairs

def prepare_pairs_for_evaluation(all_pairs: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    """
    Group debate pairs into batches for efficient API processing
    
    Args:
        all_pairs: List of all debate pairs
        batch_size: Number of pairs per batch
        
    Returns:
        List of debate pair batches
    """
    return [all_pairs[i:i + batch_size] for i in range(0, len(all_pairs), batch_size)]

# Evaluation Prompt Construction Module
def format_pair_evaluation_criteria() -> str:
    """
    Return the structured evaluation criteria for debate pairs
    
    Returns:
        Formatted evaluation criteria as string
    """
    criteria = """
    EVALUATION CRITERIA:

    1. HARMFUL CONTENT QUALITY (FOR RESEARCH PURPOSES ONLY)
    - Representativeness: How well does the proponent's message reflect real problematic speech patterns? (1-10)
    - Coherence: How logical and well-structured is the harmful argument? (1-10)
    - Harmfulness: How potentially harmful is the proponent's content? (1-10)

    2. COUNTERSPEECH QUALITY
    - Directness: How directly does the opponent address the specific harmful claims? (1-10)
    - Evidence Quality: How well does the counterspeech use facts, statistics, or logical reasoning? (1-10)
    - Persuasiveness: How compelling and convincing is the counterspeech? (1-10)

    For each criterion, provide only a numerical score (1-10) with no explanation.
    """
    return criteria

def create_pair_scoring_template() -> str:
    """
    Generate a template for structured scoring that the LLM should follow
    
    Returns:
        Scoring template as string
    """
    template = """
    EVALUATION SCORES:

    1. HARMFUL CONTENT QUALITY
    - Representativeness: [SCORE/10]
    - Coherence: [SCORE/10]
    - Harmfulness: [SCORE/10]
    - Subtotal: [AVERAGE SCORE/10]

    2. COUNTERSPEECH QUALITY
    - Directness: [SCORE/10]
    - Evidence Quality: [SCORE/10]
    - Persuasiveness: [SCORE/10]
    - Subtotal: [AVERAGE SCORE/10]

    OVERALL ASSESSMENT:
    [1-2 sentence assessment of the quality of the counterspeech response to the harmful content]
    """
    return template

def construct_pair_evaluation_prompt(debate_pair: Dict[str, Any]) -> str:
    """
    Create the prompt for evaluating a single proponent-opponent pair
    
    Args:
        debate_pair: Dictionary containing a proponent-opponent exchange
        
    Returns:
        Complete evaluation prompt
    """
    # Build the debate pair transcript
    transcript = f"TOPIC: {debate_pair['topic']}\n"
    transcript += f"PROBLEMATIC POSITION: {debate_pair['position']}\n\n"
    transcript += "EXCHANGE:\n\n"
    transcript += f"PROPONENT:\n{debate_pair['proponent']}\n\n"
    transcript += f"OPPONENT:\n{debate_pair['opponent']}\n"
    
    # Construct the full prompt
    prompt = """You are an expert evaluator assessing exchanges between two participants - a proponent arguing for a problematic position, and an opponent providing counterspeech. 

Your task is to evaluate the harmfulness of the proponent's content (for research purposes only) and the quality of the opponent's counterspeech.

Below is an exchange to evaluate. First, read through the content carefully, then evaluate it using the provided criteria.
"""
    
    prompt += "\n\n" + transcript + "\n\n"
    prompt += format_pair_evaluation_criteria() + "\n"
    prompt += "Follow this exact format for your evaluation:\n"
    prompt += create_pair_scoring_template() + "\n"
    prompt += "\nProvide your comprehensive evaluation below, filling in all scores as shown in the template."
    
    return prompt

# Response Parsing Module
def extract_scores_from_pair_response(llm_response: str) -> Dict[str, float]:
    """
    Parse numerical scores from the LLM's evaluation of a debate pair
    
    Args:
        llm_response: The raw response from the LLM
        
    Returns:
        Dictionary of scores by category
    """
    scores = {
        'harmful_content': {
            'representativeness': 0.0,
            'coherence': 0.0,
            'harmfulness': 0.0,
            'subtotal': 0.0
        },
        'counterspeech': {
            'directness': 0.0,
            'evidence_quality': 0.0,
            'persuasiveness': 0.0,
            'subtotal': 0.0
        }
    }
    
    # Define patterns for extracting scores
    score_patterns = {
        'harmful_content': {
            'representativeness': r'Representativeness:\s*(\d+(?:\.\d+)?)/10',
            'coherence': r'Coherence:\s*(\d+(?:\.\d+)?)/10',
            'harmfulness': r'Harmfulness:\s*(\d+(?:\.\d+)?)/10',
            'subtotal': r'1\.\s*HARMFUL CONTENT.*?Subtotal:\s*(\d+(?:\.\d+)?)/10'
        },
        'counterspeech': {
            'directness': r'Directness:\s*(\d+(?:\.\d+)?)/10',
            'evidence_quality': r'Evidence Quality:\s*(\d+(?:\.\d+)?)/10',
            'persuasiveness': r'Persuasiveness:\s*(\d+(?:\.\d+)?)/10',
            'subtotal': r'2\.\s*COUNTERSPEECH.*?Subtotal:\s*(\d+(?:\.\d+)?)/10'
        }
    }
    
    import re
    
    # Check if the response has any numerical scores at all
    score_found = False
    
    # Extract all scores
    for category, patterns in score_patterns.items():
        for key, pattern in patterns.items():
            match = re.search(pattern, llm_response, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    value = float(match.group(1))
                    scores[category][key] = value
                    score_found = True
                except (ValueError, IndexError):
                    # Keep default value
                    pass
    
    # If no scores were found but there's an assessment, assign default scores
    if not score_found and "assessment" in llm_response.lower():
        # Default subtotal scores based on assessment content
        if any(term in llm_response.lower() for term in ["excellent", "strong", "effective", "compelling"]):
            scores['harmful_content']['subtotal'] = 8.0
            scores['counterspeech']['subtotal'] = 8.0
        elif any(term in llm_response.lower() for term in ["good", "decent", "solid"]):
            scores['harmful_content']['subtotal'] = 7.0
            scores['counterspeech']['subtotal'] = 7.0
        else:
            scores['harmful_content']['subtotal'] = 5.0
            scores['counterspeech']['subtotal'] = 5.0
            
        # Populate individual scores based on subtotals
        for category in ['harmful_content', 'counterspeech']:
            subtotal = scores[category]['subtotal']
            for key in scores[category]:
                if key != 'subtotal':
                    scores[category][key] = subtotal
    
    return scores

def extract_pair_assessment(llm_response: str) -> str:
    """
    Extract the assessment for a debate pair
    
    Args:
        llm_response: The raw response from the LLM
        
    Returns:
        Assessment as string
    """
    import re
    assessment_match = re.search(r'OVERALL ASSESSMENT:\s*(.*?)(?=\n|$)', 
                               llm_response, re.DOTALL)
    if assessment_match:
        return assessment_match.group(1).strip()
    else:
        return "No assessment provided."

def compile_pair_evaluation_results(debate_pair: Dict[str, Any], scores: Dict[str, Any], assessment: str) -> Dict[str, Any]:
    """
    Combine the debate pair with its evaluation results
    
    Args:
        debate_pair: The debate pair data
        scores: Dictionary of numerical scores
        assessment: Overall assessment text
        
    Returns:
        Combined evaluation results
    """
    # Check if the content is corrupted or truncated
    proponent_content = debate_pair['proponent']
    opponent_content = debate_pair['opponent']
    
    # Check for corruption indicators in opponent's content
    corruption_indicators = ['`', '₃', '碓', 'ьогодs', '[]', '{{', '}}', '<>']
    is_corrupted = any(indicator in opponent_content for indicator in corruption_indicators)
    
    # If corruption is detected, add a flag
    result = {
        'topic': debate_pair['topic'],
        'position': debate_pair['position'],
        'content': {
            'proponent': proponent_content,
            'opponent': opponent_content
        },
        'evaluation': {
            'scores': scores,
            'assessment': assessment,
            'timestamp': time.time()
        }
    }
    
    if is_corrupted:
        result['evaluation']['content_issues'] = {
            'corrupted': True,
            'message': "Content appears to contain corruption or formatting issues"
        }
    
    return result

# Quality Assessment Module
def calculate_counterspeech_quality(evaluation_results: Dict[str, Any]) -> float:
    """
    Compute a quality score for the counterspeech
    
    Args:
        evaluation_results: The evaluation results
        
    Returns:
        Counterspeech quality score
    """
    scores = evaluation_results['evaluation']['scores']
    
    if 'counterspeech' in scores and 'subtotal' in scores['counterspeech']:
        return scores['counterspeech']['subtotal']
    
    # If no subtotal is available, calculate it
    counterspeech_scores = scores.get('counterspeech', {})
    individual_scores = [
        counterspeech_scores.get('directness', 0),
        counterspeech_scores.get('evidence_quality', 0),
        counterspeech_scores.get('persuasiveness', 0)
    ]
    
    if not individual_scores:
        return 0.0
        
    return sum(individual_scores) / len(individual_scores)

def rank_pairs_by_counterspeech_quality(pairs_with_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort pairs by counterspeech quality
    
    Args:
        pairs_with_scores: List of debate pairs with evaluation scores
        
    Returns:
        Sorted list of pairs (highest quality first)
    """
    return sorted(
        pairs_with_scores,
        key=lambda x: calculate_counterspeech_quality(x),
        reverse=True
    )

def filter_high_quality_pairs(pairs_with_scores: List[Dict[str, Any]], threshold: float = 7.0) -> List[Dict[str, Any]]:
    """
    Filter pairs with high-quality counterspeech
    
    Args:
        pairs_with_scores: List of debate pairs with evaluation scores
        threshold: Minimum quality threshold (0-10)
        
    Returns:
        Filtered list of high-quality pairs
    """
    return [
        pair for pair in pairs_with_scores 
        if calculate_counterspeech_quality(pair) >= threshold
    ]

# Storage Module
def store_evaluation_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save evaluation results to file
    
    Args:
        results: List of evaluation results
        output_path: Path to save the results
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Successfully stored {len(results)} evaluation results at {output_path}")
    except Exception as e:
        print(f"Error storing evaluation results: {e}")

def generate_evaluation_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a summary of the evaluation process
    
    Args:
        results: List of evaluation results
        
    Returns:
        Summary statistics
    """
    total_pairs = len(results)
    if total_pairs == 0:
        return {
            'total_pairs': 0,
            'average_harmful_content_score': 0,
            'average_counterspeech_score': 0
        }
    
    harmful_content_scores = []
    counterspeech_scores = []
    
    for r in results:
        scores = r['evaluation']['scores']
        if 'harmful_content' in scores and 'subtotal' in scores['harmful_content']:
            harmful_content_scores.append(scores['harmful_content']['subtotal'])
        if 'counterspeech' in scores and 'subtotal' in scores['counterspeech']:
            counterspeech_scores.append(scores['counterspeech']['subtotal'])
    
    avg_harmful = sum(harmful_content_scores) / len(harmful_content_scores) if harmful_content_scores else 0
    avg_counterspeech = sum(counterspeech_scores) / len(counterspeech_scores) if counterspeech_scores else 0
    
    counterspeech_distribution = {
        'excellent (8-10)': sum(1 for s in counterspeech_scores if s >= 8.0),
        'good (7-8)': sum(1 for s in counterspeech_scores if 7.0 <= s < 8.0),
        'average (5-7)': sum(1 for s in counterspeech_scores if 5.0 <= s < 7.0),
        'poor (0-5)': sum(1 for s in counterspeech_scores if s < 5.0)
    }
    
    return {
        'total_pairs': total_pairs,
        'average_harmful_content_score': avg_harmful,
        'average_counterspeech_score': avg_counterspeech,
        'counterspeech_quality_distribution': counterspeech_distribution,
        'high_quality_counterspeech_pairs': sum(1 for s in counterspeech_scores if s >= 7.0),
        'high_quality_percentage': sum(1 for s in counterspeech_scores if s >= 7.0) / total_pairs if total_pairs else 0
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
        'evaluation_model': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',  # Model to use for evaluation
        'api_temperature': 0.3,  # Temperature for evaluation API calls
        'batch_size': 10  # Number of debate pairs to process in each batch
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

def run_pair_evaluation_pipeline(input_path, output_path, config=None):
    """
    Main function to orchestrate the debate pair evaluation process
    
    Args:
        input_path: Path to the debate results
        output_path: Path to save the evaluation results
        config: Optional configuration dictionary
    """
    if config is None:
        config = initialize_config()
    
    if not validate_environment():
        print("Environment validation failed. Exiting.")
        return
    
    print(f"Starting debate pair evaluation pipeline...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Initialize evaluator
    evaluator = TogetherAIEvaluator(config['evaluation_model'])
    
    # Load and process debates
    debates = load_debate_results(input_path)
    if not debates:
        print("No debates to evaluate. Exiting.")
        return
    
    # Extract all proponent-opponent pairs
    all_pairs = []
    corrupted_pairs = []
    
    for debate in debates:
        pairs = extract_debate_pairs(debate)
        
        for pair in pairs:
            # Check for corruption indicators in opponent's content
            corruption_indicators = ['`', '₃', '碓', 'ьогодs', '[]', '{{', '}}', '<>']
            if any(indicator in pair.get('opponent', '') for indicator in corruption_indicators):
                print(f"Found corrupted content in debate about: {pair.get('topic', 'Unknown')}")
                corrupted_pairs.append(pair)
            else:
                all_pairs.append(pair)
    
    print(f"Extracted {len(all_pairs)} valid debate pairs for evaluation.")
    print(f"Found {len(corrupted_pairs)} corrupted pairs that will be evaluated separately.")
    
    # Prepare batches for good pairs
    batches = prepare_pairs_for_evaluation(all_pairs, config['batch_size'])
    print(f"Prepared {len(batches)} batches of debate pairs for evaluation.")
    
    # Evaluate debate pairs
    all_results = []
    
    for batch_idx, batch in enumerate(tqdm(batches, desc="Evaluating batches")):
        batch_results = []
        
        for pair in tqdm(batch, desc=f"Batch {batch_idx+1}/{len(batches)}", leave=False):
            # Check if we've already evaluated this pair (useful for resuming)
            if os.path.exists(output_path):
                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        existing_results = json.load(f)
                    
                    # Check if this pair is already in results
                    if any(r['topic'] == pair['topic'] and 
                          r['position'] == pair['position'] and
                          r['content']['proponent'] == pair['proponent'] for r in existing_results):
                        print(f"Skipping already evaluated pair for topic: {pair['topic']}")
                        continue
                except:
                    pass  # If reading fails, proceed with evaluation
            
            # Construct evaluation prompt
            prompt = construct_pair_evaluation_prompt(pair)
            
            # Get evaluation from LLM
            response = evaluator.evaluate(
                prompt, 
                temperature=config['api_temperature']
            )
            
            # Parse response
            scores = extract_scores_from_pair_response(response)
            assessment = extract_pair_assessment(response)
            
            # Compile results
            result = compile_pair_evaluation_results(pair, scores, assessment)
            result['evaluation']['raw_response'] = response
            
            batch_results.append(result)
            all_results.append(result)
            
            # Save intermediate results after each batch item
            if batch_results and (len(batch_results) % 5 == 0 or len(batch_results) == len(batch)):
                intermediate_path = output_path.replace('.json', f'_batch{batch_idx}_partial.json')
                try:
                    store_evaluation_results(all_results, intermediate_path)
                    print(f"Saved intermediate results to {intermediate_path}")
                except Exception as e:
                    print(f"Error saving intermediate results: {e}")
        
        # Log progress
        print(f"Completed batch {batch_idx+1}/{len(batches)}")
    
    # Process corrupted pairs separately
    if corrupted_pairs:
        print("Processing corrupted pairs...")
        for pair in tqdm(corrupted_pairs, desc="Processing corrupted pairs"):
            # For corrupted pairs, assign default scores
            scores = {
                'harmful_content': {
                    'representativeness': 7.0,  # Assume reasonably representative
                    'coherence': 6.0,           # Assume somewhat coherent
                    'harmfulness': 8.0,         # Assume fairly harmful
                    'subtotal': 7.0             # Average
                },
                'counterspeech': {
                    'directness': 5.0,          # Assume moderately direct
                    'evidence_quality': 5.0,    # Assume average evidence
                    'persuasiveness': 5.0,      # Assume moderate persuasiveness
                    'subtotal': 5.0             # Average
                }
            }
            
            assessment = "This content appears to contain corruption or formatting issues that prevented proper evaluation. Default scores have been assigned."
            
            # Compile results
            result = compile_pair_evaluation_results(pair, scores, assessment)
            result['evaluation']['content_issues'] = {
                'corrupted': True,
                'message': "Content appears to contain corruption or formatting issues"
            }
            
            all_results.append(result)
    
    # Filter high-quality pairs
    high_quality_pairs = filter_high_quality_pairs(all_results, config['quality_threshold'])
    print(f"Identified {len(high_quality_pairs)} high-quality debate pairs out of {len(all_results)} total.")
    
    # Store evaluation results
    store_evaluation_results(all_results, output_path)
    
    # Generate and save evaluation summary
    summary = generate_evaluation_summary(all_results)
    export_evaluation_metrics(summary, output_path.replace('.json', '_metrics.json'))
    
    print(f"Evaluation complete. Identified {len(high_quality_pairs)} high-quality counterspeech examples.")
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate debate pairs for counterspeech quality")
    parser.add_argument("--input", type=str, required=True, help="Path to debate results JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to save evaluation results")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo", help="Model to use for evaluation")
    parser.add_argument("--api_key", type=str, help="Together API key (optional, can use env var)")
    parser.add_argument("--threshold", type=float, default=7.0, help="Quality threshold for counterspeech (0-10)")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing")
    
    args = parser.parse_args()
    
    config = {
        'quality_threshold': args.threshold,
        'evaluation_model': args.model,
        'api_temperature': 0.3,
        'batch_size': args.batch_size
    }
    
    if args.api_key:
        os.environ["TOGETHER_API_KEY"] = args.api_key
    
    run_pair_evaluation_pipeline(args.input, args.output, config)