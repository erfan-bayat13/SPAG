import json
import pandas as pd
import argparse
from typing import List, Dict, Any, Optional
import os


def load_evaluation_results(file_path: str) -> Dict[str, Any]:
    """
    Load JSON evaluation results file
    
    Args:
        file_path: Path to the evaluation results JSON file
        
    Returns:
        Dictionary with evaluation results
    """
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"Error loading results file {file_path}: {e}")
        return {}


def extract_top_evaluations(results: Dict[str, Any], k: int = 10, 
                           sort_by: str = "overall_score") -> List[Dict[str, Any]]:
    """
    Extract the top K counter-narratives based on specified metric
    
    Args:
        results: Evaluation results dictionary
        k: Number of top examples to extract
        sort_by: Metric to sort by ('overall_score', 'effectiveness', etc.)
        
    Returns:
        List of top K examples with their metrics
    """
    # Check if results contain the needed data
    if not results or 'data' not in results or 'metrics' not in results:
        print("Invalid results format")
        return []
    
    # Extract core data
    hate_texts = results['data'].get('hate_texts', [])
    generated_counters = results['data'].get('generated_counters', [])
    targets = results['data'].get('targets', [])
    
    # Extract evaluations
    evaluations = results['metrics'].get('llm_evaluation', {}).get('evaluations', [])
    
    # Build complete examples with all metrics
    examples = []
    for i, eval_data in enumerate(evaluations):
        if i >= len(hate_texts) or i >= len(generated_counters):
            continue
            
        # Get toxicity data if available
        counter_toxicity = {}
        hate_toxicity = {}
        toxicity_reduction = {}
        
        if 'toxicity' in results['metrics']:
            if i < len(results['metrics']['toxicity']['counter_scores']):
                counter_toxicity = results['metrics']['toxicity']['counter_scores'][i]
            
            if i < len(results['metrics']['toxicity']['hate_scores']):
                hate_toxicity = results['metrics']['toxicity']['hate_scores'][i]
                
            # Calculate reduction for each attribute
            if counter_toxicity and hate_toxicity:
                for attr in hate_toxicity:
                    if attr in counter_toxicity and attr != 'error':
                        reduction = hate_toxicity[attr] - counter_toxicity[attr]
                        toxicity_reduction[attr] = reduction
        
        # Build example with all data
        example = {
            'index': i,
            'hate_text': hate_texts[i],
            'counter_text': generated_counters[i],
            'target': targets[i] if i < len(targets) else None,
            'evaluation': eval_data,
            'counter_toxicity': counter_toxicity,
            'hate_toxicity': hate_toxicity,
            'toxicity_reduction': toxicity_reduction
        }
        
        examples.append(example)
    
    # Get sort key - default to overall_score if the specified key doesn't exist
    def get_sort_key(example):
        if sort_by in example['evaluation']:
            return example['evaluation'][sort_by]
        return example['evaluation'].get('overall_score', 0)
    
    # Sort examples by the specified metric
    sorted_examples = sorted(examples, key=get_sort_key, reverse=True)
    
    # Return top K (or all if fewer than K)
    return sorted_examples[:min(k, len(sorted_examples))]


def format_example_output(example: Dict[str, Any], include_toxicity: bool = True) -> str:
    """
    Format a single example for nicer console output
    
    Args:
        example: Dictionary with example data
        include_toxicity: Whether to include toxicity metrics
        
    Returns:
        Formatted string output
    """
    output = []
    output.append(f"Example #{example['index'] + 1}")
    output.append("=" * 80)
    
    # Display target if available
    if example['target']:
        output.append(f"Target Group: {example['target']}")
    
    # Display hate text
    output.append("\nHATE TEXT:")
    output.append(f"{example['hate_text']}")
    
    # Display counter text
    output.append("\nCOUNTER TEXT:")
    output.append(f"{example['counter_text']}")
    
    # Display LLM evaluation scores
    eval_data = example['evaluation']
    output.append("\nLLM EVALUATION:")
    output.append(f"Overall Score: {eval_data.get('overall_score', 'N/A'):.2f}")
    output.append(f"Effectiveness: {eval_data.get('effectiveness', 'N/A'):.2f}")
    output.append(f"Relevance: {eval_data.get('relevance', 'N/A'):.2f}")
    output.append(f"Evidence Quality: {eval_data.get('evidence_quality', 'N/A'):.2f}")
    output.append(f"Persuasiveness: {eval_data.get('persuasiveness', 'N/A'):.2f}")
    output.append(f"Directness: {eval_data.get('directness', 'N/A'):.2f}")
    
    # Display feedback if available
    if 'feedback' in eval_data:
        output.append(f"\nFeedback: {eval_data['feedback']}")
    
    # Include toxicity metrics if requested
    if include_toxicity:
        if example['counter_toxicity']:
            output.append("\nCOUNTER TOXICITY:")
            for attr, score in example['counter_toxicity'].items():
                if attr != 'error':
                    output.append(f"{attr.title()}: {score:.4f}")
        
        if example['hate_toxicity']:
            output.append("\nHATE TOXICITY:")
            for attr, score in example['hate_toxicity'].items():
                if attr != 'error':
                    output.append(f"{attr.title()}: {score:.4f}")
        
        if example['toxicity_reduction']:
            output.append("\nTOXICITY REDUCTION:")
            for attr, reduction in example['toxicity_reduction'].items():
                output.append(f"{attr.title()}: {reduction:.4f}")
    
    return "\n".join(output)


def analyze_top_examples(top_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze common patterns in top-performing examples
    
    Args:
        top_examples: List of top-performing examples
        
    Returns:
        Dictionary with analysis results
    """
    if not top_examples:
        return {"error": "No examples to analyze"}
    
    # Calculate average scores
    avg_scores = {
        'overall_score': 0,
        'effectiveness': 0,
        'relevance': 0,
        'evidence_quality': 0,
        'persuasiveness': 0,
        'directness': 0
    }
    
    for example in top_examples:
        eval_data = example['evaluation']
        for key in avg_scores:
            avg_scores[key] += eval_data.get(key, 0)
    
    # Calculate averages
    for key in avg_scores:
        avg_scores[key] /= len(top_examples)
    
    # Analyze length characteristics
    avg_hate_length = sum(len(ex['hate_text'].split()) for ex in top_examples) / len(top_examples)
    avg_counter_length = sum(len(ex['counter_text'].split()) for ex in top_examples) / len(top_examples)
    
    # Analyze frequency of target groups if available
    target_counts = {}
    for example in top_examples:
        if example['target']:
            target = example['target']
            target_counts[target] = target_counts.get(target, 0) + 1
    
    # Sort targets by frequency
    top_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'avg_scores': avg_scores,
        'avg_hate_length': avg_hate_length,
        'avg_counter_length': avg_counter_length,
        'top_targets': top_targets
    }


def export_to_csv(top_examples: List[Dict[str, Any]], output_path: str):
    """
    Export top examples to a CSV file
    
    Args:
        top_examples: List of top examples
        output_path: Path to save CSV file
    """
    # Convert nested dictionaries to flat structure
    rows = []
    for example in top_examples:
        row = {
            'index': example['index'],
            'hate_text': example['hate_text'],
            'counter_text': example['counter_text'],
            'target': example['target'] if example['target'] else ''
        }
        
        # Add evaluation metrics
        for key, value in example['evaluation'].items():
            row[f'eval_{key}'] = value
        
        # Add toxicity metrics
        for key, value in example['counter_toxicity'].items():
            if key != 'error':
                row[f'counter_tox_{key}'] = value
        
        for key, value in example['hate_toxicity'].items():
            if key != 'error':
                row[f'hate_tox_{key}'] = value
        
        for key, value in example['toxicity_reduction'].items():
            row[f'tox_reduction_{key}'] = value
        
        rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Exported {len(rows)} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract top counter-narratives from evaluation results")
    parser.add_argument("--input", type=str, required=True, help="Path to evaluation results JSON file")
    parser.add_argument("--output", type=str, help="Path to save CSV output (optional)")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top examples to extract")
    parser.add_argument("--sort_by", type=str, default="overall_score", 
                        choices=["overall_score", "effectiveness", "relevance", 
                                "evidence_quality", "persuasiveness", "directness"],
                        help="Metric to sort by")
    parser.add_argument("--no_toxicity", action="store_true", 
                        help="Exclude toxicity metrics from output")
    
    args = parser.parse_args()
    
    # Load evaluation results
    results = load_evaluation_results(args.input)
    if not results:
        return
    
    # Extract top examples
    top_examples = extract_top_evaluations(
        results=results, 
        k=args.top_k, 
        sort_by=args.sort_by
    )
    
    if not top_examples:
        print("No examples found or error in extraction")
        return
    
    # Print top examples
    print(f"\nTop {len(top_examples)} Counter-Narratives (sorted by {args.sort_by}):\n")
    for i, example in enumerate(top_examples):
        print(format_example_output(example, include_toxicity=not args.no_toxicity))
        print("\n" + "-" * 80 + "\n")
    
    # Analyze patterns in top examples
    analysis = analyze_top_examples(top_examples)
    
    print("\nAnalysis of Top Examples:")
    print("=" * 80)
    print("\nAverage Scores:")
    for key, value in analysis['avg_scores'].items():
        print(f"{key.replace('_', ' ').title()}: {value:.2f}")
    
    print(f"\nAverage Hate Text Length: {analysis['avg_hate_length']:.1f} words")
    print(f"Average Counter Text Length: {analysis['avg_counter_length']:.1f} words")
    
    if analysis['top_targets']:
        print("\nMost Common Target Groups:")
        for target, count in analysis['top_targets']:
            print(f"{target}: {count} examples ({count/len(top_examples)*100:.1f}%)")
    
    # Export to CSV if output path provided
    if args.output:
        export_to_csv(top_examples, args.output)


if __name__ == "__main__":
    main()