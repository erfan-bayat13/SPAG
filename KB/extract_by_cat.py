import json
import pandas as pd
import argparse
from typing import List, Dict, Any, Optional
import os
from collections import defaultdict


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


def extract_top_by_category(results: Dict[str, Any], k: int = 5, 
                           sort_by: str = "overall_score") -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract the top K counter-narratives for each target category
    
    Args:
        results: Evaluation results dictionary
        k: Number of top examples to extract per category
        sort_by: Metric to sort by ('overall_score', 'effectiveness', etc.)
        
    Returns:
        Dictionary with target categories as keys and lists of top examples as values
    """
    # Check if results contain the needed data
    if not results or 'data' not in results or 'metrics' not in results:
        print("Invalid results format")
        return {}
    
    # Extract core data
    hate_texts = results['data'].get('hate_texts', [])
    generated_counters = results['data'].get('generated_counters', [])
    targets = results['data'].get('targets', [])
    
    if not targets:
        print("No target categories found in the data")
        return {}
    
    # Extract evaluations
    evaluations = results['metrics'].get('llm_evaluation', {}).get('evaluations', [])
    
    # Group examples by target category
    examples_by_category = defaultdict(list)
    
    for i, eval_data in enumerate(evaluations):
        if i >= len(hate_texts) or i >= len(generated_counters) or i >= len(targets):
            continue
        
        target = targets[i]
        if not target:
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
            'target': target,
            'evaluation': eval_data,
            'counter_toxicity': counter_toxicity,
            'hate_toxicity': hate_toxicity,
            'toxicity_reduction': toxicity_reduction
        }
        
        examples_by_category[target].append(example)
    
    # Get sort key function - default to overall_score if the specified key doesn't exist
    def get_sort_key(example):
        if sort_by in example['evaluation']:
            return example['evaluation'][sort_by]
        return example['evaluation'].get('overall_score', 0)
    
    # Sort examples by the specified metric for each category and get top K
    top_examples_by_category = {}
    
    for category, examples in examples_by_category.items():
        sorted_examples = sorted(examples, key=get_sort_key, reverse=True)
        top_examples_by_category[category] = sorted_examples[:min(k, len(sorted_examples))]
    
    return top_examples_by_category


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


def compare_categories(top_by_category: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Compare performance across different target categories
    
    Args:
        top_by_category: Dictionary with target categories and their top examples
        
    Returns:
        Dictionary with comparative analysis
    """
    if not top_by_category:
        return {"error": "No category data to analyze"}
    
    # Compare average scores across categories
    category_scores = {}
    category_lengths = {}
    
    for category, examples in top_by_category.items():
        if not examples:
            continue
            
        # Calculate average scores for this category
        avg_scores = {
            'overall_score': 0,
            'effectiveness': 0,
            'relevance': 0,
            'evidence_quality': 0,
            'persuasiveness': 0,
            'directness': 0
        }
        
        for example in examples:
            eval_data = example['evaluation']
            for key in avg_scores:
                avg_scores[key] += eval_data.get(key, 0)
        
        # Calculate averages
        for key in avg_scores:
            avg_scores[key] /= len(examples)
        
        category_scores[category] = avg_scores
        
        # Calculate average text lengths
        avg_hate_length = sum(len(ex['hate_text'].split()) for ex in examples) / len(examples)
        avg_counter_length = sum(len(ex['counter_text'].split()) for ex in examples) / len(examples)
        
        category_lengths[category] = {
            'avg_hate_length': avg_hate_length,
            'avg_counter_length': avg_counter_length
        }
    
    # Identify best and worst performing categories
    best_category = max(category_scores.items(), key=lambda x: x[1]['overall_score'])
    worst_category = min(category_scores.items(), key=lambda x: x[1]['overall_score'])
    
    return {
        'category_scores': category_scores,
        'category_lengths': category_lengths,
        'best_category': best_category[0],
        'worst_category': worst_category[0]
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


def export_categories_to_csv(top_by_category: Dict[str, List[Dict[str, Any]]], 
                           output_dir: str, prefix: str = "top_counters"):
    """
    Export top examples for each category to separate CSV files
    
    Args:
        top_by_category: Dictionary with target categories and their top examples
        output_dir: Directory to save CSV files
        prefix: Prefix for CSV filenames
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Export each category to a separate CSV
    for category, examples in top_by_category.items():
        if not examples:
            continue
            
        # Create sanitized category name for filename
        safe_category = category.lower().replace(' ', '_')
        output_path = os.path.join(output_dir, f"{prefix}_{safe_category}.csv")
        
        # Export to CSV
        export_to_csv(examples, output_path)
    
    # Also export a combined CSV with all categories
    all_examples = []
    for examples in top_by_category.values():
        all_examples.extend(examples)
    
    combined_path = os.path.join(output_dir, f"{prefix}_all_categories.csv")
    export_to_csv(all_examples, combined_path)
    
    print(f"Exported CSVs for {len(top_by_category)} categories to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract top counter-narratives from evaluation results")
    parser.add_argument("--input", type=str, required=True, help="Path to evaluation results JSON file")
    parser.add_argument("--output_dir", type=str, default="top_counters", 
                        help="Directory to save CSV outputs")
    parser.add_argument("--top_k", type=int, default=5, 
                        help="Number of top examples to extract per category")
    parser.add_argument("--sort_by", type=str, default="overall_score", 
                        choices=["overall_score", "effectiveness", "relevance", 
                                "evidence_quality", "persuasiveness", "directness"],
                        help="Metric to sort by")
    parser.add_argument("--no_toxicity", action="store_true", 
                        help="Exclude toxicity metrics from output")
    parser.add_argument("--by_category", action="store_true", default=True,
                        help="Analyze results by target category")
    
    args = parser.parse_args()
    
    # Load evaluation results
    results = load_evaluation_results(args.input)
    if not results:
        return
    
    # Check for targets data
    has_targets = ('data' in results and 'targets' in results['data'] and results['data']['targets'])
    
    if args.by_category and has_targets:
        # Extract top examples for each category
        top_by_category = extract_top_by_category(
            results=results,
            k=args.top_k,
            sort_by=args.sort_by
        )
        
        if not top_by_category:
            print("No category examples found or error in extraction")
            return
        
        # Print top examples for each category
        print(f"\nTop {args.top_k} Counter-Narratives by Category (sorted by {args.sort_by}):\n")
        
        for category, examples in top_by_category.items():
            print(f"\n{'#' * 40}")
            print(f"CATEGORY: {category}")
            print(f"{'#' * 40}\n")
            
            for i, example in enumerate(examples):
                print(format_example_output(example, include_toxicity=not args.no_toxicity))
                print("\n" + "-" * 80 + "\n")
            
            # Analyze patterns in top examples for this category
            category_analysis = analyze_top_examples(examples)
            
            print(f"\nAnalysis for Category: {category}")
            print("=" * 80)
            print("\nAverage Scores:")
            for key, value in category_analysis['avg_scores'].items():
                print(f"{key.replace('_', ' ').title()}: {value:.2f}")
            
            print(f"\nAverage Hate Text Length: {category_analysis['avg_hate_length']:.1f} words")
            print(f"Average Counter Text Length: {category_analysis['avg_counter_length']:.1f} words")
            print("\n" + "=" * 80 + "\n")
        
        # Compare performance across categories
        comparison = compare_categories(top_by_category)
        
        print("\nCross-Category Comparison:")
        print("=" * 80)
        print(f"\nBest Performing Category: {comparison['best_category']}")
        print(f"Worst Performing Category: {comparison['worst_category']}")
        
        print("\nAverage Overall Score by Category:")
        for category, scores in comparison['category_scores'].items():
            print(f"{category}: {scores['overall_score']:.2f}")
        
        # Export to CSV files
        if args.output_dir:
            export_categories_to_csv(top_by_category, args.output_dir)
    
    else:
        # Extract top examples across all data
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
        
        # Export to single CSV if output directory provided
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, "top_counters_overall.csv")
            export_to_csv(top_examples, output_path)


if __name__ == "__main__":
    main()