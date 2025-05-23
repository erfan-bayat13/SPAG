import json
import argparse

def find_entry_by_index(json_path, index):
    """
    Find and display the hate speech and counter narrative pair at the given index
    
    Args:
        json_path: Path to the evaluation results JSON file
        index: Index of the entry to find
    """
    try:
        # Load the JSON file
        with open(json_path, 'r') as f:
            results = json.load(f)
        
        # Get data section
        data = results.get('data', {})
        
        # Check if index is valid
        hate_texts = data.get('hate_texts', [])
        if not hate_texts or index < 0 or index >= len(hate_texts):
            print(f"Error: Index {index} is out of range. Valid range: 0-{len(hate_texts)-1}")
            return
        
        # Get the hate speech
        hate_speech = hate_texts[index]
        
        # Get the counter narrative
        generated_counters = data.get('generated_counters', [])
        counter_narrative = generated_counters[index] if index < len(generated_counters) else "No counter narrative found"
        
        # Get target if available
        targets = data.get('targets', [])
        target = targets[index] if targets and index < len(targets) else "Not specified"
        
        # Get LLM evaluation if available
        evaluation = None
        if 'metrics' in results and 'llm_evaluation' in results['metrics']:
            evaluations = results['metrics']['llm_evaluation'].get('evaluations', [])
            if index < len(evaluations):
                evaluation = evaluations[index]
        
        # Print the results
        print("\n" + "="*80)
        print(f"ENTRY #{index}")
        print("="*80)
        print(f"TARGET: {target}")
        print("-"*80)
        print("HATE SPEECH:")
        print(hate_speech)
        print("-"*80)
        print("COUNTER NARRATIVE:")
        print(counter_narrative)
        
        # Print evaluation metrics if available
        if evaluation:
            print("-"*80)
            print("LLM EVALUATION:")
            print(f"Overall Score: {evaluation.get('overall_score', 'N/A')}")
            print(f"Effectiveness: {evaluation.get('effectiveness', 'N/A')}")
            print(f"Relevance: {evaluation.get('relevance', 'N/A')}")
            print(f"Evidence Quality: {evaluation.get('evidence_quality', 'N/A')}")
            print(f"Persuasiveness: {evaluation.get('persuasiveness', 'N/A')}")
            print(f"Directness: {evaluation.get('directness', 'N/A')}")
            print(f"Feedback: {evaluation.get('feedback', 'No feedback provided')}")
        
        # Print toxicity scores if available
        if 'metrics' in results and 'toxicity' in results['metrics']:
            toxicity = results['metrics']['toxicity']
            if 'hate_scores' in toxicity and index < len(toxicity['hate_scores']):
                print("-"*80)
                print("HATE SPEECH TOXICITY:")
                for attr, score in toxicity['hate_scores'][index].items():
                    if attr != 'error':
                        print(f"{attr.title()}: {score:.4f}")
            
            if 'counter_scores' in toxicity and index < len(toxicity['counter_scores']):
                print("-"*80)
                print("COUNTER NARRATIVE TOXICITY:")
                for attr, score in toxicity['counter_scores'][index].items():
                    if attr != 'error':
                        print(f"{attr.title()}: {score:.4f}")
        
        print("="*80)
        
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file at {json_path}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find hate speech and counter narrative by index")
    parser.add_argument("json_file", help="Path to the evaluation results JSON file")
    parser.add_argument("index", type=int, help="Index of the entry to find")
    
    args = parser.parse_args()
    find_entry_by_index(args.json_file, args.index)