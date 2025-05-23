import os
import json
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any, Optional
from together import Together
import argparse
from tqdm import tqdm

# Import evaluation metrics for consistency
try:
    from evaluation_metrics_v2 import EvaluationMetrics
except ImportError:
    print("Warning: Could not import EvaluationMetrics, evaluation features will be limited")
    EvaluationMetrics = None

class ZeroShotGenerator:
    """
    Zero-shot counter-narrative generator using LLMs directly, without a knowledge base.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-70B-Instruct", api_key: str = None):
        """
        Initialize the zero-shot generator with an LLM.
        
        Args:
            model_name: Name of the model on TogetherAI platform
            api_key: API key for TogetherAI (optional, will use env var if not provided)
        """
        self.model_name = model_name
        
        # Set API key from args or environment
        if api_key:
            os.environ["TOGETHER_API_KEY"] = api_key
        elif "TOGETHER_API_KEY" not in os.environ:
            raise ValueError("TOGETHER_API_KEY environment variable must be set")
            
        self.client = Together()
    
    def generate_counter_narrative(self, hate_text: str, target_group: Optional[str] = None) -> str:
        """
        Generate a counter-narrative for the given hate speech text using zero-shot prompting.
        
        Args:
            hate_text: The hate speech text
            target_group: Optional target group (e.g., "immigrants", "women", etc.)
            
        Returns:
            Generated counter-narrative text
        """
        # Construct a zero-shot prompt
        prompt = self._create_zero_shot_prompt(hate_text, target_group)
        
        # Make API call with retry logic for rate limits
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=512
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
    
    def _create_zero_shot_prompt(self, hate_text: str, target_group: Optional[str] = None) -> str:
        """
        Create a zero-shot prompt for counter-narrative generation.
        
        Args:
            hate_text: The hate speech text
            target_group: Optional target group
            
        Returns:
            Formatted prompt string
        """
        # Base prompt instruction
        prompt = """You are tasked with generating counternarratives to the query. if the query is hateful, you should generate a counter-narrative.
        if not ansswer normally

The query is:
"""
        
        # Add hate speech
        prompt += f"\"{hate_text}\"\n\n"
        
        prompt += """Write a answer that Is concise (3-5 sentences)

Your counter-narrative should aim to neutralize the harmful speech while educating and promoting understanding.
Provide ONLY the counter-narrative text with no additional explanations or commentary."""

        return prompt

class CONANEvaluator:
    """
    System to evaluate zero-shot counter-narrative generation on the CONAN dataset.
    """
    
    def __init__(self, 
                 generator, 
                 evaluator=None,
                 perspective_api_key=None, 
                 together_api_key=None):
        """
        Initialize the CONAN evaluator.
        
        Args:
            generator: An instance of ZeroShotGenerator
            evaluator: Optional EvaluationMetrics instance
            perspective_api_key: API key for Google's Perspective API
            together_api_key: API key for TogetherAI
        """
        self.generator = generator
        self.evaluator = evaluator
        
        # Create evaluator if not provided and we have API keys
        if self.evaluator is None and EvaluationMetrics is not None:
            if perspective_api_key or together_api_key:
                self.evaluator = EvaluationMetrics(
                    perspective_api_key=perspective_api_key,
                    together_api_key=together_api_key
                )
    
    def load_conan_dataset(self, csv_path: str) -> pd.DataFrame:
        """
        Load the CONAN dataset.
        
        Args:
            csv_path: Path to CONAN dataset CSV
            
        Returns:
            DataFrame with the CONAN data
        """
        # Load dataset
        df = pd.read_csv(csv_path)
        
        # Rename columns to consistent naming scheme
        if 'HATE_SPEECH' in df.columns and 'COUNTER_NARRATIVE' in df.columns:
            df = df.rename(columns={
                'HATE_SPEECH': 'hate_speech',
                'COUNTER_NARRATIVE': 'counter_narrative',
                'TARGET': 'target' if 'TARGET' in df.columns else None
            })
        
        print(f"Loaded CONAN dataset with {len(df)} examples")
        return df
    
    def generate_counter_narratives(self, 
                                   dataset: pd.DataFrame, 
                                   sample_size: Optional[int] = None,
                                   batch_size: int = 10) -> List[str]:
        """
        Generate zero-shot counter-narratives for the CONAN dataset.
        
        Args:
            dataset: DataFrame with the CONAN dataset
            sample_size: Optional number of samples to use (for testing)
            batch_size: Batch size for processing
            
        Returns:
            List of generated counter-narratives
        """
        # Take a sample if specified
        if sample_size and sample_size < len(dataset):
            dataset = dataset.sample(sample_size, random_state=42)
            print(f"Using sample of {sample_size} examples")
        
        # Extract hate texts and targets
        hate_texts = dataset['hate_speech'].tolist()
        targets = dataset['target'].tolist() if 'target' in dataset.columns else [None] * len(hate_texts)
        
        generated_counters = []
        
        # Process in batches
        total_batches = (len(hate_texts) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(total_batches), desc="Generating counter-narratives"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(hate_texts))
            
            batch_hate_texts = hate_texts[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]
            
            # Generate counter-narratives for each hate text in batch
            batch_counters = []
            for hate_text, target in zip(batch_hate_texts, batch_targets):
                # Generate counter-narrative
                counter_narrative = self.generator.generate_counter_narrative(hate_text, target)
                batch_counters.append(counter_narrative)
                
                # Add a small delay to avoid rate limits
                time.sleep(0.5)
            
            generated_counters.extend(batch_counters)
            
            # Log progress intermittently
            if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
                print(f"Processed {end_idx}/{len(hate_texts)} examples")
        
        return generated_counters
    
    def evaluate_generation(self, 
                           hate_texts: List[str],
                           generated_counters: List[str],
                           gold_counters: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate the generated counter-narratives.
        
        Args:
            hate_texts: List of hate speech texts
            generated_counters: List of generated counter-narrative texts
            gold_counters: Optional list of gold standard counter-narratives
            
        Returns:
            Dictionary with evaluation results
        """
        if self.evaluator is None:
            print("Warning: No evaluator available, skipping comprehensive evaluation")
            return {"message": "No evaluator available"}
        
        # Evaluate generated counter-narratives
        results = self.evaluator.evaluate_generation(
            hate_texts=hate_texts,
            generated_counters=generated_counters
        )
        
        # If gold counter-narratives are available, evaluate them too
        if gold_counters:
            gold_results = self.evaluator.evaluate_generation(
                hate_texts=hate_texts,
                generated_counters=gold_counters
            )
            results["gold_standard"] = gold_results
        
        return results
    
    def run_experiment(self, 
                      csv_path: str, 
                      output_path: str,
                      sample_size: Optional[int] = None,
                      batch_size: int = 10) -> Dict[str, Any]:
        """
        Run a complete zero-shot generation experiment on the CONAN dataset.
        
        Args:
            csv_path: Path to CONAN dataset CSV
            output_path: Path to save results
            sample_size: Optional number of samples to use (for testing)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with experiment results
        """
        # Load the dataset
        dataset = self.load_conan_dataset(csv_path)
        
        # Take a sample if specified
        if sample_size and sample_size < len(dataset):
            dataset = dataset.sample(sample_size, random_state=42)
        
        # Extract hate texts
        hate_texts = dataset['hate_speech'].tolist()
        
        # Extract gold standard counter-narratives if available
        gold_counters = dataset['counter_narrative'].tolist() if 'counter_narrative' in dataset.columns else None
        
        # Extract targets if available
        targets = dataset['target'].tolist() if 'target' in dataset.columns else None
        
        # Generate counter-narratives
        print("Generating zero-shot counter-narratives...")
        generated_counters = self.generate_counter_narratives(
            dataset=dataset,
            batch_size=batch_size
        )
        
        # Basic reporting of generation
        basic_results = {
            'config': {
                'dataset': csv_path,
                'sample_size': len(dataset),
                'model': self.generator.model_name,
                'approach': 'zero-shot'
            },
            'data': {
                'hate_texts': hate_texts,
                'generated_counters': generated_counters
            }
        }
        
        if targets:
            basic_results['data']['targets'] = targets
        
        if gold_counters:
            basic_results['data']['gold_counters'] = gold_counters
        
        # Run evaluation if evaluator is available
        if self.evaluator is not None:
            print("Running evaluation metrics...")
            evaluation_results = self.evaluate_generation(
                hate_texts=hate_texts,
                generated_counters=generated_counters,
                gold_counters=gold_counters
            )
            
            basic_results['metrics'] = evaluation_results
            
        # Save results
        with open(output_path, 'w') as f:
            # Convert any numpy types to Python native types for JSON serialization
            json.dump(basic_results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'dtype') else x)
        
        print(f"Experiment completed. Results saved to {output_path}")
        return basic_results
    
    def generate_simple_report(self, results_path: str, report_path: str):
        """
        Generate a simple human-readable report of the experiment.
        
        Args:
            results_path: Path to the results JSON
            report_path: Path to save the report
        """
        # Load results
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Create a markdown report
        with open(report_path, 'w') as f:
            f.write("# Zero-Shot Counter-Narrative Generation Report\n\n")
            
            # Write configuration
            f.write("## Configuration\n\n")
            for key, value in results['config'].items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")
            
            # Write metrics summary if available
            if 'metrics' in results:
                metrics = results['metrics']
                f.write("## Metrics Summary\n\n")
                
                # LLM Evaluation
                if 'llm_evaluation' in metrics:
                    f.write("### LLM Judge Evaluation\n\n")
                    f.write(f"- **Average Overall Score**: {metrics['llm_evaluation']['avg_overall_score']:.2f}\n")
                    f.write(f"- **Average Effectiveness**: {metrics['llm_evaluation']['avg_effectiveness']:.2f}\n")
                    f.write(f"- **Average Relevance**: {metrics['llm_evaluation']['avg_relevance']:.2f}\n")
                    f.write(f"- **Average Evidence Quality**: {metrics['llm_evaluation']['avg_evidence_quality']:.2f}\n")
                    f.write(f"- **Average Persuasiveness**: {metrics['llm_evaluation']['avg_persuasiveness']:.2f}\n")
                    f.write(f"- **Average Directness**: {metrics['llm_evaluation']['avg_directness']:.2f}\n")
                    f.write("\n")
                
                # Toxicity metrics
                if 'toxicity' in metrics:
                    f.write("### Toxicity Results\n\n")
                    
                    # Counter-narrative toxicity
                    f.write("#### Counter-Narrative Toxicity\n\n")
                    for attr, score in metrics['toxicity']['avg_counter_scores'].items():
                        f.write(f"- **{attr.title()}**: {score:.4f}\n")
                    f.write("\n")
                    
                    # Hate speech toxicity
                    f.write("#### Hate Speech Toxicity\n\n")
                    for attr, score in metrics['toxicity']['avg_hate_scores'].items():
                        f.write(f"- **{attr.title()}**: {score:.4f}\n")
                    f.write("\n")
                    
                    # Toxicity reduction
                    f.write("#### Toxicity Reduction\n\n")
                    for attr, reduction in metrics['toxicity']['reduction_metrics']['avg_reduction'].items():
                        f.write(f"- **{attr.title()}**: {reduction:.4f}\n")
                    f.write("\n")
            
            # Write examples
            f.write("## Example Outputs\n\n")
            
            # Get 5 random examples
            import random
            indices = random.sample(range(len(results['data']['hate_texts'])), min(5, len(results['data']['hate_texts'])))
            
            for i, idx in enumerate(indices):
                hate = results['data']['hate_texts'][idx]
                generated = results['data']['generated_counters'][idx]
                
                f.write(f"### Example {i+1}\n\n")
                f.write(f"**Hate Speech**:\n> {hate}\n\n")
                f.write(f"**Generated Counter-Narrative**:\n> {generated}\n\n")
                
                # Add target information if available
                if 'targets' in results['data']:
                    f.write(f"**Target Group**: {results['data']['targets'][idx]}\n\n")
                
                # Add gold standard if available
                if 'gold_counters' in results['data']:
                    f.write(f"**Gold Standard Counter-Narrative**:\n> {results['data']['gold_counters'][idx]}\n\n")
                
                # Add separator between examples
                f.write("\n---\n\n")
        
        print(f"Report generated at {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Counter-Narrative Generation for CONAN Dataset")
    
    # Required arguments
    parser.add_argument("--dataset", type=str, required=True, help="Path to CONAN dataset CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output")
    
    # Optional arguments
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B-Instruct", 
                        help="LLM model name to use for generation")
    parser.add_argument("--together_api_key", type=str, default=None, 
                        help="TogetherAI API key (uses env var if not provided)")
    parser.add_argument("--perspective_api_key", type=str, default=None, 
                        help="Google Perspective API key (optional, for toxicity evaluation)")
    parser.add_argument("--sample_size", type=int, default=None, 
                        help="Number of examples to use (for testing)")
    parser.add_argument("--batch_size", type=int, default=10, 
                        help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize generator
    generator = ZeroShotGenerator(
        model_name=args.model,
        api_key=args.together_api_key
    )
    
    # Create evaluator with metrics if possible
    evaluator = None
    if EvaluationMetrics is not None:
        evaluator = EvaluationMetrics(
            perspective_api_key=args.perspective_api_key,
            together_api_key=args.together_api_key,
            model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo"  # Evaluation model
        )
        print("Initialized evaluation metrics")
    
    # Initialize CONAN evaluator
    conan_evaluator = CONANEvaluator(
        generator=generator,
        evaluator=evaluator
    )
    
    # Run experiment
    results_file = os.path.join(args.output_dir, "zero_shot_results.json")
    results = conan_evaluator.run_experiment(
        csv_path=args.dataset,
        output_path=results_file,
        sample_size=args.sample_size,
        batch_size=args.batch_size
    )
    
    # Generate report
    report_file = os.path.join(args.output_dir, "zero_shot_report.md")
    conan_evaluator.generate_simple_report(
        results_path=results_file,
        report_path=report_file
    )
    
    print("Zero-shot experiment complete!")


if __name__ == "__main__":
    main()