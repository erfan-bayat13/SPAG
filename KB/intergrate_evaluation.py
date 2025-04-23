import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from evaluation_metrics import EvaluationMetrics
from retrieve import RAGRetriever
from gen2 import HateAssessmentSystem, TogetherAIPlayer

class HateCounterEvaluator:
    """
    System to evaluate hate speech counter-narrative generation
    with LLM judge evaluation and toxicity metrics using TogetherAI.
    """
    
    def __init__(self, 
                 perspective_api_key: str,
                 together_api_key: str,
                 llm_model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
                 memgraph_host: str = "127.0.0.1", 
                 memgraph_port: int = 7687):
        """
        Initialize the counter-narrative evaluation system.
        
        Args:
            perspective_api_key: API key for Google's Perspective API
            together_api_key: API key for TogetherAI
            llm_model_name: Name of the model on TogetherAI platform for evaluation
            memgraph_host: Host address for Memgraph
            memgraph_port: Port for Memgraph connection
        """
        # Initialize metrics evaluation with TogetherAI
        self.metrics = EvaluationMetrics(
            perspective_api_key=perspective_api_key,
            together_api_key=together_api_key,
            model_name=llm_model_name
        )
        
        # Initialize retrieval and generation components
        self.retriever = RAGRetriever(memgraph_host, memgraph_port)
        # For generation, we'll use TogetherAIPlayer or HateAssessmentSystem with TogetherAI
        self.generator = HateAssessmentSystem(self.retriever, api_key=together_api_key)
        
        # Storage for evaluation results
        self.results_cache = {}
    
    def load_dataset(self, csv_path: str) -> pd.DataFrame:
        """
        Load benchmark dataset for evaluation
        
        Args:
            csv_path: Path to CSV with hate speech and counter-narratives
            
        Returns:
            DataFrame with the data
        """
        # Load dataset
        df = pd.read_csv(csv_path)
        
        # Rename columns to consistent naming scheme if needed
        if 'HATE_SPEECH' in df.columns and 'COUNTER_NARRATIVE' in df.columns:
            df = df.rename(columns={
                'HATE_SPEECH': 'hate_speech',
                'COUNTER_NARRATIVE': 'counter_narrative'
            })
        
        # Ensure all required columns exist
        required_cols = ['hate_speech', 'counter_narrative']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset missing required columns: {missing_cols}")
        
        print(f"Loaded dataset with {len(df)} examples")
        return df
    
    def generate_counter_narratives(self, 
                                    hate_texts: List[str], 
                                    search_method: str = "hybrid",
                                    batch_size: int = 10) -> List[str]:
        """
        Generate counter-narratives for the given hate texts
        
        Args:
            hate_texts: List of hate speech texts
            search_method: Search method to use ("hybrid", "semantic", "syntax")
            batch_size: Batch size for processing
            
        Returns:
            List of generated counter-narratives
        """
        generated_counters = []
        
        # Process in batches
        for i in range(0, len(hate_texts), batch_size):
            batch = hate_texts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(hate_texts) + batch_size - 1)//batch_size}")
            
            # Generate counter-narratives for each hate text in batch
            batch_counters = []
            for hate_text in batch:
                # Generate response using the assessment system
                result = self.generator.process_query(
                    user_query=hate_text,
                    search_method=search_method,
                    num_results=5
                )
                
                # Extract the generated counter-narrative
                counter_narrative = result['response']
                batch_counters.append(counter_narrative)
                
                # Print for monitoring
                print(f"Hate text: {hate_text[:50]}...")
                print(f"Generated: {counter_narrative[:50]}...")
                print("-" * 50)
            
            generated_counters.extend(batch_counters)
        
        return generated_counters
    
    def run_full_evaluation(self, 
                           dataset_path: str, 
                           output_path: str,
                           sample_size: Optional[int] = None,
                           search_method: str = "hybrid") -> Dict[str, Any]:
        """
        Run a complete evaluation using a dataset and save results
        
        Args:
            dataset_path: Path to CSV dataset
            output_path: Path to save results
            sample_size: Optional number of samples to use (for testing)
            search_method: Search method to use
            
        Returns:
            Dictionary with evaluation results
        """
        # Load the dataset
        dataset = self.load_dataset(dataset_path)
        
        # Take a sample if specified
        if sample_size and sample_size < len(dataset):
            print(f"Using sample of {sample_size} examples")
            dataset = dataset.sample(sample_size, random_state=42)
        
        # Extract hate texts and reference counter-narratives
        hate_texts = dataset['hate_speech'].tolist()
        reference_counters = dataset['counter_narrative'].tolist()
        
        # Generate counter-narratives
        print(f"Generating counter-narratives using {search_method} search")
        generated_counters = self.generate_counter_narratives(
            hate_texts=hate_texts,
            search_method=search_method
        )
        
        # Run evaluation metrics
        print("Running LLM-based evaluation and toxicity metrics...")
        evaluation_results = self.metrics.evaluate_generation(
            hate_texts=hate_texts,
            generated_counters=generated_counters,
            reference_counters=reference_counters
        )
        
        # Add generated counters to results
        full_results = {
            'config': {
                'dataset': dataset_path,
                'sample_size': len(dataset),
                'search_method': search_method
            },
            'data': {
                'hate_texts': hate_texts,
                'reference_counters': reference_counters,
                'generated_counters': generated_counters
            },
            'metrics': evaluation_results
        }
        
        # Save results
        with open(output_path, 'w') as f:
            # Convert any numpy types to Python native types for JSON serialization
            json.dump(full_results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'dtype') else x)
        
        print(f"Evaluation completed. Results saved to {output_path}")
        return full_results
    
    def generate_comparison_report(self, results_path: str, report_path: str):
        """
        Generate a human-readable report comparing the evaluation results
        
        Args:
            results_path: Path to the evaluation results JSON
            report_path: Path to save the report
        """
        # Load results
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        metrics = results['metrics']
        
        # Create a markdown report
        with open(report_path, 'w') as f:
            f.write("# Counter-Narrative Generation Evaluation Report\n\n")
            
            # Write configuration
            f.write("## Configuration\n\n")
            for key, value in results['config'].items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")
            
            # Write metrics summary
            f.write("## Metrics Summary\n\n")
            
            # LLM Evaluation
            f.write("### LLM Judge Evaluation\n\n")
            f.write(f"- **Average Overall Score**: {metrics['llm_evaluation']['avg_overall_score']:.2f}\n")
            f.write(f"- **Average Effectiveness**: {metrics['llm_evaluation']['avg_effectiveness']:.2f}\n")
            f.write(f"- **Average Relevance**: {metrics['llm_evaluation']['avg_relevance']:.2f}\n")
            f.write(f"- **Average Persuasiveness**: {metrics['llm_evaluation']['avg_persuasiveness']:.2f}\n")
            f.write("\n")
            
            # Toxicity
            if 'toxicity' in metrics:
                f.write("### Toxicity Results\n\n")
                for attr, score in metrics['toxicity']['avg_scores'].items():
                    f.write(f"- **{attr.title()}**: {score:.4f}\n")
                f.write("\n")
            
            # Write examples
            f.write("## Example Outputs\n\n")
            
            # Get 5 random examples
            import random
            indices = random.sample(range(len(results['data']['hate_texts'])), min(5, len(results['data']['hate_texts'])))
            
            for i, idx in enumerate(indices):
                hate = results['data']['hate_texts'][idx]
                reference = results['data']['reference_counters'][idx]
                generated = results['data']['generated_counters'][idx]
                
                f.write(f"### Example {i+1}\n\n")
                f.write(f"**Hate Speech**:\n> {hate}\n\n")
                f.write(f"**Generated Counter-Narrative**:\n> {generated}\n\n")
                f.write(f"**Reference Counter-Narrative**:\n> {reference}\n\n")
                
                # Add LLM evaluation for this example
                if 'llm_evaluation' in metrics and idx < len(metrics['llm_evaluation']['evaluations']):
                    eval_data = metrics['llm_evaluation']['evaluations'][idx]
                    f.write("**LLM Evaluation**:\n")
                    f.write(f"- Overall Score: {eval_data.get('overall_score', 'N/A')}\n")
                    f.write(f"- Effectiveness: {eval_data.get('effectiveness', 'N/A')}\n")
                    f.write(f"- Relevance: {eval_data.get('relevance', 'N/A')}\n")
                    f.write(f"- Persuasiveness: {eval_data.get('persuasiveness', 'N/A')}\n")
                    f.write(f"- Feedback: {eval_data.get('feedback', 'No feedback provided')}\n")
                
                # Add toxicity scores if available
                if 'toxicity' in metrics and idx < len(metrics['toxicity']['scores']):
                    tox_data = metrics['toxicity']['scores'][idx]
                    f.write("\n**Toxicity Scores**:\n")
                    for attr, score in tox_data.items():
                        if attr != 'error':
                            f.write(f"- {attr.title()}: {score:.4f}\n")
                
                f.write("\n")
        
        print(f"Comparison report generated at {report_path}")

# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = HateCounterEvaluator(
        perspective_api_key=os.environ.get("PERSPECTIVE_API_KEY"),
        together_api_key=os.environ.get("TOGETHER_API_KEY"),
        llm_model_name="meta-llama/Llama-3.1-70B-Instruct"
    )
    
    # Run evaluation on MultitargetCONAN dataset
    results = evaluator.run_full_evaluation(
        dataset_path="MultitargetCONAN.csv",
        output_path="evaluation_results.json",
        sample_size=20,  # Use a small sample for testing
        search_method="hybrid"
    )
    
    # Generate report
    evaluator.generate_comparison_report(
        results_path="evaluation_results.json",
        report_path="evaluation_report.md"
    )