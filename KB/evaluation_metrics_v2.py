"""
Enhanced evaluation system for hate speech counter-narratives.
Includes comprehensive toxicity metrics for both hate speech and counter-narratives.
"""

import os
import json
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any, Optional
from together import Together
import requests
import re
from gen2 import HateAssessmentSystem
from retrieve import RAGRetriever


class EvaluationMetrics:
    """
    An evaluation system for counter-narrative generation that implements:
    - LLM-based evaluation (using TogetherAI models)
    - Toxicity (via Perspective API)
    - Toxicity reduction measurement
    """
    
    def __init__(self, perspective_api_key: str = None, together_api_key: str = None, 
                 model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"):
        """
        Initialize the evaluation metrics system.
        
        Args:
            perspective_api_key: API key for Google's Perspective API
            together_api_key: API key for TogetherAI
            model_name: Name of the model on TogetherAI platform
        """
        self.perspective_api_key = perspective_api_key
        self.together_api_key = together_api_key
        self.model_name = model_name
        
        # Initialize TogetherAI client if API key is provided
        if together_api_key:
            os.environ["TOGETHER_API_KEY"] = together_api_key
            self.together_client = Together()
            self.llm_available = True
        elif "TOGETHER_API_KEY" in os.environ:
            self.together_client = Together()
            self.llm_available = True
        else:
            self.together_client = None
            self.llm_available = False
    
    def get_toxicity_scores(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Get comprehensive toxicity scores using the Perspective API.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries with toxicity scores for each text
        """
        if not self.perspective_api_key:
            raise ValueError("Perspective API key is required for toxicity scoring")
        
        base_url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
        results = []
        
        for text in texts:
            # Prepare request with expanded attributes
            payload = {
                'comment': {'text': text},
                'languages': ['en'],
                'requestedAttributes': {
                    'TOXICITY': {},
                    'SEVERE_TOXICITY': {},
                    'IDENTITY_ATTACK': {},
                    'INSULT': {},
                    'THREAT': {},
                    'PROFANITY': {},
                    'SEXUALLY_EXPLICIT': {},
                    'FLIRTATION': {}
                }
            }
            
            params = {
                'key': self.perspective_api_key
            }
            
            # Make API request
            try:
                response = requests.post(base_url, params=params, json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Extract scores
                scores = {}
                for attribute, score_data in data.get('attributeScores', {}).items():
                    scores[attribute.lower()] = score_data['summaryScore']['value']
                
                results.append(scores)
                
                # Respect rate limits
                time.sleep(1)  # Sleep to avoid exceeding rate limits
                
            except requests.RequestException as e:
                # Handle API errors
                print(f"Error getting toxicity score: {e}")
                results.append({'error': str(e)})
                time.sleep(2)  # Sleep longer on error
        
        return results
    
    def evaluate_with_llm(self, hate_texts: List[str], 
                         counter_texts: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate counter-narratives using an LLM judge via TogetherAI.
        
        Args:
            hate_texts: List of hate speech texts
            counter_texts: List of generated counter-narrative texts
            
        Returns:
            List of dictionaries with evaluation scores for each counter-narrative
        """
        if not self.llm_available:
            raise ValueError("TogetherAI API key is required for LLM evaluation")
        
        results = []
        
        for i, (hate, counter) in enumerate(zip(hate_texts, counter_texts)):
            # Construct prompt for LLM evaluation
            prompt = self._create_evaluation_prompt(hate, counter)
            
            # Make API call with retry logic for rate limits
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Get evaluation from TogetherAI
                    response = self.together_client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,  # Lower temperature for more consistent evaluation
                        max_tokens=512
                    )
                    
                    # Extract the response text
                    response_text = response.choices[0].message.content
                    
                    # Extract structured evaluation from response
                    evaluation = self._parse_llm_evaluation(response_text)
                    
                    # Add raw response for debugging
                    evaluation["raw_response"] = response_text
                    
                    results.append(evaluation)
                    
                    # Add a short delay to avoid rate limits
                    time.sleep(0.5)
                    break  # Break out of retry loop on success
                    
                except Exception as e:
                    retry_count += 1
                    print(f"Error in LLM evaluation: {e}, retrying ({retry_count}/{max_retries})...")
                    
                    if "rate_limit" in str(e).lower():
                        time.sleep(5)  # Wait 5 seconds before retrying on rate limit
                    else:
                        time.sleep(1)
                    
                    # If all retries failed, add placeholder result
                    if retry_count == max_retries:
                        results.append({
                            "error": str(e),
                            "directness": 0,
                            "evidence_quality": 0,
                            "persuasiveness": 0,
                            "relevance": 0,
                            "effectiveness": 0,
                            "raw_response": "Error: API failed to respond"
                        })
        
        return results
    
    def _create_evaluation_prompt(self, hate_text: str, counter_text: str) -> str:
        """
        Create a prompt for the LLM to evaluate counter-narratives with thesis-specific metrics.
        
        Args:
            hate_text: The hate speech text
            counter_text: The generated counter-narrative text
            
        Returns:
            Prompt string for the LLM
        """
        prompt = f"""You are a highly detailed expert evaluator of counter-narratives to hate speech. You are known for your exceptionally rigorous standards, critical eye, and harsh but fair assessments. You never overrate content and are not afraid to point out even minor flaws.
 Evaluate how effectively the counter-narrative addresses the hate speech.

HATE SPEECH:
"{hate_text}"

COUNTER-NARRATIVE:
"{counter_text}"

Please evaluate the counter-narrative based on the following criteria, scoring each one from 1-10:

1. Directness (1-10): How directly does the counter-narrative address the specific claims in the hate speech?
2. Evidence Quality (1-10): How well does the counter-narrative use facts, statistics, or logical reasoning?
3. Persuasiveness (1-10): How compelling and convincing is the counter-narrative in changing minds?
4. Relevance (1-10): How relevant is the counter-narrative to the topic of the hate speech?
5. Effectiveness (1-10): How effective is the counter-narrative at countering the harmful narrative overall?
6. Provide a brief critical feedback on the counter-narrative's strengths and weaknesses (2-3 sentences).

Format your response as JSON:
{{
  "directness": <1-10>,
  "evidence_quality": <1-10>,
  "persuasiveness": <1-10>,
  "relevance": <1-10>,
  "effectiveness": <1-10>,
  "feedback": "<your critical feedback>"
}}"""
        return prompt
    
    def _parse_llm_evaluation(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse the LLM's evaluation response into a structured format.
        
        Args:
            llm_response: The raw response from the LLM
            
        Returns:
            Dictionary with parsed evaluation scores
        """
        # Try to extract JSON from the response
        try:
            # Find JSON in the response (it may be embedded in other text)
            import re
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                evaluation = json.loads(json_str)
                
                # Extract scores and feedback
                result = {
                    "directness": float(evaluation.get("directness", 0)),
                    "evidence_quality": float(evaluation.get("evidence_quality", 0)),
                    "persuasiveness": float(evaluation.get("persuasiveness", 0)),
                    "relevance": float(evaluation.get("relevance", 0)),
                    "effectiveness": float(evaluation.get("effectiveness", 0)),
                    "feedback": evaluation.get("feedback", "No feedback provided")
                }
                
                # Calculate overall score as average of metrics
                result["overall_score"] = sum([
                    result["directness"],
                    result["evidence_quality"],
                    result["persuasiveness"],
                    result["relevance"],
                    result["effectiveness"]
                ]) / 5.0
                
                return result
            
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Error parsing LLM response: {e}")
        
        # Fallback: try to extract scores using regex
        try:
            directness = float(re.search(r'Directness.*?(\d+(?:\.\d+)?)', llm_response).group(1))
            evidence_quality = float(re.search(r'Evidence Quality.*?(\d+(?:\.\d+)?)', llm_response).group(1))
            persuasiveness = float(re.search(r'Persuasiveness.*?(\d+(?:\.\d+)?)', llm_response).group(1))
            relevance = float(re.search(r'Relevance.*?(\d+(?:\.\d+)?)', llm_response).group(1))
            effectiveness = float(re.search(r'Effectiveness.*?(\d+(?:\.\d+)?)', llm_response).group(1))
            
            # Try to extract feedback
            feedback_match = re.search(r'feedback.*?["\']?(.*?)["\']?(?=\}|$)', llm_response, re.IGNORECASE | re.DOTALL)
            feedback = feedback_match.group(1).strip() if feedback_match else "No feedback extracted"
            
            result = {
                "directness": directness,
                "evidence_quality": evidence_quality,
                "persuasiveness": persuasiveness,
                "relevance": relevance,
                "effectiveness": effectiveness,
                "feedback": feedback,
                "overall_score": (directness + evidence_quality + persuasiveness + relevance + effectiveness) / 5.0
            }
            
            return result
            
        except (AttributeError, ValueError) as e:
            print(f"Error extracting scores from LLM response: {e}")
        
        # If all parsing attempts fail, return default values
        return {
            "directness": 0,
            "evidence_quality": 0,
            "persuasiveness": 0,
            "relevance": 0,
            "effectiveness": 0,
            "feedback": "Failed to extract evaluation",
            "overall_score": 0
        }
    
    def calculate_toxicity_reduction(self, hate_toxicity: List[Dict[str, float]], 
                                    counter_toxicity: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Calculate metrics showing the reduction in toxicity from hate speech to counter-narratives.
        
        Args:
            hate_toxicity: List of toxicity scores for hate speech
            counter_toxicity: List of toxicity scores for counter-narratives
            
        Returns:
            Dictionary with toxicity reduction metrics
        """
        results = {
            "avg_reduction": {},
            "avg_reduction_percent": {},
            "samples_with_reduction": {}
        }
        
        # Skip if either list is empty or contains errors
        if not hate_toxicity or not counter_toxicity:
            return results
        
        # Get all toxicity attributes (keys)
        all_attrs = set()
        for scores in hate_toxicity + counter_toxicity:
            if 'error' not in scores:
                all_attrs.update(scores.keys())
        
        # Calculate reductions for each attribute
        for attr in all_attrs:
            # Calculate raw reduction
            reductions = []
            percent_reductions = []
            samples_with_reduction = 0
            
            for i, (hate, counter) in enumerate(zip(hate_toxicity, counter_toxicity)):
                if attr in hate and attr in counter and 'error' not in hate and 'error' not in counter:
                    hate_score = hate[attr]
                    counter_score = counter[attr]
                    
                    reduction = hate_score - counter_score
                    reductions.append(reduction)
                    
                    # Calculate percent reduction, avoiding division by zero
                    if hate_score > 0:
                        percent_reduction = (reduction / hate_score) * 100
                    else:
                        percent_reduction = 0 if counter_score == 0 else -100  # -100% if counter has toxicity but hate doesn't
                        
                    percent_reductions.append(percent_reduction)
                    
                    # Count samples with positive reduction
                    if reduction > 0:
                        samples_with_reduction += 1
            
            # Calculate averages
            if reductions:
                results["avg_reduction"][attr] = sum(reductions) / len(reductions)
                results["avg_reduction_percent"][attr] = sum(percent_reductions) / len(percent_reductions)
                results["samples_with_reduction"][attr] = samples_with_reduction / len(reductions)
        
        return results
    
    def evaluate_generation(self, 
                           hate_texts: List[str],
                           generated_counters: List[str]) -> Dict[str, Any]:
        """
        Comprehensive evaluation of generated counter-narratives.
        Stores toxicity metrics for both hate speech and counter-narratives,
        and calculates toxicity reduction metrics.
        
        Args:
            hate_texts: List of hate speech texts
            generated_counters: List of generated counter-narrative texts
            
        Returns:
            Dictionary with all evaluation metrics
        """
        results = {}
        
        # LLM-based evaluation
        print("Running LLM-based evaluation...")
        llm_evaluations = self.evaluate_with_llm(
            hate_texts=hate_texts,
            counter_texts=generated_counters
        )
        
        # Calculate average scores across all examples
        avg_directness = sum(eval.get("directness", 0) for eval in llm_evaluations) / len(llm_evaluations) if llm_evaluations else 0
        avg_evidence_quality = sum(eval.get("evidence_quality", 0) for eval in llm_evaluations) / len(llm_evaluations) if llm_evaluations else 0
        avg_persuasiveness = sum(eval.get("persuasiveness", 0) for eval in llm_evaluations) / len(llm_evaluations) if llm_evaluations else 0
        avg_relevance = sum(eval.get("relevance", 0) for eval in llm_evaluations) / len(llm_evaluations) if llm_evaluations else 0
        avg_effectiveness = sum(eval.get("effectiveness", 0) for eval in llm_evaluations) / len(llm_evaluations) if llm_evaluations else 0
        avg_overall_score = sum(eval.get("overall_score", 0) for eval in llm_evaluations) / len(llm_evaluations) if llm_evaluations else 0
        
        results['llm_evaluation'] = {
            'evaluations': llm_evaluations,
            'avg_directness': avg_directness,
            'avg_evidence_quality': avg_evidence_quality,
            'avg_persuasiveness': avg_persuasiveness,
            'avg_relevance': avg_relevance,
            'avg_effectiveness': avg_effectiveness,
            'avg_overall_score': avg_overall_score
        }
        
        # Get toxicity scores if API key is available
        if self.perspective_api_key:
            print("Running toxicity evaluation for counter-narratives...")
            counter_toxicity_scores = self.get_toxicity_scores(generated_counters)
            
            # Also evaluate the hate speech texts for toxicity
            print("Running toxicity evaluation for hate speech...")
            hate_toxicity_scores = self.get_toxicity_scores(hate_texts)
            
            # Calculate average toxicity for counter-narratives
            avg_counter_toxicity = {}
            if counter_toxicity_scores and 'error' not in counter_toxicity_scores[0]:
                for key in counter_toxicity_scores[0].keys():
                    avg_counter_toxicity[key] = sum(score.get(key, 0) for score in counter_toxicity_scores if 'error' not in score) / len(counter_toxicity_scores)
            
            # Calculate average toxicity for hate speech
            avg_hate_toxicity = {}
            if hate_toxicity_scores and 'error' not in hate_toxicity_scores[0]:
                for key in hate_toxicity_scores[0].keys():
                    avg_hate_toxicity[key] = sum(score.get(key, 0) for score in hate_toxicity_scores if 'error' not in score) / len(hate_toxicity_scores)
            
            # Calculate toxicity reduction metrics
            toxicity_reduction = self.calculate_toxicity_reduction(
                hate_toxicity=hate_toxicity_scores,
                counter_toxicity=counter_toxicity_scores
            )
            
            # Store all toxicity-related metrics
            results['toxicity'] = {
                'counter_scores': counter_toxicity_scores,
                'hate_scores': hate_toxicity_scores,
                'avg_counter_scores': avg_counter_toxicity,
                'avg_hate_scores': avg_hate_toxicity,
                'reduction_metrics': toxicity_reduction
            }
        
        return results


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
        # For generation, we'll use HateAssessmentSystem with TogetherAI
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
                'COUNTER_NARRATIVE': 'counter_narrative',
                'TARGET': 'target' if 'TARGET' in df.columns else None
            })
        
        # Ensure all required columns exist
        required_cols = ['hate_speech']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset missing required columns: {missing_cols}")
        
        print(f"Loaded dataset with {len(df)} examples")
        return df
    
    def generate_counter_narratives(self, 
                                    hate_texts: List[str], 
                                    search_method: str = "semantic",
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
                print(f"Hate score: {result['hate_score']}")
                print(f"Mitigation level: {result['mitigation_level']}")
                print("-" * 50)
            
            generated_counters.extend(batch_counters)
        
        return generated_counters
    
    def run_full_evaluation(self, 
                           dataset_path: str, 
                           output_path: str,
                           sample_size: Optional[int] = None,
                           search_method: str = "semantic") -> Dict[str, Any]:
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
        
        # Extract hate texts
        hate_texts = dataset['hate_speech'].tolist()
        
        # Extract gold standard counter-narratives if available
        gold_counters = dataset['counter_narrative'].tolist() if 'counter_narrative' in dataset.columns else None
        
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
            generated_counters=generated_counters
        )
        
        # # If gold standard counter-narratives are available, evaluate them too
        # if gold_counters:
        #     print("Also evaluating gold standard counter-narratives...")
        #     gold_evaluation = self.metrics.evaluate_generation(
        #         hate_texts=hate_texts,
        #         generated_counters=gold_counters
        #     )
        #     evaluation_results['gold_standard'] = gold_evaluation
        
        # Add generated counters to results
        full_results = {
            'config': {
                'dataset': dataset_path,
                'sample_size': len(dataset),
                'search_method': search_method
            },
            'data': {
                'hate_texts': hate_texts,
                'generated_counters': generated_counters
            },
            'metrics': evaluation_results
        }
        
        # If we have target information, add it to the results
        if 'target' in dataset.columns:
            full_results['data']['targets'] = dataset['target'].tolist()
        
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
            f.write(f"- **Average Evidence Quality**: {metrics['llm_evaluation']['avg_evidence_quality']:.2f}\n")
            f.write(f"- **Average Persuasiveness**: {metrics['llm_evaluation']['avg_persuasiveness']:.2f}\n")
            f.write(f"- **Average Directness**: {metrics['llm_evaluation']['avg_directness']:.2f}\n")
            f.write("\n")
            
            # Toxicity metrics - enhanced with both hate speech and counter-narratives
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
                
                # Toxicity reduction metrics
                f.write("#### Toxicity Reduction\n\n")
                f.write("##### Average Absolute Reduction\n\n")
                for attr, reduction in metrics['toxicity']['reduction_metrics']['avg_reduction'].items():
                    f.write(f"- **{attr.title()}**: {reduction:.4f}\n")
                f.write("\n")
                
                f.write("##### Average Percent Reduction\n\n")
                for attr, reduction in metrics['toxicity']['reduction_metrics']['avg_reduction_percent'].items():
                    f.write(f"- **{attr.title()}**: {reduction:.2f}%\n")
                f.write("\n")
                
                f.write("##### Percentage of Samples with Toxicity Reduction\n\n")
                for attr, pct in metrics['toxicity']['reduction_metrics']['samples_with_reduction'].items():
                    f.write(f"- **{attr.title()}**: {pct*100:.2f}%\n")
                f.write("\n")
            
            # Add comparison with gold standard if available
            if 'gold_standard' in metrics:
                f.write("### Comparison with Gold Standard\n\n")
                f.write("#### LLM Evaluation\n\n")
                
                # Create comparison table
                f.write("| Metric | Generated | Gold Standard | Difference |\n")
                f.write("|--------|-----------|---------------|------------|\n")
                
                gen_scores = metrics['llm_evaluation']
                gold_scores = metrics['gold_standard']['llm_evaluation']
                
                for metric in ['avg_overall_score', 'avg_effectiveness', 'avg_relevance', 'avg_evidence_quality', 'avg_persuasiveness', 'avg_directness']:
                    gen_value = gen_scores.get(metric, 0)
                    gold_value = gold_scores.get(metric, 0)
                    diff = gen_value - gold_value
                    diff_str = f"{diff:.2f}" if abs(diff) < 0.01 else f"{diff:+.2f}"
                    
                    metric_name = metric.replace('avg_', '').replace('_', ' ').title()
                    f.write(f"| {metric_name} | {gen_value:.2f} | {gold_value:.2f} | {diff_str} |\n")
                
                f.write("\n")
                
                # Toxicity comparison if available
                if 'toxicity' in metrics['gold_standard']:
                    f.write("#### Toxicity Comparison\n\n")
                    f.write("| Attribute | Generated | Gold Standard | Difference |\n")
                    f.write("|-----------|-----------|---------------|------------|\n")
                    
                    gen_tox = metrics['toxicity']['avg_counter_scores']
                    gold_tox = metrics['gold_standard']['toxicity']['avg_counter_scores']
                    
                    for attr in gen_tox.keys():
                        if attr in gold_tox:
                            gen_value = gen_tox[attr]
                            gold_value = gold_tox[attr]
                            diff = gen_value - gold_value
                            diff_str = f"{diff:.4f}" if abs(diff) < 0.0001 else f"{diff:+.4f}"
                            
                            attr_name = attr.title()
                            f.write(f"| {attr_name} | {gen_value:.4f} | {gold_value:.4f} | {diff_str} |\n")
                    
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
                if 'gold_standard' in metrics and 'counter_narrative' in results['data']:
                    f.write(f"**Gold Standard Counter-Narrative**:\n> {results['data']['counter_narrative'][idx]}\n\n")
                
                # Add LLM evaluation for this example
                if 'llm_evaluation' in metrics and idx < len(metrics['llm_evaluation']['evaluations']):
                    eval_data = metrics['llm_evaluation']['evaluations'][idx]
                    f.write("**LLM Evaluation**:\n")
                    f.write(f"- Overall Score: {eval_data.get('overall_score', 'N/A')}\n")
                    f.write(f"- Effectiveness: {eval_data.get('effectiveness', 'N/A')}\n")
                    f.write(f"- Relevance: {eval_data.get('relevance', 'N/A')}\n")
                    f.write(f"- Evidence Quality: {eval_data.get('evidence_quality', 'N/A')}\n")
                    f.write(f"- Persuasiveness: {eval_data.get('persuasiveness', 'N/A')}\n")
                    f.write(f"- Directness: {eval_data.get('directness', 'N/A')}\n")
                    f.write(f"- Feedback: {eval_data.get('feedback', 'No feedback provided')}\n\n")
                
                # Add toxicity scores for both counter-narrative and hate speech
                if 'toxicity' in metrics:
                    # Counter-narrative toxicity
                    if idx < len(metrics['toxicity']['counter_scores']):
                        counter_tox = metrics['toxicity']['counter_scores'][idx]
                        f.write("**Counter-Narrative Toxicity**:\n")
                        for attr, score in counter_tox.items():
                            if attr != 'error':
                                f.write(f"- {attr.title()}: {score:.4f}\n")
                        f.write("\n")
                    
                    # Hate speech toxicity
                    if idx < len(metrics['toxicity']['hate_scores']):
                        hate_tox = metrics['toxicity']['hate_scores'][idx]
                        f.write("**Hate Speech Toxicity**:\n")
                        for attr, score in hate_tox.items():
                            if attr != 'error':
                                f.write(f"- {attr.title()}: {score:.4f}\n")
                        f.write("\n")
                
                # Add separator between examples
                f.write("\n---\n\n")
        
        print(f"Comparison report generated at {report_path}")
        
    def export_metrics_csv(self, results_path: str, csv_path: str):
        """
        Export key metrics to CSV for easier analysis in spreadsheets or statistical tools
        
        Args:
            results_path: Path to the evaluation results JSON
            csv_path: Path to save the CSV file
        """
        # Load results
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        metrics = results['metrics']
        
        # Create data for CSV
        rows = []
        
        # Process each example
        for i in range(len(results['data']['hate_texts'])):
            row = {
                'example_id': i,
                'hate_text': results['data']['hate_texts'][i],
                'counter_text': results['data']['generated_counters'][i]
            }
            
            # Add target if available
            if 'targets' in results['data']:
                row['target'] = results['data']['targets'][i]
            
            # Add LLM evaluation scores if available
            if 'llm_evaluation' in metrics and i < len(metrics['llm_evaluation']['evaluations']):
                eval_data = metrics['llm_evaluation']['evaluations'][i]
                for key in ['overall_score', 'effectiveness', 'relevance', 'evidence_quality', 
                           'persuasiveness', 'directness']:
                    row[f'llm_{key}'] = eval_data.get(key, '')
            
            # Add toxicity scores if available
            if 'toxicity' in metrics:
                # Counter narrative toxicity
                if i < len(metrics['toxicity']['counter_scores']):
                    counter_tox = metrics['toxicity']['counter_scores'][i]
                    for attr, score in counter_tox.items():
                        if attr != 'error':
                            row[f'counter_{attr}'] = score
                
                # Hate speech toxicity
                if i < len(metrics['toxicity']['hate_scores']):
                    hate_tox = metrics['toxicity']['hate_scores'][i]
                    for attr, score in hate_tox.items():
                        if attr != 'error':
                            row[f'hate_{attr}'] = score
                            
                    # Calculate and add reduction for each attribute
                    if i < len(metrics['toxicity']['counter_scores']):
                        counter_tox = metrics['toxicity']['counter_scores'][i]
                        for attr in hate_tox.keys():
                            if attr in counter_tox and attr != 'error':
                                reduction = hate_tox[attr] - counter_tox[attr]
                                row[f'reduction_{attr}'] = reduction
                                
                                # Add percent reduction
                                if hate_tox[attr] > 0:
                                    pct_reduction = (reduction / hate_tox[attr]) * 100
                                    row[f'pct_reduction_{attr}'] = pct_reduction
            
            rows.append(row)
        
        # Convert to DataFrame and save as CSV
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        
        print(f"Metrics exported to CSV: {csv_path}")

# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = HateCounterEvaluator(
        perspective_api_key=os.environ.get("PERSPECTIVE_API_KEY"),
        together_api_key=os.environ.get("TOGETHER_API_KEY"),
        llm_model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
    )
    
    # Run evaluation on MultitargetCONAN dataset
    results = evaluator.run_full_evaluation(
        dataset_path="/Users/erfanbayat/Documents/SPAG/KB/subsampled_data.csv",
        output_path="/Users/erfanbayat/Downloads/full_eval/gemmaxgemma_results.json",
        #sample_size=1,  # Use a small sample for testing
        search_method="semantic"
    )
    
    # Generate report
    evaluator.generate_comparison_report(
        results_path="/Users/erfanbayat/Downloads/full_eval/gemmaxgemma_results.json",
        report_path="/Users/erfanbayat/Downloads/full_eval/evaluation_report.md"
    )
    
    # Export metrics to CSV
    evaluator.export_metrics_csv(
        results_path="/Users/erfanbayat/Downloads/full_eval/gemmaxgemma_results.json",
        csv_path="/Users/erfanbayat/Downloads/full_eval/evaluation_metrics.csv"
    )
