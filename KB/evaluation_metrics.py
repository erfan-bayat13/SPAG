import requests
import time
import json
import os
from typing import List, Dict, Any, Union, Optional
from together import Together

class EvaluationMetrics:
    """
    An evaluation system for counter-narrative generation that implements:
    - LLM-based evaluation (using TogetherAI models)
    - Toxicity (via Perspective API)
    """
    
    def __init__(self, perspective_api_key: str = None, together_api_key: str = None, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"):
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
        Get toxicity scores using the Perspective API.
        
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
            # Prepare request
            payload = {
                'comment': {'text': text},
                'languages': ['en'],
                'requestedAttributes': {
                    'TOXICITY': {},
                    'SEVERE_TOXICITY': {},
                    'IDENTITY_ATTACK': {},
                    'INSULT': {},
                    'THREAT': {}
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
                         counter_texts: List[str], 
                         reference_counters: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate counter-narratives using an LLM judge via TogetherAI.
        
        Args:
            hate_texts: List of hate speech texts
            counter_texts: List of generated counter-narrative texts
            reference_counters: Optional list of reference counter-narratives
            
        Returns:
            List of dictionaries with evaluation scores and feedback for each counter-narrative
        """
        if not self.llm_available:
            raise ValueError("TogetherAI API key is required for LLM evaluation")
        
        results = []
        
        for i, (hate, counter) in enumerate(zip(hate_texts, counter_texts)):
            reference = reference_counters[i] if reference_counters and i < len(reference_counters) else None
            
            # Construct prompt for LLM evaluation
            prompt = self._create_evaluation_prompt(hate, counter, reference)
            
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
                        max_tokens=1024
                    )
                    
                    # Extract the response text
                    response_text = response.choices[0].message.content
                    
                    # Extract structured evaluation from response
                    evaluation = self._parse_llm_evaluation(response_text)
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
                            "effectiveness": 0,
                            "relevance": 0,
                            "persuasiveness": 0,
                            "overall_score": 0,
                            "feedback": f"Error occurred during evaluation after {max_retries} attempts"
                        })
        
        return results
    
    def _create_evaluation_prompt(self, hate_text: str, counter_text: str, 
                               reference_counter: Optional[str] = None) -> str:
        """
        Create a prompt for the LLM to evaluate counter-narratives.
        
        Args:
            hate_text: The hate speech text
            counter_text: The generated counter-narrative text
            reference_counter: Optional reference counter-narrative
            
        Returns:
            Prompt string for the LLM
        """
        prompt = f"""You are an expert evaluator of counter-narratives to hate speech. Evaluate how effectively the counter-narrative addresses the hate speech.

HATE SPEECH:
"{hate_text}"

COUNTER-NARRATIVE:
"{counter_text}"
"""
        
        if reference_counter:
            prompt += f"""
REFERENCE COUNTER-NARRATIVE (for comparison only):
"{reference_counter}"
"""
        
        prompt += """
Please evaluate the counter-narrative based on the following criteria:

1. Effectiveness (1-10): How effectively does the counter-narrative address and counter the specific harmful claims or misinformation in the hate speech?
2. Relevance (1-10): How directly does the counter-narrative address the core issues raised in the hate speech?
3. Persuasiveness (1-10): How compelling and convincing is the counter-narrative in changing minds?
4. Overall Score (1-10): Your overall assessment of the counter-narrative's quality.

For each criterion, provide a numerical score (1-10) and a brief explanation. Then provide constructive feedback on how the counter-narrative could be improved.

Format your response as JSON with the following structure:
{
  "effectiveness": {
    "score": <1-10>,
    "explanation": "<brief explanation>"
  },
  "relevance": {
    "score": <1-10>,
    "explanation": "<brief explanation>"
  }, 
  "persuasiveness": {
    "score": <1-10>,
    "explanation": "<brief explanation>"
  },
  "overall_score": <1-10>,
  "feedback": "<constructive feedback for improvement>"
}
"""
        return prompt
    
    def _parse_llm_evaluation(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse the LLM's evaluation response into a structured format.
        
        Args:
            llm_response: The raw response from the LLM
            
        Returns:
            Dictionary with parsed evaluation scores and feedback
        """
        # Try to extract JSON from the response
        try:
            # Find JSON in the response (it may be embedded in other text)
            import re
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                evaluation = json.loads(json_str)
                
                # Extract main scores for easier access
                result = {
                    "effectiveness": evaluation.get("effectiveness", {}).get("score", 0),
                    "relevance": evaluation.get("relevance", {}).get("score", 0),
                    "persuasiveness": evaluation.get("persuasiveness", {}).get("score", 0),
                    "overall_score": evaluation.get("overall_score", 0),
                    "feedback": evaluation.get("feedback", "No feedback provided"),
                    "detailed_evaluation": evaluation  # Include the full evaluation
                }
                return result
            
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Error parsing LLM response: {e}")
        
        # Fallback: try to extract scores using regex
        try:
            effectiveness = float(re.search(r'Effectiveness.*?(\d+(?:\.\d+)?)', llm_response).group(1))
            relevance = float(re.search(r'Relevance.*?(\d+(?:\.\d+)?)', llm_response).group(1))
            persuasiveness = float(re.search(r'Persuasiveness.*?(\d+(?:\.\d+)?)', llm_response).group(1))
            overall = float(re.search(r'Overall Score.*?(\d+(?:\.\d+)?)', llm_response).group(1))
            
            feedback_match = re.search(r'Feedback:(.+?)(?=\n\n|\Z)', llm_response, re.DOTALL)
            feedback = feedback_match.group(1).strip() if feedback_match else "No feedback extracted"
            
            return {
                "effectiveness": effectiveness,
                "relevance": relevance,
                "persuasiveness": persuasiveness,
                "overall_score": overall,
                "feedback": feedback,
                "raw_response": llm_response  # Include raw response for debugging
            }
            
        except (AttributeError, ValueError) as e:
            print(f"Error extracting scores from LLM response: {e}")
        
        # If all parsing attempts fail, return default values
        return {
            "effectiveness": 0,
            "relevance": 0,
            "persuasiveness": 0,
            "overall_score": 0,
            "feedback": "Failed to parse evaluation",
            "raw_response": llm_response
        }
    
    def evaluate_generation(self, 
                           hate_texts: List[str],
                           generated_counters: List[str],
                           reference_counters: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of generated counter-narratives.
        
        Args:
            hate_texts: List of hate speech texts
            generated_counters: List of generated counter-narrative texts
            reference_counters: List of reference counter-narratives (optional)
            
        Returns:
            Dictionary with all evaluation metrics
        """
        results = {}
        
        # LLM-based evaluation
        print("Running LLM-based evaluation...")
        llm_evaluations = self.evaluate_with_llm(
            hate_texts=hate_texts,
            counter_texts=generated_counters,
            reference_counters=reference_counters
        )
        
        # Calculate average scores across all examples
        avg_effectiveness = sum(eval.get("effectiveness", 0) for eval in llm_evaluations) / len(llm_evaluations) if llm_evaluations else 0
        avg_relevance = sum(eval.get("relevance", 0) for eval in llm_evaluations) / len(llm_evaluations) if llm_evaluations else 0
        avg_persuasiveness = sum(eval.get("persuasiveness", 0) for eval in llm_evaluations) / len(llm_evaluations) if llm_evaluations else 0
        avg_overall = sum(eval.get("overall_score", 0) for eval in llm_evaluations) / len(llm_evaluations) if llm_evaluations else 0
        
        results['llm_evaluation'] = {
            'evaluations': llm_evaluations,
            'avg_effectiveness': avg_effectiveness,
            'avg_relevance': avg_relevance,
            'avg_persuasiveness': avg_persuasiveness,
            'avg_overall_score': avg_overall
        }
        
        # Get toxicity scores if API key is available
        if self.perspective_api_key:
            print("Running toxicity evaluation...")
            toxicity_scores = self.get_toxicity_scores(generated_counters)
            
            # Calculate average toxicity
            avg_toxicity = {}
            if toxicity_scores and 'error' not in toxicity_scores[0]:
                for key in toxicity_scores[0].keys():
                    avg_toxicity[key] = sum(score.get(key, 0) for score in toxicity_scores) / len(toxicity_scores)
            
            results['toxicity'] = {
                'scores': toxicity_scores,
                'avg_scores': avg_toxicity
            }
        
        return results

# Example usage
def evaluate_counterspeech_examples():
    """Example usage of the evaluation metrics"""
    # Initialize evaluation metrics
    metrics = EvaluationMetrics(
        perspective_api_key="AIzaSyA5PI6IGKRV85Z-gDvaidMfFheAHfOTnj4",
        together_api_key="cecc55070f02c161af17c05085115bc012a16754ea34d890ca883df38cba108a"
    )
    
    # Example data
    hate_texts = [
        "Immigrants are stealing our jobs and ruining our economy!",
        "Women are too emotional to be leaders in business."
    ]
    
    generated_counters = [
        "Research consistently shows immigrants create jobs and contribute significantly to economic growth through their work, spending, and taxes.",
        "Studies show diverse leadership teams with women executives consistently outperform male-only teams, delivering higher profits and innovation."
    ]
    
    reference_counters = [
        "Immigration actually strengthens our economy. Immigrants start businesses at higher rates than natives and fill crucial jobs in growing industries.",
        "Data from hundreds of companies proves women leaders deliver better financial results, with higher emotional intelligence leading to better team management."
    ]
    
    # Run evaluation
    results = metrics.evaluate_generation(
        hate_texts=hate_texts,
        generated_counters=generated_counters,
        reference_counters=reference_counters
    )
    
    # Print results
    print("Evaluation Results:")
    print(f"LLM Evaluation - Average Overall Score: {results['llm_evaluation']['avg_overall_score']:.2f}")
    print(f"LLM Evaluation - Average Effectiveness: {results['llm_evaluation']['avg_effectiveness']:.2f}")
    print(f"LLM Evaluation - Average Relevance: {results['llm_evaluation']['avg_relevance']:.2f}")
    print(f"LLM Evaluation - Average Persuasiveness: {results['llm_evaluation']['avg_persuasiveness']:.2f}")
    
    if 'toxicity' in results:
        print(f"Average Toxicity: {results['toxicity']['avg_scores'].get('toxicity', 'N/A'):.4f}")
    
    return results

if __name__ == "__main__":
    evaluate_counterspeech_examples()