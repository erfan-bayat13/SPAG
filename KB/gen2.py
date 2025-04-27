import math
import time
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from together import Together
import os

class GoogleAPIPlayer:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=1.2,
                )
            )
            # Add delay to respect rate limits
            time.sleep(1)
            return response.text.strip()
        except Exception as e:
            print(f"API call failed: {str(e)}")
            return ""

class TogetherAIPlayer:
    """A wrapper class for using the Together AI API"""
    
    def __init__(self, model_name, api_key=None):
        """
        Initialize the Together AI player with a model name
        
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
    
    def generate(self, prompt, max_tokens=512, temperature=1.0, do_sample=True):
        """
        Generate text using the Together AI API
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
        
        Returns:
            The generated text response
        """
        # Make API call with retry logic for rate limits
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature if do_sample else 0.7,
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

class HateAssessmentSystem:
    """
    System for assessing and mitigating potential hate speech in user queries.
    """
    
    # Define threshold constants
    HATE_THRESHOLD_LOW = 4.0    # Below this is considered non-hateful
    HATE_THRESHOLD_HIGH = 7.0   # Above this requires stronger mitigation
    
    # Define score weights
    SIMILARITY_WEIGHT = 0.5    # Weight for similarity score
    HARMFULNESS_WEIGHT = 0.5   # Weight for harmfulness score
    
    def __init__(self, retriever, api_key: str):
        """
        Initialize the system with retriever and LLM components.
        
        Args:
            retriever: An instance of RAGRetriever for knowledge base access
            api_key: Google AI API key for Gemini
        """
        self.retriever = retriever
        #self.llm = GoogleAPIPlayer(api_key=api_key)
        self.llm = TogetherAIPlayer(model_name="google/gemma-2-9b-it", api_key=api_key)
        
    def process_query(self, user_query: str, search_method: str = "semantic", 
                      num_results: int = 5, order_by: str = "similarity") -> Dict[str, Any]:
        """
        Process a user query through the entire assessment and response pipeline.
        
        Args:
            user_query: The user's question or query text
            search_method: Retrieval method ("syntax", "semantic", or "hybrid")
            num_results: Number of results to retrieve from KB
            
        Returns:
            Dict containing response, assessment details, and retrieved content
        """
        # Step 1: Retrieve relevant content from knowledge base
        retrieved_content = self._retrieve_content(user_query, search_method, num_results,order_by=order_by)
        
        # Step 2: Calculate hate score based on retrieved content
        hate_score, assessment_details = self._calculate_hate_score(user_query, retrieved_content)
        
        # Step 3: Determine response strategy based on hate score
        mitigation_level = self._determine_mitigation_level(hate_score)
        
        # Step 4: Generate appropriate response using selected strategy
        response = self._generate_response(user_query, retrieved_content, 
                                          mitigation_level, hate_score)
        
        # Step 5: Return complete result with metadata
        result = {
            "query": user_query,
            "response": response,
            "hate_score": round(hate_score, 2),
            "mitigation_level": mitigation_level,
            "assessment_details": assessment_details,
            "retrieved_content": retrieved_content,
        }
        
        return result
    
    def _retrieve_content(self, query: str, search_method: str, limit: int, order_by: str = "similarity") -> List[Dict]:
        """
        Retrieve content from the knowledge base using the specified search method.
        For synthetic hate content with no counter-speech, finds and assigns a random
        counter-speech example from the same topic.
        
        Args:
            query: User query text
            search_method: Type of search to perform
            limit: Maximum number of results
            
        Returns:
            List of counter-speech content items with retained metrics from hate paragraphs
        """
        # First retrieve the raw results
        if search_method == "syntax":
            raw_results = self.retriever.syntax_search_v2(query, limit_per_layer=limit)
        elif search_method == "semantic":
            raw_results = self.retriever.semantic_search(query, limit_per_layer=limit,order_by=order_by)
        else:  # hybrid
            # Get results from both methods
            syntax_results = self.retriever.syntax_search_v2(query, limit_per_layer=limit)
            semantic_results = self.retriever.semantic_search(query, limit_per_layer=limit,order_by=order_by)
            
            # Debug print to check what's in the semantic results
            # print(f"Debug - Semantic search returned {len(semantic_results)} results")
            # if semantic_results:
            #     first_result = semantic_results[0]
            #     print(f"Debug - First semantic result keys: {first_result.keys()}")
            #     if 'similarity_score' in first_result:
            #         print(f"Debug - First result similarity score: {first_result['similarity_score']}")
            
            # Combine and deduplicate results
            all_ids = set()
            raw_results = []
            
            # First add semantic results which should have similarity scores
            for result in semantic_results:
                if result["id"] not in all_ids:
                    all_ids.add(result["id"])
                    raw_results.append(result)
            
            # Then add syntax results that aren't duplicates
            for result in syntax_results:
                if result["id"] not in all_ids:
                    all_ids.add(result["id"])
                    raw_results.append(result)
                    
                if len(raw_results) >= limit*2:
                    break
        
        # Debug print to check combined results
        #print(f"Debug - Combined raw results: {len(raw_results)} items")
        
        # Identify synthetic hate content that lacks counter-speech
        synthetic_items_without_counter = []
        for item in raw_results:
            # Check if it's synthetic and missing counter content
            is_synthetic = item.get("layer") == 2 or item.get("is_synthetic") == True
            has_counter = item.get("counter_content") or item.get("counter_id")
            
            if is_synthetic and not has_counter:
                synthetic_items_without_counter.append(item)
        
        # If we have synthetic items without counter content, find matching counters
        if synthetic_items_without_counter:
            #print(f"Debug - Found {len(synthetic_items_without_counter)} synthetic items without counter content")
            
            # Get topics from synthetic items
            topics = [item.get("topic") for item in synthetic_items_without_counter if item.get("topic")]
            unique_topics = list(set([t for t in topics if t]))
            
            if unique_topics:
                #print(f"Debug - Found these topics needing counters: {unique_topics}")
                
                # Fetch random counter content for each topic
                topic_to_counter_map = {}
                
                for topic in unique_topics:
                    counter_query = """
                    MATCH (cp:CounterParagraph)-[:TALKS_ABOUT]->(t:Topic {name: $topic})
                    RETURN cp.id AS id, cp.content AS content, 
                        cp.directness AS directness, 
                        cp.evidence_quality AS evidence_quality,
                        cp.persuasiveness AS persuasiveness,
                        cp.quality_score AS quality_score
                    LIMIT 5
                    """
                    try:
                        counters = list(self.retriever.memgraph.execute_and_fetch(counter_query, {"topic": topic}))
                        if counters:
                            # Select a random counter from the results
                            import random
                            random_counter = random.choice(counters)
                            topic_to_counter_map[topic] = random_counter
                            #print(f"Debug - Found counter for topic '{topic}'")
                    except Exception as e:
                        print(f"Error fetching counters for topic {topic}: {e}")
                
                # Assign counters to synthetic items
                for item in synthetic_items_without_counter:
                    topic = item.get("topic")
                    if topic and topic in topic_to_counter_map:
                        counter = topic_to_counter_map[topic]
                        item["counter_id"] = counter.get("id")
                        item["counter_content"] = counter.get("content")
                        item["directness"] = counter.get("directness", 0)
                        item["evidence_quality"] = counter.get("evidence_quality", 0)
                        item["persuasiveness"] = counter.get("persuasiveness", 0)
                        #print(f"Debug - Assigned counter to synthetic item with id {item.get('id')}")
        
        # Transform results to only include counter content while preserving metrics
        processed_results = []
        for item in raw_results:
            # Skip items that still don't have counter content after our fix
            if not item.get("counter_content") and not item.get("counter_id"):
                continue
                
            # Create a new item focused on counter content
            processed_item = {
                "counter_id": item.get("counter_id"),
                "content": item.get("counter_content", ""),  # Now using counter content as primary content
                "topic": item.get("topic"),
                
                # Keep original hate paragraph ID for reference
                "hate_paragraph_id": item.get("id"),
                
                # In your KB, quality_score already contains the harmfulness rating
                "quality_score": item.get("quality_score", 0),
                
                # Explicitly preserve similarity score from semantic search
                "similarity_score": item.get("similarity_score", 0),
                
                # For counter paragraphs, look at the correct fields based on your KB structure
                "directness": item.get("directness", 0),
                "evidence_quality": item.get("evidence_quality", 0),
                "persuasiveness": item.get("persuasiveness", 0)
            }
            
            processed_results.append(processed_item)
        
        # Debug processed results
        #print(f"Debug - Processed results: {len(processed_results)} items")
        if processed_results:
            first_proc = processed_results[0]
            #print(f"Debug - First processed keys: {first_proc.keys()}")
            #print(f"Debug - Processed similarity_score: {first_proc['similarity_score']}")
            
        return processed_results
    
    def _calculate_hate_score(self, query: str, retrieved_content: List[Dict]) -> Tuple[float, Dict]:
        """
        Calculate a properly normalized hate score based on retrieved content,
        with adjusted calibration for real-world score distributions.
        
        Args:
            query: User query
            retrieved_content: Retrieved counter content with hate paragraph metrics
            
        Returns:
            Tuple containing:
            - calibrated hate score (0-10 scale)
            - assessment details dictionary
        """
        if not retrieved_content:
            return 0.0, {"message": "No relevant content found", "matches": []}
        
        # Extract relevant scores from each retrieved item
        item_scores = []
        
        for i, item in enumerate(retrieved_content):
            # Get harmfulness score from quality_score
            harmfulness = float(item.get("quality_score", 0))
            
            # Get similarity score
            similarity = float(item.get("similarity_score", 0.5))
            
            # Raw weighted score calculation
            raw_weighted_score = (self.SIMILARITY_WEIGHT * similarity + 
                                self.HARMFULNESS_WEIGHT * (harmfulness / 10.0)) * 10.0
            
            item_scores.append({
                "id": item.get("hate_paragraph_id", "unknown"),
                "counter_id": item.get("counter_id", "unknown"),
                "similarity": round(similarity, 2),
                "harmfulness": round(harmfulness, 2),
                "raw_weighted_score": raw_weighted_score
            })
        
        # Sort by weighted score in descending order
        item_scores.sort(key=lambda x: x["raw_weighted_score"], reverse=True)
        
        # Take the top 3 scores (or fewer if there aren't 3)
        top_scores = item_scores[:min(3, len(item_scores))]
        
        # Calculate the average raw score
        raw_final_score = sum(item["raw_weighted_score"] for item in top_scores) / len(top_scores)
        
        # ADJUSTED NORMALIZATION APPROACH
        # Based on empirical observations that scores cluster at the low end
        
        # Theoretical range
        KB_MIN_SCORE = 5.5
        KB_MAX_SCORE = 10.0
        
        # Observed real-world range (adjust these based on your actual data)
        # Setting a narrower observed range to better distribute scores
        OBSERVED_MIN = 5.5
        OBSERVED_MAX = 8.0  # Adjust this based on your actual maximum observed scores
        
        # Apply square root transformation to spread out lower values
        transformed_score = (raw_final_score - OBSERVED_MIN) / (OBSERVED_MAX - OBSERVED_MIN)
        transformed_score = max(0, min(transformed_score, 1))  # Clamp to 0-1
        
        # Apply power transformation to spread values across range
        # Using square root to give more weight to lower values
        calibrated_score = math.sqrt(transformed_score) * 10
        
        # Ensure we have a minimum non-zero score for any relevant match
        if len(retrieved_content) > 0 and calibrated_score < 2:
            calibrated_score = max(calibrated_score, 2.0)
        
        # Update the top scores with calibrated values
        for item in top_scores:
            transformed = (item["raw_weighted_score"] - OBSERVED_MIN) / (OBSERVED_MAX - OBSERVED_MIN)
            transformed = max(0, min(transformed, 1))
            item["calibrated_score"] = math.sqrt(transformed) * 10
        
        assessment_details = {
            "message": f"Evaluated {len(retrieved_content)} relevant counter-speech items",
            "theoretical_range": {"min": KB_MIN_SCORE, "max": KB_MAX_SCORE},
            "observed_range": {"min": OBSERVED_MIN, "max": OBSERVED_MAX},
            "top_matches": top_scores,
            "raw_score": raw_final_score,
            "calibrated_score": calibrated_score,
            "calculation_method": "Square root transformation with calibrated range",
            "weights": {
                "similarity": self.SIMILARITY_WEIGHT,
                "harmfulness": self.HARMFULNESS_WEIGHT
            }
        }
        final_score = calibrated_score
        #print(f"Debug - Final hate score: {final_score}")
        
        return final_score, assessment_details
    
    def _determine_mitigation_level(self, hate_score: float) -> str:
        """
        Determine the appropriate mitigation level based on hate score.
        
        Args:
            hate_score: Calculated hate score (0-10)
            
        Returns:
            String indicating mitigation level: "none", "mild", or "strong"
        """
        if hate_score < self.HATE_THRESHOLD_LOW:
            return "none"
        elif hate_score < self.HATE_THRESHOLD_HIGH:
            return "mild"
        else:
            return "strong"
    
    def _generate_response(self, query: str, retrieved_content: List[Dict], 
                          mitigation_level: str, hate_score: float) -> str:
        """
        Generate an appropriate response based on the mitigation level.
        
        Args:
            query: User query
            retrieved_content: Retrieved content
            mitigation_level: Determined mitigation level
            hate_score: Calculated hate score
            
        Returns:
            Generated response text
        """
        # Choose appropriate prompt template based on mitigation level
        if mitigation_level == "none":
            # No mitigation needed - normal response
            prompt = self._create_normal_prompt(query, retrieved_content)
        elif mitigation_level == "mild":
            # Mild mitigation needed
            prompt = self._create_mild_mitigation_prompt(query, retrieved_content)
        else:
            # Strong mitigation needed
            prompt = self._create_strong_mitigation_prompt(query, retrieved_content)
        
        # Generate the response using the LLM
        response = self.llm.generate(prompt, max_tokens=512)
        return response
    
    def _create_normal_prompt(self, query: str, retrieved_content: List[Dict]) -> str:
        """Create a normal prompt with no special mitigation."""
        return f"""Please provide a helpful, informative response to this user query:

User query: {query}

Our system has not seen any indication that this query is problematic.
YOU decide whether this query requires mitigation. If you believe the query contains harmful content or seeks harmful information, mitigate it.
If you determine the query is NOT problematic, simply answer it directly and factually without any special cautioning.
"""
    
    def _create_mild_mitigation_prompt(self, query: str, retrieved_content: List[Dict]) -> str:
        """Create a prompt with mild mitigation for potentially hateful content."""
        # Get counter-speech examples (now the content itself is the counter speech)
        counter_examples = []
        for item in retrieved_content:
            if item["content"]:  # Only add if there's actual content
                counter_examples.append(item["content"])
        
        prompt = f"""Please analyze and respond to this user query:

User query: {query}

Our system has flagged this query as potentially touching on sensitive topics (hate score: moderate).

YOU decide whether this query requires mitigation. If you believe the query contains harmful content or seeks harmful information, please provide an educational response that:
1. Gently corrects any misconceptions
2. Provides factual context and nuance
3. Maintains a balanced tone
4. Presents multiple perspectives where appropriate

However, if you determine the query is NOT problematic, simply answer it directly and factually without any special cautioning.

Trust your own assessment of whether the content requires mitigation or not.
"""
        
        # Add counter-speech examples if available
        if counter_examples:
            prompt += "You may consider these reference perspectives in your response:\n\n"
            for i, example in enumerate(counter_examples[:2]):
                prompt += f"Perspective {i+1}: {example}\n\n"
        #print(f"Debug - Mild mitigation prompt: {prompt}")
        
        return prompt
    
    def _create_strong_mitigation_prompt(self, query: str, retrieved_content: List[Dict]) -> str:
        """Create a prompt with strong mitigation for hateful content."""
        # Process counter examples with quality metrics
        counter_examples = []
        for item in retrieved_content:
            if not item["content"]:  # Skip empty content
                continue
                
            # Calculate a quality score using available metrics
            quality_score = (
                float(item.get("directness", 0)) + 
                float(item.get("evidence_quality", 0)) + 
                float(item.get("persuasiveness", 0))
            )
            
            # If no specific counter quality metrics, use a default value
            if quality_score == 0:
                quality_score = 5.0
                
            counter_examples.append({
                "content": item["content"],
                "quality_score": quality_score
            })
        
        # Sort by quality metrics
        if counter_examples:
            counter_examples.sort(key=lambda x: x["quality_score"], reverse=True)
            top_counters = [ex["content"] for ex in counter_examples[:3]]
        else:
            top_counters = []
        
        prompt = f"""Please analyze and respond to this user query:

User query: {query}

Our system has flagged this query as potentially containing or seeking content that could promote harmful narratives (hate score: high).

YOU decide whether this query requires mitigation. If you believe the query contains harmful content or seeks harmful information, please provide an educational response that:
1. Directly addresses and corrects misconceptions or harmful framing
2. Provides accurate, factual information to counter potential misinformation
3. Maintains a respectful but firm educational tone
4. Offers evidence-based context that promotes understanding
5. Highlights the human and social impact of harmful narratives
6. Response must be 3-4 sentences (60-80 words) maximum

However, if you determine the query is NOT problematic despite our system's high hate score, simply answer it directly and factually without special cautioning.

Trust your own assessment of whether the content requires mitigation or not.(NO need to mention if you think the query is problematic or not)
"""
        
        # Add counter-speech examples if available
        if top_counters:
            prompt += "Here are effective counter-arguments that address this type of content:\n\n"
            for i, example in enumerate(top_counters):
                prompt += f"Counter-argument {i+1}: {example}\n\n"
        
        prompt += """
Your response should be educational rather than accusatory. Focus on providing accurate information
that counters harmful narratives while treating the user with respect.
"""
        
        return prompt

# Usage example
if __name__ == "__main__":
    from retrieve import RAGRetriever
    
    # Initialize components
    retriever = RAGRetriever()
    assessment_system = HateAssessmentSystem(retriever, api_key="API_KEY_HERE")
    
    # Process a sample query
    result = assessment_system.process_query(
        user_query="eligious minorities are a clear and present danger",
        search_method="semantic",
        num_results=5
    )  
    
    # Print results
    # print(f"Query: {result['query']}")
    # print(f"Hate Score: {result['hate_score']}")
    # print(f"Mitigation Level: {result['mitigation_level']}")
    # print("\nResponse:")
    # print(result['response'])