import time
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

class GoogleAPIPlayer:
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
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

class HateAssessmentSystem:
    """
    System for assessing and mitigating potential hate speech in user queries.
    """
    
    # Define threshold constants
    HATE_THRESHOLD_LOW = 5.0    # Below this is considered non-hateful
    HATE_THRESHOLD_HIGH = 7.0   # Above this requires stronger mitigation
    
    # Define score weights
    SIMILARITY_WEIGHT = 0.45    # Weight for similarity score
    HARMFULNESS_WEIGHT = 0.55   # Weight for harmfulness score
    
    def __init__(self, retriever, api_key: str):
        """
        Initialize the system with retriever and LLM components.
        
        Args:
            retriever: An instance of RAGRetriever for knowledge base access
            api_key: Google AI API key for Gemini
        """
        self.retriever = retriever
        self.llm = GoogleAPIPlayer(api_key=api_key)
        
    def process_query(self, user_query: str, search_method: str = "hybrid", 
                      num_results: int = 5) -> Dict[str, Any]:
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
        retrieved_content = self._retrieve_content(user_query, search_method, num_results)
        
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
    
    def _retrieve_content(self, query: str, search_method: str, limit: int) -> List[Dict]:
        """
        Retrieve content from the knowledge base using the specified search method.
        Only passes counter-speech content to the next phase while retaining metrics
        from the hate paragraphs for scoring.
        
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
            raw_results = self.retriever.semantic_search(query, limit_per_layer=limit)
        else:  # hybrid
            # Get results from both methods
            syntax_results = self.retriever.syntax_search_v2(query, limit_per_layer=limit)
            semantic_results = self.retriever.semantic_search(query, limit_per_layer=limit)
            
            # Debug print to check what's in the semantic results
            print(f"Debug - Semantic search returned {len(semantic_results)} results")
            if semantic_results:
                first_result = semantic_results[0]
                print(f"Debug - First semantic result keys: {first_result.keys()}")
                if 'similarity_score' in first_result:
                    print(f"Debug - First result similarity score: {first_result['similarity_score']}")
            
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
        print(f"Debug - Combined raw results: {len(raw_results)} items")
        if raw_results:
            first_raw = raw_results[0]
            print(f"Debug - First raw result keys: {first_raw.keys()}")
            if 'similarity_score' in first_raw:
                print(f"Debug - Raw similarity score: {first_raw['similarity_score']}")
        
        # Transform results to only include counter content while preserving metrics
        processed_results = []
        for item in raw_results:
            # Skip items without counter content
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
                
                # Keep relevance score from syntax search as backup
                #"relevance_score": item.get("relevance_score", 0),
                
                # For counter paragraphs, look at the correct fields based on your KB structure
                "directness": item.get("directness", 0),
                "evidence_quality": item.get("evidence_quality", 0),
                "persuasiveness": item.get("persuasiveness", 0)
            }
            
            processed_results.append(processed_item)
        
        # Debug processed results
        print(f"Debug - Processed results: {len(processed_results)} items")
        if processed_results:
            first_proc = processed_results[0]
            print(f"Debug - First processed keys: {first_proc.keys()}")
            print(f"Debug - Processed similarity_score: {first_proc['similarity_score']}")
            #print(f"Debug - Processed relevance_score: {first_proc['relevance_score']}")
            
        return processed_results
    
    def _calculate_hate_score(self, query: str, retrieved_content: List[Dict]) -> Tuple[float, Dict]:
        """
        Calculate a composite hate score based on retrieved content.
        Works with counter-content that still contains metrics from original hate paragraphs.
        
        Args:
            query: User query
            retrieved_content: Retrieved counter content with hate paragraph metrics
            
        Returns:
            Tuple containing:
            - composite hate score (0-10 scale)
            - assessment details dictionary
        """
        if not retrieved_content:
            return 0.0, {"message": "No relevant content found", "matches": []}
        
        # Extract relevant scores from each retrieved item
        item_scores = []
        for i, item in enumerate(retrieved_content):
            # Debug - print each item's relevant metrics
            print(f"Debug - Item {i} metrics:")
            print(f"  similarity_score: {item.get('similarity_score')}")
            #print(f"  relevance_score: {item.get('relevance_score')}")
            print(f"  quality_score: {item.get('quality_score')}")
            
            # Get harmfulness score directly from quality_score which contains the actual harmfulness
            harmfulness = float(item.get("quality_score", 0))
            
            # Get similarity or relevance score (default to 0 if not available)
            similarity = 0.0
            
            # First check for similarity_score from semantic search
            if "similarity_score" in item and item["similarity_score"] is not None and item["similarity_score"] != 0:
                similarity = float(item["similarity_score"])
                print(f"  Using similarity_score: {similarity}")
            # Then check for relevance_score from syntax search
            # elif "relevance_score" in item and item["relevance_score"] is not None and item["relevance_score"] != 0:
            #     # Normalize relevance score to 0-1 range
            #     raw_relevance = float(item["relevance_score"])
            #     # If relevance score is already between 0-1, use it directly
            #     if raw_relevance <= 1.0:
            #         similarity = raw_relevance
            #     else:
            #         # Otherwise normalize it down to 0-1 range
            #         similarity = min(raw_relevance / 10.0, 1.0)
            #     print(f"  Using normalized relevance_score: {similarity}")
            else:
                # If we don't have a proper similarity/relevance score, use a default value
                # This ensures items with high harmfulness but low similarity still contribute
                similarity = 0.5
                print(f"  Using default similarity: {similarity}")
            
            # Calculate weighted item score
            weighted_score = (self.SIMILARITY_WEIGHT * similarity + 
                             self.HARMFULNESS_WEIGHT * (harmfulness / 10.0)) * 10.0
            
            print(f"  Final weighted score: {weighted_score}")
            
            item_scores.append({
                "id": item.get("hate_paragraph_id", "unknown"),
                "counter_id": item.get("counter_id", "unknown"),
                "similarity": round(similarity, 2),
                "harmfulness": round(harmfulness, 2),
                "weighted_score": round(weighted_score, 2)
            })
        
        # Sort by weighted score in descending order
        item_scores.sort(key=lambda x: x["weighted_score"], reverse=True)
        
        # Take the top 3 scores (or fewer if there aren't 3)
        top_scores = item_scores[:min(3, len(item_scores))]
        
        # Calculate the final score as the average of the top scores
        final_score = sum(item["weighted_score"] for item in top_scores) / len(top_scores)
        
        # Cap at 10.0
        final_score = min(final_score, 10.0)
        
        assessment_details = {
            "message": f"Evaluated {len(retrieved_content)} relevant counter-speech items",
            "top_matches": top_scores,
            "calculation_method": "Weighted average of top 3 matches",
            "weights": {
                "similarity": self.SIMILARITY_WEIGHT,
                "harmfulness": self.HARMFULNESS_WEIGHT
            }
        }
        
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

Provide a direct, factual answer without any special cautioning or mitigation.
"""
    
    def _create_mild_mitigation_prompt(self, query: str, retrieved_content: List[Dict]) -> str:
        """Create a prompt with mild mitigation for potentially hateful content."""
        # Get counter-speech examples (now the content itself is the counter speech)
        counter_examples = []
        for item in retrieved_content:
            if item["content"]:  # Only add if there's actual content
                counter_examples.append(item["content"])
        
        prompt = f"""Please provide a balanced, educational response to this user query:

User query: {query}

The query appears to touch on potentially sensitive topics. In your response:
1. Provide factual context and nuance
2. Gently correct any misconceptions if present
3. Maintain a balanced, educational tone
4. Present multiple perspectives where appropriate

"""
        
        # Add counter-speech examples if available
        if counter_examples:
            prompt += "You may consider these reference perspectives in your response:\n\n"
            for i, example in enumerate(counter_examples[:2]):
                prompt += f"Perspective {i+1}: {example}\n\n"
        
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
        
        prompt = f"""Please provide a firm, educational counter-response to this potentially problematic query:

User query: {query}

This query appears to contain or seek content that could promote harmful narratives. In your response:
1. Directly address and correct misconceptions or harmful framing
2. Provide accurate, factual information to counter potential misinformation
3. Maintain a respectful but firm educational tone
4. Offer evidence-based context that promotes understanding
5. Highlight the human and social impact of harmful narratives

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
    assessment_system = HateAssessmentSystem(retriever, api_key="YOUR_API_KEY")
    
    # Process a sample query
    result = assessment_system.process_query(
        user_query="i hate immigrants",
        search_method="hybrid",
        num_results=5
    )
    
    # Print results
    print(f"Query: {result['query']}")
    print(f"Hate Score: {result['hate_score']}")
    print(f"Mitigation Level: {result['mitigation_level']}")
    print("\nResponse:")
    print(result['response'])