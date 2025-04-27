from gqlalchemy import Memgraph
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from typing import List, Dict, Any, Optional

class RAGRetriever:
    def __init__(self, memgraph_host: str = "127.0.0.1", memgraph_port: int = 7687):
        """
        Initialize the RAG retriever with connections to Memgraph.
        
        Args:
            memgraph_host: Host address for Memgraph
            memgraph_port: Port for Memgraph connection
        """
        # Connect to Memgraph
        self.memgraph = Memgraph(memgraph_host, memgraph_port)
        
        # Initialize the embedding model (we'll use it later)
        self.model = None
        
    def syntax_search_debate_content(self, query_text: str, limit: int = 5) -> List[Dict]:
        """
        Perform syntax-based search on debate-based hate content.
        
        Args:
            query_text: The search query text
            limit: Maximum number of results to return
            
        Returns:
            List of matching hate paragraph nodes with their details
        """
        # Process query text for better pattern matching
        # Convert to lowercase for case-insensitive matching
        query_lower = query_text.lower()
        
        # Split into keywords for more flexible matching
        keywords = [keyword.strip() for keyword in query_lower.split() if len(keyword.strip()) > 3]
        
        # Create a Cypher query using CONTAINS which is more widely supported
        query = """
        MATCH (hp:HateParagraph)
        WHERE (hp.is_synthetic IS NULL OR hp.is_synthetic = false)
        AND (
        """
        
        # Add WHERE conditions for each keyword with lowercase transformation
        conditions = []
        for keyword in keywords:
            conditions.append(f"toLower(hp.content) CONTAINS '{keyword}'")
        
        # If no valid keywords, use the original query text
        if not conditions:
            conditions.append(f"toLower(hp.content) CONTAINS '{query_lower}'")
        
        query += " OR ".join(conditions)
        query += """
        )
        OPTIONAL MATCH (hp)-[:TALKS_ABOUT]->(t:Topic)
        OPTIONAL MATCH (hp)-[:COUNTERED_WITH]->(cp:CounterParagraph)
        
        WITH hp, t, cp, 
             CASE WHEN toLower(hp.content) CONTAINS '{0}' THEN 5 ELSE 0 END +
             {1} AS relevance_score
        
        RETURN hp.id AS id, 
               hp.content AS content,
               t.name AS topic,
               hp.quality_score AS quality_score,
               cp.id AS counter_id,
               cp.content AS counter_content,
               relevance_score
        ORDER BY relevance_score DESC
        LIMIT $limit
        """.format(query_lower, 
                  ' + '.join([f"CASE WHEN toLower(hp.content) CONTAINS '{k}' THEN 1 ELSE 0 END" for k in keywords]))
        
        # Execute the query
        params = {"limit": limit}
        results = list(self.memgraph.execute_and_fetch(query, params))
        
        return results
    def syntax_search(self, query_text: str, limit_per_layer: int = 5) -> List[Dict]:
        """
        Perform a two-layer syntax-based search on hate content.
        Layer 1: Debate-based hate content (non-synthetic)
        Layer 2: Synthetic hate paragraphs
        
        Args:
            query_text: The search query text
            limit_per_layer: Maximum number of results to return per layer
            
        Returns:
            List of matching hate paragraph nodes with their details and layer information
        """
        # Process query text for better pattern matching
        query_lower = query_text.lower()
        
        # Split into keywords for more flexible matching
        keywords = [keyword.strip() for keyword in query_lower.split() if len(keyword.strip()) > 3]
        
        # Layer 1: Debate-based content (non-synthetic)
        layer1_query = """
        MATCH (hp:HateParagraph)
        WHERE (hp.is_synthetic IS NULL OR hp.is_synthetic = false)
        """
        
        # Layer 2: Synthetic content
        layer2_query = """
        MATCH (hp:HateParagraph)
        WHERE hp.is_synthetic = true
        """
        
        # Build conditions for both queries
        conditions = []
        for keyword in keywords:
            conditions.append(f"toLower(hp.content) CONTAINS '{keyword}'")
        
        # If no valid keywords, use the original query text
        if not conditions:
            conditions.append(f"toLower(hp.content) CONTAINS '{query_lower}'")
        
        condition_str = " AND (" + " OR ".join(conditions) + ")"
        
        # Add conditions and complete both queries
        common_suffix = """
        OPTIONAL MATCH (hp)-[:TALKS_ABOUT]->(t:Topic)
        OPTIONAL MATCH (hp)-[:COUNTERED_WITH]->(cp:CounterParagraph)
        
        WITH hp, t, cp, 
             CASE WHEN toLower(hp.content) CONTAINS '{0}' THEN 5 ELSE 0 END +
             {1} AS relevance_score
        
        RETURN hp.id AS id, 
               hp.content AS content,
               t.name AS topic,
               hp.quality_score AS quality_score,
               cp.id AS counter_id,
               cp.content AS counter_content,
               relevance_score,
               {2} AS layer
        ORDER BY relevance_score DESC
        LIMIT $limit
        """.format(
            query_lower, 
            ' + '.join([f"CASE WHEN toLower(hp.content) CONTAINS '{k}' THEN 1 ELSE 0 END" for k in keywords]),
            "{0}"  # Placeholder for layer number
        )
        
        # Complete both queries
        layer1_query += condition_str + common_suffix.format("1")
        layer2_query += condition_str + common_suffix.format("2")
        
        # Execute both queries
        params = {"limit": limit_per_layer}
        layer1_results = list(self.memgraph.execute_and_fetch(layer1_query, params))
        layer2_results = list(self.memgraph.execute_and_fetch(layer2_query, params))
        
        # Combine results, with layer 1 first
        all_results = layer1_results + layer2_results
        
        return all_results
    def syntax_search_v2(self, query_text: str, limit_per_layer: int = 5) -> List[Dict]:
        """
        Perform a two-layer syntax-based search on hate content.
        Handles both keywords and complete sentences using NLP techniques.
        
        Args:
            query_text: The search query text (keyword or sentence)
            limit_per_layer: Maximum number of results to return per layer
            
        Returns:
            List of matching hate paragraph nodes with their details and layer information
        """
        # Process query using NLP
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
        try:
            # Try to use NLTK's stopwords
            nltk_stopwords = set(stopwords.words('english'))
        except:
            # If NLTK data is not available, use a basic stopword list
            nltk_stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                        'to', 'of', 'in', 'for', 'with', 'on', 'at', 'by', 'this', 'that'}
        
        # Tokenize and clean the query
        query_lower = query_text.lower()
        tokens = word_tokenize(query_lower) if 'word_tokenize' in dir(nltk.tokenize) else query_lower.split()
        
        # Extract meaningful keywords (non-stopwords, longer than 3 chars)
        keywords = [word for word in tokens if len(word) > 3 and word not in nltk_stopwords]
        
        # Extract potential phrases (2-3 word combinations)
        phrases = []
        # Generate bigrams if there are enough words
        if len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                if tokens[i] not in nltk_stopwords or tokens[i+1] not in nltk_stopwords:
                    phrase = tokens[i] + ' ' + tokens[i+1]
                    if len(phrase) > 5:  # Only meaningful phrases
                        phrases.append(phrase)
        
        # Generate trigrams if there are enough words
        if len(tokens) >= 3:
            for i in range(len(tokens) - 2):
                if (tokens[i] not in nltk_stopwords or 
                    tokens[i+1] not in nltk_stopwords or 
                    tokens[i+2] not in nltk_stopwords):
                    phrase = tokens[i] + ' ' + tokens[i+1] + ' ' + tokens[i+2]
                    if len(phrase) > 8:  # Only meaningful phrases
                        phrases.append(phrase)
        
        # Layer 1: Debate-based content (non-synthetic)
        layer1_query = """
        MATCH (hp:HateParagraph)
        WHERE (hp.is_synthetic IS NULL OR hp.is_synthetic = false)
        """
        
        # Layer 2: Synthetic content
        layer2_query = """
        MATCH (hp:HateParagraph)
        WHERE hp.is_synthetic = true
        """
        
        # Build keyword conditions
        keyword_conditions = []
        for keyword in keywords:
            keyword_conditions.append(f"toLower(hp.content) CONTAINS '{keyword}'")
        
        # Build phrase conditions with higher weight
        phrase_conditions = []
        for phrase in phrases:
            phrase_conditions.append(f"toLower(hp.content) CONTAINS '{phrase}'")
        
        # If no keywords or phrases, use the entire query
        if not keyword_conditions and not phrase_conditions:
            keyword_conditions.append(f"toLower(hp.content) CONTAINS '{query_lower}'")
        
        # Combine all conditions
        all_conditions = keyword_conditions + phrase_conditions
        condition_str = " AND (" + " OR ".join(all_conditions) + ")"
        
        # Define relevance scoring (phrases get higher scores)
        relevance_parts = []
        relevance_parts.append(f"CASE WHEN toLower(hp.content) CONTAINS '{query_lower}' THEN 10 ELSE 0 END")
        
        for phrase in phrases:
            relevance_parts.append(f"CASE WHEN toLower(hp.content) CONTAINS '{phrase}' THEN 3 ELSE 0 END")
        
        for keyword in keywords:
            relevance_parts.append(f"CASE WHEN toLower(hp.content) CONTAINS '{keyword}' THEN 1 ELSE 0 END")
        
        relevance_calc = " + ".join(relevance_parts)
        
        # Complete both queries with common suffix
        common_suffix = """
        OPTIONAL MATCH (hp)-[:TALKS_ABOUT]->(t:Topic)
        OPTIONAL MATCH (hp)-[:COUNTERED_WITH]->(cp:CounterParagraph)
        
        WITH DISTINCT hp, t, cp, 
             {0} AS relevance_score
        
        RETURN hp.id AS id, 
               hp.content AS content,
               t.name AS topic,
               hp.quality_score AS quality_score,
               cp.id AS counter_id,
               cp.content AS counter_content,
               relevance_score,
               {1} AS layer
        ORDER BY relevance_score DESC
        LIMIT $limit
        """.format(relevance_calc, "{0}")  # Placeholder for layer number
        
        # Complete both queries
        layer1_query += condition_str + common_suffix.format("1")
        layer2_query += condition_str + common_suffix.format("2")
        
        # Execute both queries
        params = {"limit": limit_per_layer}
        layer1_results = list(self.memgraph.execute_and_fetch(layer1_query, params))
        layer2_results = list(self.memgraph.execute_and_fetch(layer2_query, params))
        
        # Combine results, with layer 1 first
        combined_results = layer1_results + layer2_results
        
        # Additional Python-side deduplication (as a safeguard)
        seen_ids = set()
        unique_results = []
        
        for result in combined_results:
            paragraph_id = result['id']
            if paragraph_id not in seen_ids:
                seen_ids.add(paragraph_id)
                unique_results.append(result)
        
        return unique_results
        
    def semantic_search(self, query_text: str, limit_per_layer: int = 5, similarity_threshold: float = 0.1, order_by: str = "similarity") -> List[Dict]:
        """
        Perform semantic search on hate content using vector search capabilities.
        Searches both debate-based content (layer 1) and synthetic content (layer 2).
        
        Args:
            query_text: The search query text
            limit_per_layer: Maximum number of results to return per layer
            similarity_threshold: Minimum similarity score (0-1) for results
            
        Returns:
            List of matching hate content with similarity scores, ordered by relevance
        """
        # Lazy-load the embedding model if it's not already loaded
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Ensure vector indices exist (should be done during initialization/setup)
        self._ensure_vector_indices()
        
        # Generate embedding for the query text
        query_embedding = self.model.encode(query_text).tolist()
        embeddings_str = ",".join([str(x) for x in query_embedding])
        
        order_by_clause = "similarity_score" if order_by == "similarity" else "hp.quality_score"
        # Layer 1: Search in debate-based hate content using vector search
        # We request more results from vector search to ensure we have enough after filtering
        layer1_query = f"""
        CALL vector_search.search('hate_content_embedding_index', $search_limit, [{embeddings_str}]) 
        YIELD node, similarity
        WITH node AS hc, similarity
        WHERE similarity >= $threshold
        
        // Get parent paragraph and ensure it's not synthetic
        MATCH (hp:HateParagraph)-[:CONTAINS]->(hc)
        WHERE (hp.is_synthetic IS NULL OR hp.is_synthetic = false)
        
        // Get related nodes
        OPTIONAL MATCH (hp)-[:TALKS_ABOUT]->(t:Topic)
        OPTIONAL MATCH (hp)-[:COUNTERED_WITH]->(cp:CounterParagraph)
        
        // Aggregate results at paragraph level (in case multiple sentences match)
        // Use DISTINCT to ensure each paragraph appears only once
        WITH DISTINCT hp, t, cp, MAX(similarity) AS max_similarity
        
        RETURN hp.id AS id, 
               hp.content AS content,
               t.name AS topic,
               hp.quality_score AS quality_score,
               cp.id AS counter_id, 
               cp.content AS counter_content,
               max_similarity AS similarity_score,
               1 AS layer
        ORDER BY {order_by_clause} DESC
        LIMIT $result_limit
        """
        
        # Layer 2: Search in synthetic hate content using vector search
        layer2_query = f"""
        CALL vector_search.search('hate_content_embedding_index', $search_limit, [{embeddings_str}]) 
        YIELD node, similarity
        WITH node AS hc, similarity
        WHERE similarity >= $threshold
        
        // Get parent paragraph and ensure it's synthetic
        MATCH (hp:HateParagraph)-[:CONTAINS]->(hc)
        WHERE hp.is_synthetic = true
        
        // Get related nodes
        OPTIONAL MATCH (hp)-[:TALKS_ABOUT]->(t:Topic)
        OPTIONAL MATCH (hp)-[:COUNTERED_WITH]->(cp:CounterParagraph)
        
        // Aggregate results at paragraph level (in case multiple sentences match)
        // Use DISTINCT to ensure each paragraph appears only once
        WITH DISTINCT hp, t, cp, MAX(similarity) AS max_similarity
        
        RETURN hp.id AS id, 
               hp.content AS content,
               t.name AS topic,
               hp.quality_score AS quality_score,
               cp.id AS counter_id, 
               cp.content AS counter_content,
               max_similarity AS similarity_score,
               2 AS layer
        ORDER BY {order_by_clause} DESC
        LIMIT $result_limit
        """
        
        # We request more from the vector search to ensure we have enough after filtering
        # A multiplier of 3 is a reasonable starting point
        search_limit = limit_per_layer * 3
        
        # Execute queries
        params = {
            "threshold": similarity_threshold,
            "search_limit": search_limit,
            "result_limit": limit_per_layer
        }
        
        try:
            layer1_results = list(self.memgraph.execute_and_fetch(layer1_query, params))
            layer2_results = list(self.memgraph.execute_and_fetch(layer2_query, params))
            
            # Combine results with layer 1 first
            all_results = layer1_results + layer2_results
            
            # Additional Python filtering to ensure no duplicate paragraphs
            # This is a backup in case DISTINCT in Cypher doesn't work perfectly
            seen_ids = set()
            unique_results = []
            
            for result in all_results:
                paragraph_id = result['id']
                if paragraph_id not in seen_ids:
                    seen_ids.add(paragraph_id)
                    unique_results.append(result)
            
            # Sort results by similarity score
            if order_by == "similarity":
                unique_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            elif order_by == "quality_score":
                unique_results.sort(key=lambda x: x['quality_score'], reverse=True)
            
            return unique_results
        
        except Exception as e:
            print(f"Error in semantic search: {e}")
            
            # If vector search is not available, fall back to topic-based search
            fallback_query = """
            MATCH (hp:HateParagraph)-[:TALKS_ABOUT]->(t:Topic)
            WHERE toLower(t.name) CONTAINS toLower($query_text)
            OPTIONAL MATCH (hp)-[:COUNTERED_WITH]->(cp:CounterParagraph)
            
            // Use DISTINCT to ensure each paragraph appears only once
            WITH DISTINCT hp, t, cp
            
            RETURN hp.id AS id, 
                   hp.content AS content,
                   t.name AS topic,
                   hp.quality_score AS quality_score,
                   cp.id AS counter_id, 
                   cp.content AS counter_content,
                   1.0 AS similarity_score,
                   CASE WHEN hp.is_synthetic = true THEN 2 ELSE 1 END AS layer
            LIMIT $limit
            """
            
            fallback_params = {"query_text": query_text, "limit": limit_per_layer * 2}
            fallback_results = list(self.memgraph.execute_and_fetch(fallback_query, fallback_params))
            
            # Additional Python filtering for fallback results
            seen_ids = set()
            unique_fallback_results = []
            
            for result in fallback_results:
                paragraph_id = result['id']
                if paragraph_id not in seen_ids:
                    seen_ids.add(paragraph_id)
                    unique_fallback_results.append(result)
            
            return unique_fallback_results
        
    def _ensure_vector_indices(self):
        """
        Ensure that vector indices exist for hate content embeddings.
        This should be called once during initialization or setup.
        """
        try:
            # Check if index exists
            check_query = """
            SHOW INDEX INFO
            """
            indices = list(self.memgraph.execute_and_fetch(check_query))
            
            # Look for our index in the returned indices
            index_exists = any(index.get('name') == 'hate_content_embedding_index' for index in indices)
            
            if not index_exists:
                # Create vector index for HateContent embeddings
                create_index_query = """
                CREATE VECTOR INDEX hate_content_embedding_index ON :HateContent(embedding) 
                WITH CONFIG {"dimension": 384, "capacity": 1000, "metric": "cos"};
                """
                self.memgraph.execute(create_index_query)
                print("Created vector index for HateContent embeddings")
        except Exception as e:
            print(f"Warning: Could not ensure vector indices: {e}")
            print("Semantic search will fall back to alternative methods if vector search fails")