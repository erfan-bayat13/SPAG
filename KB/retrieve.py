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
        
        WITH hp, t, cp, 
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
        all_results = layer1_results + layer2_results
        
        return all_results
        
    def semantic_search(self, query_text: str, limit_per_layer: int = 5, similarity_threshold: float = 0.5) -> List[Dict]:
        """
        Perform semantic search on hate content using embeddings.
        Searches both debate-based content (layer 1) and synthetic content (layer 2).
        
        Args:
            query_text: The search query text
            limit_per_layer: Maximum number of results to return per layer
            similarity_threshold: Minimum cosine similarity score (0-1) for results
            
        Returns:
            List of matching hate content with similarity scores, ordered by relevance
        """
        # Lazy-load the embedding model if it's not already loaded
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embedding for the query text
        query_embedding = self.model.encode(query_text).tolist()
        query_embedding_str = json.dumps(query_embedding)
        
        # Layer 1: Search in debate-based hate content
        layer1_query = """
        MATCH (hp:HateParagraph)-[:CONTAINS]->(hc:HateContent)
        WHERE (hp.is_synthetic IS NULL OR hp.is_synthetic = false)
        AND hc.embedding IS NOT NULL
        
        // Calculate cosine similarity
        WITH hp, hc, gds.similarity.cosine($query_embedding, hc.embedding) AS similarity
        WHERE similarity >= $threshold
        
        // Get related nodes
        OPTIONAL MATCH (hp)-[:TALKS_ABOUT]->(t:Topic)
        OPTIONAL MATCH (hp)-[:COUNTERED_WITH]->(cp:CounterParagraph)
        
        // Aggregate results at paragraph level
        WITH hp, t, cp, MAX(similarity) AS max_similarity
        
        RETURN hp.id AS id, 
               hp.content AS content,
               t.name AS topic,
               hp.quality_score AS quality_score,
               cp.id AS counter_id, 
               cp.content AS counter_content,
               max_similarity AS similarity_score,
               1 AS layer
        ORDER BY similarity_score DESC
        LIMIT $limit
        """
        
        # Layer 2: Search in synthetic hate content
        layer2_query = """
        MATCH (hp:HateParagraph)-[:CONTAINS]->(hc:HateContent)
        WHERE hp.is_synthetic = true
        AND hc.embedding IS NOT NULL
        
        // Calculate cosine similarity
        WITH hp, hc, gds.similarity.cosine($query_embedding, hc.embedding) AS similarity
        WHERE similarity >= $threshold
        
        // Get related nodes
        OPTIONAL MATCH (hp)-[:TALKS_ABOUT]->(t:Topic)
        OPTIONAL MATCH (hp)-[:COUNTERED_WITH]->(cp:CounterParagraph)
        
        // Aggregate results at paragraph level
        WITH hp, t, cp, MAX(similarity) AS max_similarity
        
        RETURN hp.id AS id, 
               hp.content AS content,
               t.name AS topic,
               hp.quality_score AS quality_score,
               cp.id AS counter_id, 
               cp.content AS counter_content,
               max_similarity AS similarity_score,
               2 AS layer
        ORDER BY similarity_score DESC
        LIMIT $limit
        """
        
        # Execute queries
        params = {
            "query_embedding": query_embedding,
            "threshold": similarity_threshold,
            "limit": limit_per_layer
        }
        
        try:
            layer1_results = list(self.memgraph.execute_and_fetch(layer1_query, params))
            layer2_results = list(self.memgraph.execute_and_fetch(layer2_query, params))
            
            # Combine results with layer 1 first
            all_results = layer1_results + layer2_results
            
            return all_results
        except Exception as e:
            print(f"Error in semantic search: {e}")
            
            # If cosine similarity function is not available, try a simpler approach
            # This fallback doesn't use embeddings but provides results based on topic
            fallback_query = """
            MATCH (hp:HateParagraph)-[:TALKS_ABOUT]->(t:Topic)
            WHERE toLower(t.name) CONTAINS toLower($query_text)
            OPTIONAL MATCH (hp)-[:COUNTERED_WITH]->(cp:CounterParagraph)
            
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
            
            return fallback_results