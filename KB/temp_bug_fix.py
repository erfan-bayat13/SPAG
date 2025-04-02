from gqlalchemy import Memgraph
import json

def fix_existing_embeddings(memgraph_host="127.0.0.1", memgraph_port=7687):
    """Fix existing embeddings in the database and recreate the vector index."""
    # Connect to Memgraph
    memgraph = Memgraph(memgraph_host, memgraph_port)
    
    print("Fixing existing embeddings...")
    
    # First, check if we have HateContent nodes
    check_nodes_query = """
    MATCH (s:HateContent) 
    RETURN count(s) as node_count
    """
    
    result = next(memgraph.execute_and_fetch(check_nodes_query))
    node_count = result['node_count']
    
    if node_count == 0:
        print("No HateContent nodes found in database")
        return False
    else:
        print(f"Found {node_count} HateContent nodes")
    
    # Check embedding format by getting a sample
    check_query = """
    MATCH (s:HateContent) 
    WHERE s.embedding IS NOT NULL
    RETURN s.embedding AS sample_embedding
    LIMIT 1
    """
    
    result = list(memgraph.execute_and_fetch(check_query))
    if not result:
        print("No embeddings found in HateContent nodes")
        return False
    
    sample_embedding = result[0]['sample_embedding']
    print(f"Sample embedding: {str(sample_embedding)[:100]}...")
    
    # Determine the type in Python
    if isinstance(sample_embedding, str):
        print("Embeddings are stored as strings, converting to arrays...")
        
        # First try: Handle JSON string format
        try:
            # Try parsing with json
            try:
                parsed = json.loads(sample_embedding)
                if isinstance(parsed, list):
                    print(f"Confirmed JSON format. Sample parsed: {str(parsed[:5])}...")
                    
                    # Update nodes batch by batch to avoid memory issues
                    batch_size = 100
                    total_processed = 0
                    
                    query = """
                    MATCH (s:HateContent)
                    WHERE s.embedding IS NOT NULL
                    RETURN s.id AS id
                    ORDER BY s.id
                    SKIP $skip LIMIT $limit
                    """
                    
                    while True:
                        batch = list(memgraph.execute_and_fetch(query, {'skip': total_processed, 'limit': batch_size}))
                        if not batch:
                            break
                        
                        print(f"Processing batch of {len(batch)} nodes...")
                        for node in batch:
                            # Get embedding for this node
                            get_embedding_query = """
                            MATCH (s:HateContent {id: $id})
                            RETURN s.embedding AS embedding
                            """
                            embedding_result = next(memgraph.execute_and_fetch(get_embedding_query, {'id': node['id']}))
                            embedding_str = embedding_result['embedding']
                            
                            # Skip if already an array
                            if not isinstance(embedding_str, str):
                                continue
                                
                            # Parse JSON to get array
                            try:
                                embedding_array = json.loads(embedding_str)
                                # Convert to comma-separated string for Cypher
                                array_str = ','.join(str(x) for x in embedding_array)
                                
                                # Update node with proper array
                                update_query = f"""
                                MATCH (s:HateContent {{id: {node['id']}}})
                                SET s.embedding = [{array_str}]
                                """
                                memgraph.execute(update_query)
                            except Exception as e:
                                print(f"Error processing node {node['id']}: {e}")
                        
                        total_processed += len(batch)
                        print(f"Processed {total_processed} nodes so far")
                    
                    print(f"Finished processing {total_processed} nodes")
                    
                else:
                    print("Parsed embedding is not a list, trying alternative approach")
                    raise ValueError("Not a valid JSON list")
                    
            except json.JSONDecodeError:
                raise ValueError("Not a valid JSON string")
                
        except Exception as e:
            print(f"Error converting JSON embeddings: {e}")
            print("Trying alternative approach with comma-separated values...")
            
            # Try comma-separated approach
            try:
                # Check if it's comma-separated
                if ',' in sample_embedding:
                    print("Detected comma-separated format")
                    
                    # Process in batches for comma-separated values too
                    batch_size = 100
                    total_processed = 0
                    
                    query = """
                    MATCH (s:HateContent)
                    WHERE s.embedding IS NOT NULL
                    RETURN s.id AS id
                    ORDER BY s.id
                    SKIP $skip LIMIT $limit
                    """
                    
                    while True:
                        batch = list(memgraph.execute_and_fetch(query, {'skip': total_processed, 'limit': batch_size}))
                        if not batch:
                            break
                        
                        print(f"Processing batch of {len(batch)} nodes (CSV format)...")
                        for node in batch:
                            try:
                                # Get embedding for this node
                                get_embedding_query = """
                                MATCH (s:HateContent {id: $id})
                                RETURN s.embedding AS embedding
                                """
                                embedding_result = next(memgraph.execute_and_fetch(get_embedding_query, {'id': node['id']}))
                                embedding_str = embedding_result['embedding']
                                
                                # Skip if already an array
                                if not isinstance(embedding_str, str):
                                    continue
                                
                                # Split by comma and convert to array
                                embed_parts = embedding_str.split(',')
                                array_str = ','.join(embed_parts)
                                
                                # Update node with proper array
                                update_query = f"""
                                MATCH (s:HateContent {{id: {node['id']}}})
                                SET s.embedding = [{array_str}]
                                """
                                memgraph.execute(update_query)
                            except Exception as e:
                                print(f"Error processing CSV node {node['id']}: {e}")
                        
                        total_processed += len(batch)
                        print(f"Processed {total_processed} nodes so far (CSV format)")
                    
                    print(f"Finished processing {total_processed} nodes (CSV format)")
                else:
                    print(f"Unknown embedding format: {sample_embedding[:50]}...")
                    return False
            except Exception as e2:
                print(f"Error converting CSV embeddings: {e2}")
                return False
    elif isinstance(sample_embedding, list):
        print("Embeddings are already stored as arrays. No conversion needed.")
    else:
        print(f"Embeddings are stored in an unexpected format: {type(sample_embedding).__name__}")
        return False
    
    # Drop existing index
    try:
        print("Checking for existing indices...")
        show_index_query = "SHOW INDEX INFO;"
        indices = list(memgraph.execute_and_fetch(show_index_query))
        
        vector_index_exists = False
        for idx in indices:
            if idx.get('name') == 'hate_content_embedding_index':
                vector_index_exists = True
                break
        
        if vector_index_exists:
            print("Dropping existing vector index...")
            drop_query = "DROP INDEX hate_content_embedding_index;"
            memgraph.execute(drop_query)
            print("Existing index dropped")
    except Exception as e:
        print(f"Note: Could not check or drop index: {e}")
        print("Proceeding to create index...")
    
    # Create new vector index
    print("Creating new vector index...")
    create_index_query = """
    CREATE VECTOR INDEX hate_content_embedding_index ON :HateContent(embedding) 
    WITH CONFIG {"dimension": 384, "capacity": 1000, "metric": "cos"};
    """
    
    try:
        memgraph.execute(create_index_query)
        print("Vector index created successfully")
        
        # Verify index population
        check_index_query = "SHOW INDEX INFO;"
        indices = list(memgraph.execute_and_fetch(check_index_query))
        for idx in indices:
            if idx.get('name') == 'hate_content_embedding_index':
                print(f"Vector index name: {idx.get('name')}")
                print(f"Vector index count: {idx.get('count', 'unknown')}")
        
        return True
    except Exception as e:
        print(f"Error creating vector index: {e}")
        return False

if __name__ == "__main__":
    # Run the fix
    success = fix_existing_embeddings()
    
    if success:
        print("\nEmbedding fix completed successfully!")
        print("You should now be able to run semantic search.")
    else:
        print("\nEmbedding fix encountered issues.")
        print("Please check the error messages above for details.")
        