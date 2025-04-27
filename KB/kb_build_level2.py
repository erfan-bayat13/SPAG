from gqlalchemy import Memgraph, Node, Relationship, Field
import json
from typing import Optional

# Make sure these class definitions match what you already have in your codebase
class Topic(Node):
    name: str = Field(index=True, unique=True, exists=True, db=Memgraph("127.0.0.1", 7687))

class HateParagraph(Node):
    id: int = Field(index=True, unique=True, exists=True, db=Memgraph("127.0.0.1", 7687))
    content: str = Field()
    source_debate: int = Field()
    representativeness: float = Field()
    coherence: float = Field()
    harmfulness: float = Field()
    quality_score: float = Field()
    is_synthetic: bool = Field()
    content_type: str = Field()

class TalksAbout(Relationship, type="TALKS_ABOUT"):
    pass

def remove_synthetic_hate_paragraphs(memgraph):
    """
    Remove all synthetic hate paragraphs that were recently added
    """
    print("Removing synthetic hate paragraphs...")
    
    # Query to count how many nodes will be deleted
    count_query = """
    MATCH (hp:HateParagraph)
    WHERE hp.is_synthetic = true
    RETURN count(hp) as count
    """
    
    count_result = next(memgraph.execute_and_fetch(count_query))
    print(f"Found {count_result['count']} synthetic hate paragraphs to remove")
    
    # Delete the synthetic nodes and their relationships
    delete_query = """
    MATCH (hp:HateParagraph)
    WHERE hp.is_synthetic = true
    DETACH DELETE hp
    """
    
    memgraph.execute(delete_query)
    print("Synthetic hate paragraphs removed successfully")

def extend_knowledge_base_with_generated_content(jsonl_file_path, memgraph_connection):
    """
    Import generated hate speech examples from a JSONL file into the knowledge base.
    
    Parameters:
    - jsonl_file_path: Path to the JSONL file with generated hate speech
    - memgraph_connection: Memgraph connection instance
    """
    print(f"Extending knowledge base with generated content from: {jsonl_file_path}")
    
    # Load generated examples
    examples_by_topic = {}
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
                
            try:
                example = json.loads(line.strip())
                topic = example.get('topic', '').strip()
                
                if not topic:
                    continue
                    
                if topic not in examples_by_topic:
                    examples_by_topic[topic] = []
                    
                examples_by_topic[topic].append(example)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line[:50]}... - {str(e)}")
                continue
    
    print(f"Loaded examples for {len(examples_by_topic)} topics")
    
    # Get the highest existing ID to avoid conflicts
    max_id_query = "MATCH (hp:HateParagraph) RETURN coalesce(max(hp.id), 0) as max_id"
    max_hate_id = next(memgraph_connection.execute_and_fetch(max_id_query))["max_id"]
    print(f"Current maximum HateParagraph ID: {max_hate_id}")
    
    # Track statistics for reporting
    stats = {
        "topics_processed": 0,
        "examples_imported": 0,
        "errors": 0
    }
    
    # Process each topic
    for topic_name, examples in examples_by_topic.items():
        print(f"Processing topic: {topic_name} with {len(examples)} examples")
        stats["topics_processed"] += 1
        
        # Find or create Topic node
        topic_node = None
        try:
            # Try to find existing topic
            topic_query = f"MATCH (t:Topic {{name: '{topic_name}'}}) RETURN t"
            topic_result = list(memgraph_connection.execute_and_fetch(topic_query))
            
            if topic_result:
                # Topic exists
                topic_node_from_query = topic_result[0]["t"]                
                # Create a proper Topic node instance
                topic_node = Topic(name=topic_name)
                topic_node._id = topic_node_from_query._id
            else:
                # Create new Topic
                topic_node = Topic(name=topic_name)
                topic_node.save(memgraph_connection)
        except Exception as e:
            print(f"Error with topic {topic_name}: {str(e)}")
            stats["errors"] += 1
            continue
        
        if topic_node is None or not hasattr(topic_node, '_id'):
            print(f"Failed to create or find topic node for {topic_name}")
            continue
            
        # Import each example
        current_hate_id = max_hate_id
        
        for example in examples:
            current_hate_id += 1
            
            hate_content = example["content"]
            
            try:
                # Create HateParagraph node
                hate_node = HateParagraph(
                    id=current_hate_id,
                    content=hate_content,
                    source_debate=0,
                    representativeness=7.0,
                    coherence=7.0,
                    harmfulness=8.0,
                    quality_score=7.5,
                    is_synthetic=True,
                    content_type='social_media'
                )
                hate_node.save(memgraph_connection)
                
                # Create relationship
                talks_about_rel = TalksAbout(_start_node_id=hate_node._id, _end_node_id=topic_node._id)
                talks_about_rel.save(memgraph_connection)
                
                stats["examples_imported"] += 1
            except Exception as e:
                print(f"Error importing example for topic {topic_name}: {str(e)}")
                stats["errors"] += 1
                continue
        
        # Update max ID for next topic
        max_hate_id = current_hate_id
    
    # Print summary statistics
    print("\nImport Summary:")
    print(f"Topics processed: {stats['topics_processed']}")
    print(f"Examples imported: {stats['examples_imported']}")
    print(f"Errors encountered: {stats['errors']}")
    
    return stats

if __name__ == "__main__":
    # Connect to Memgraph
    memgraph = Memgraph("127.0.0.1", 7687)
    
    # Import the generated content
    stats = extend_knowledge_base_with_generated_content(
        jsonl_file_path="/Users/erfanbayat/Downloads/generated_hate_speech (2).jsonl",
        memgraph_connection=memgraph
    )