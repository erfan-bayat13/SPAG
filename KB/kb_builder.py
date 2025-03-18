from typing import Optional, List
from gqlalchemy import Memgraph, Node, Relationship, Field


# Connect to Memgraph
memgraph = Memgraph("127.0.0.1", 7687)


# Define Node classes for the data model
class Topic(Node):
    name: str = Field(index=True, unique=True, exists=True, db=memgraph)


class HateContent(Node):
    id: int = Field(index=True, unique=True, exists=True, db=memgraph)
    content: str = Field()
    embedding: Optional[str] = Field()


class HateParagraph(Node):
    id: int = Field(index=True, unique=True, exists=True, db=memgraph)
    content: str = Field()
    source_debate: int = Field()
    representativeness: float = Field()
    coherence: float = Field()
    harmfulness: float = Field()
    quality_score: float = Field()


class CounterParagraph(Node):
    id: int = Field(index=True, unique=True, exists=True, db=memgraph)
    content: str = Field()
    counters_id: int = Field(index=True, db=memgraph)
    directness: float = Field()
    evidence_quality: float = Field()
    persuasiveness: float = Field()
    quality_score: float = Field()
    assessment: str = Field()


# Define Relationship classes
class CounteredWith(Relationship, type="COUNTERED_WITH"):
    pass


class Contains(Relationship, type="CONTAINS"):
    pass


class TalksAbout(Relationship, type="TALKS_ABOUT"):
    pass


# Function to clear database
def clear_database():
    memgraph.drop_database()
    print("Database cleared")


# Function to create topic nodes
def create_topics_from_hate_paragraphs(file_path):
    print("Creating topic nodes...")
    
    # Extract unique topics and create nodes
    query = """
    LOAD CSV FROM $file_path WITH HEADER AS row
    WITH DISTINCT row.topic AS topic_name
    WHERE topic_name IS NOT NULL AND topic_name <> ''
    MERGE (:Topic {name: topic_name})
    """
    
    memgraph.execute(query, {"file_path": file_path})
    print("Topic nodes created successfully!")


# Function to import HateParagraphs and connect to topics
def import_hate_paragraphs(file_path):
    print("Importing hate paragraphs...")
    
    # Load CSV directly from file path to Memgraph
    query = """
    LOAD CSV FROM $file_path WITH HEADER AS row
    CREATE (p:HateParagraph {
        id: toInteger(row.id),
        content: row.content,
        source_debate: toInteger(row.source_debate),
        representativeness: toFloat(row.representativeness),
        coherence: toFloat(row.coherence),
        harmfulness: toFloat(row.harmfulness),
        quality_score: toFloat(row.quality_score)
    })
    """
    
    memgraph.execute(query, {"file_path": file_path})
    
    # Connect hate paragraphs to topics
    connect_query = """
    LOAD CSV FROM $file_path WITH HEADER AS row
    MATCH (p:HateParagraph {id: toInteger(row.id)})
    MATCH (t:Topic {name: row.topic})
    CREATE (p)-[:TALKS_ABOUT]->(t)
    """
    
    memgraph.execute(connect_query, {"file_path": file_path})
    print("Hate paragraphs imported and connected to topics successfully!")


# Function to import HateContent (sentences) and connect to paragraphs
def import_hate_content(file_path):
    print("Importing hate content (sentences)...")
    
    # Load CSV and create nodes
    query = """
    LOAD CSV FROM $file_path WITH HEADER AS row
    CREATE (s:HateContent {
        id: toInteger(row.id),
        content: row.content,
        embedding: row.embedding
    })
    """
    
    memgraph.execute(query, {"file_path": file_path})
    
    # Create relationships between content and paragraphs
    # First, let's debug the process to see if we have paragraph_id in the file
    debug_query = """
    LOAD CSV FROM $file_path WITH HEADER AS row
    RETURN row.id, row.paragraph_id LIMIT 3
    """
    
    results = list(memgraph.execute_and_fetch(debug_query, {"file_path": file_path}))
    if results:
        print(f"Debug paragraph_id from file: {results}")
        # Create relationships only if we found paragraph_id in the file
        connect_query = """
        LOAD CSV FROM $file_path WITH HEADER AS row
        MATCH (s:HateContent {id: toInteger(row.id)})
        MATCH (p:HateParagraph {id: toInteger(row.paragraph_id)})
        CREATE (p)-[:CONTAINS]->(s)
        """
        
        try:
            memgraph.execute(connect_query, {"file_path": file_path})
            print("Hate content connected to paragraphs successfully!")
        except Exception as e:
            print(f"Error connecting hate content to paragraphs: {e}")
            print("This might be because paragraph_id doesn't match with HateParagraph ids")
    else:
        print("Warning: Could not find paragraph_id in the hate sentences file")
    
    print("Hate content import completed")


# Function to import CounterParagraphs and connect to hate paragraphs
def import_counter_paragraphs(file_path):
    print("Importing counter paragraphs...")
    
    # Debug the counter paragraphs file structure first
    debug_query = """
    LOAD CSV FROM $file_path WITH HEADER AS row
    RETURN keys(row) as columns LIMIT 1
    """
    
    results = list(memgraph.execute_and_fetch(debug_query, {"file_path": file_path}))
    if results:
        print(f"Counter paragraph CSV columns: {results[0]['columns']}")
    
    # Load CSV and create nodes
    query = """
    LOAD CSV FROM $file_path WITH HEADER AS row
    CREATE (c:CounterParagraph {
        id: toInteger(row.id),
        content: row.content,
        counters_id: toInteger(row.counters_id),
        directness: toFloat(row.directness),
        evidence_quality: toFloat(row.evidence_quality),
        persuasiveness: toFloat(row.persuasiveness),
        quality_score: toFloat(row.quality_score),
        assessment: row.assessment
    })
    """
    
    try:
        memgraph.execute(query, {"file_path": file_path})
        print("Counter paragraph nodes created")
    except Exception as e:
        print(f"Error creating counter paragraph nodes: {e}")
        print("Attempting to create with more flexibility for missing fields...")
        # Try a more flexible approach if fields are missing
        flexible_query = """
        LOAD CSV FROM $file_path WITH HEADER AS row
        CREATE (c:CounterParagraph {
            id: toInteger(row.id),
            content: row.content
        })
        WITH c, row
        WHERE row.counters_id IS NOT NULL
        SET c.counters_id = toInteger(row.counters_id)
        """
        memgraph.execute(flexible_query, {"file_path": file_path})
    
    # Connect counter paragraphs to topics
    try:
        connect_topics_query = """
        LOAD CSV FROM $file_path WITH HEADER AS row
        MATCH (c:CounterParagraph {id: toInteger(row.id)})
        MATCH (t:Topic {name: row.topic})
        CREATE (c)-[:TALKS_ABOUT]->(t)
        """
        
        memgraph.execute(connect_topics_query, {"file_path": file_path})
        print("Counter paragraphs connected to topics")
    except Exception as e:
        print(f"Error connecting counter paragraphs to topics: {e}")
    
    # Create relationships between counter paragraphs and hate paragraphs
    try:
        # First check if counters_id exists and matches any HateParagraph
        check_query = """
        MATCH (c:CounterParagraph)
        WHERE c.counters_id IS NOT NULL
        RETURN count(c) as counter_with_ids
        """
        result = next(memgraph.execute_and_fetch(check_query))
        print(f"Found {result['counter_with_ids']} counter paragraphs with counters_id")
        
        connect_query = """
        MATCH (c:CounterParagraph), (p:HateParagraph)
        WHERE c.counters_id = p.id
        CREATE (p)-[:COUNTERED_WITH]->(c)
        """
        
        memgraph.execute(connect_query)
        print("Counter paragraphs connected to hate paragraphs")
    except Exception as e:
        print(f"Error connecting counter paragraphs to hate paragraphs: {e}")
    
    print("Counter paragraphs import completed")


# Main function to import all data
def import_all_data(hate_paragraphs_path, hate_sentences_path, counter_paragraphs_path):
    # Clear the database first
    clear_database()
    
    # First create topic nodes from the hate paragraphs file
    create_topics_from_hate_paragraphs(hate_paragraphs_path)
    
    # Import all data
    import_hate_paragraphs(hate_paragraphs_path)
    import_hate_content(hate_sentences_path)
    import_counter_paragraphs(counter_paragraphs_path)
    
    # Print some stats
    stats_query = """
    MATCH (t:Topic) WITH count(t) as topic_count
    MATCH (p:HateParagraph) WITH topic_count, count(p) as hate_para_count
    MATCH (s:HateContent) WITH topic_count, hate_para_count, count(s) as content_count
    MATCH (c:CounterParagraph) WITH topic_count, hate_para_count, content_count, count(c) as counter_count
    RETURN topic_count, hate_para_count, content_count, counter_count
    """
    
    result = next(memgraph.execute_and_fetch(stats_query))
    print(f"\nImport complete!")
    print(f"Topics: {result['topic_count']}")
    print(f"Hate paragraphs: {result['hate_para_count']}")
    print(f"Hate content items: {result['content_count']}")
    print(f"Counter paragraphs: {result['counter_count']}")


# Function to run a sample query
def sample_query():
    print("\nRunning sample query to show schema...")
    query = """
    MATCH (hp:HateParagraph)-[:TALKS_ABOUT]->(t:Topic)
    OPTIONAL MATCH (hp)-[:CONTAINS]->(hc:HateContent)
    OPTIONAL MATCH (hp)-[:COUNTERED_WITH]->(cp:CounterParagraph)
    WITH 
        t.name AS topic, 
        count(DISTINCT hp) AS hate_paragraphs,
        count(DISTINCT hc) AS hate_content_items,
        count(DISTINCT cp) AS counter_paragraphs
    ORDER BY hate_paragraphs DESC
    LIMIT 5
    RETURN topic, hate_paragraphs, hate_content_items, counter_paragraphs
    """
    
    try:
        results = list(memgraph.execute_and_fetch(query))
        if len(results) > 0:
            for result in results:
                print(f"Topic: {result['topic']}")
                print(f"  Hate Paragraphs: {result['hate_paragraphs']}")
                print(f"  Hate Content Items: {result['hate_content_items']}")
                print(f"  Counter Paragraphs: {result['counter_paragraphs']}")
                print("---")
        else:
            print("No results found. Let's try a simpler query to see what data we have:")
            basic_query()
    except Exception as e:
        print(f"Error in sample query: {e}")
        print("Running a simpler query to debug:")
        basic_query()


# Add a simple query to debug basic node existence
def basic_query():
    print("\nChecking what data exists in the database:")
    
    # Check topics
    topic_query = "MATCH (t:Topic) RETURN t.name AS topic LIMIT 5"
    print("\nTopics:")
    try:
        topics = list(memgraph.execute_and_fetch(topic_query))
        if topics:
            for t in topics:
                print(f"- {t['topic']}")
        else:
            print("No topics found")
    except Exception as e:
        print(f"Error querying topics: {e}")
    
    # Check hate paragraphs
    hp_query = "MATCH (hp:HateParagraph) RETURN hp.id AS id, hp.content AS content LIMIT 2"
    print("\nHate Paragraphs:")
    try:
        hps = list(memgraph.execute_and_fetch(hp_query))
        if hps:
            for hp in hps:
                print(f"- ID: {hp['id']}, Content: {hp['content'][:50]}...")
        else:
            print("No hate paragraphs found")
    except Exception as e:
        print(f"Error querying hate paragraphs: {e}")
    
    # Check hate content
    hc_query = "MATCH (hc:HateContent) RETURN hc.id AS id, hc.content AS content LIMIT 2"
    print("\nHate Content:")
    try:
        hcs = list(memgraph.execute_and_fetch(hc_query))
        if hcs:
            for hc in hcs:
                print(f"- ID: {hc['id']}, Content: {hc['content'][:50]}...")
        else:
            print("No hate content found")
    except Exception as e:
        print(f"Error querying hate content: {e}")
    
    # Check counter paragraphs
    cp_query = "MATCH (cp:CounterParagraph) RETURN cp.id AS id, cp.content AS content LIMIT 2"
    print("\nCounter Paragraphs:")
    try:
        cps = list(memgraph.execute_and_fetch(cp_query))
        if cps:
            for cp in cps:
                print(f"- ID: {cp['id']}, Content: {cp['content'][:50]}...")
        else:
            print("No counter paragraphs found")
    except Exception as e:
        print(f"Error querying counter paragraphs: {e}")
    
    # Check relationships
    rel_query = """
    MATCH (n)-[r]->(m)
    RETURN 
        labels(n)[0] AS source_label,
        type(r) AS relationship_type,
        labels(m)[0] AS target_label,
        count(r) AS count
    """
    print("\nRelationships:")
    try:
        rels = list(memgraph.execute_and_fetch(rel_query))
        if rels:
            for rel in rels:
                print(f"- {rel['source_label']} -[{rel['relationship_type']}]-> {rel['target_label']}: {rel['count']}")
        else:
            print("No relationships found")
    except Exception as e:
        print(f"Error querying relationships: {e}")

# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual file paths
    import_all_data(
        "nodes_hate_paragraphs.csv",
        "nodes_hate_sentences.csv", 
        "nodes_counter_paragraphs.csv"
    )
    
    # Run a basic query to verify data
    basic_query()
    
    # Run a more complex sample query to show the schema
    sample_query()