import json
import csv
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the model for sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def split_into_sentences(text):
    """Split a paragraph into sentences"""
    # Basic sentence splitting - this could be improved with a more sophisticated NLP approach
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Clean up sentences and remove empty ones
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def extract_turn_number(text):
    """Extract turn number from the content"""
    turn_match = re.search(r'Turn (\d+)', text)
    if turn_match:
        return int(turn_match.group(1))
    # Look for other indicators
    if "Opening Statement" in text or "Turn 1" in text:
        return 1
    if "Turn 2" in text:
        return 2
    if "Turn 3" in text:
        return 3
    if "Turn 4" in text:
        return 4
    # Default to 1 if no turn number found
    return 1

def process_debates(json_file):
    """Process the debate JSON file and extract information for CSVs"""
    # Output file paths
    os.makedirs('output1', exist_ok=True)
    hate_paragraphs_file = 'output1/nodes_hate_paragraphs.csv'
    hate_sentences_file = 'output1/nodes_hate_sentences.csv'
    counter_paragraphs_file = 'output1/nodes_counter_paragraphs.csv'
    
    # Read the JSON data
    with open(json_file, 'r') as f:
        debates = json.load(f)
    
    # Initialize data structures for each CSV
    hate_paragraphs = []
    hate_sentences = []
    counter_paragraphs = []
    
    # ID counters
    hate_paragraph_id = 0
    hate_sentence_id = 0
    counter_paragraph_id = 0
    
    # Process each debate
    for debate_index, debate in enumerate(debates):
        # Extract debate info
        topic = debate['topic']
        position = debate['position']
        source_debate_id = debate_index + 1
        
        # Process proponent (hate speech) content
        proponent_content = debate['content']['proponent']
        
        # Extract evaluation metrics for harmful content
        harmful_representativeness = 0
        harmful_coherence = 0
        harmful_harmfulness = 0
        harmful_subtotal = 0
        
        if 'evaluation' in debate and 'scores' in debate['evaluation']:
            scores = debate['evaluation']['scores']
            if 'harmful_content' in scores:
                harmful_representativeness = scores['harmful_content'].get('representativeness', 0)
                harmful_coherence = scores['harmful_content'].get('coherence', 0)
                harmful_harmfulness = scores['harmful_content'].get('harmfulness', 0)
                harmful_subtotal = scores['harmful_content'].get('subtotal', 0)
            # For backward compatibility with different naming schemes
            elif 'hate_speech' in scores:
                harmful_representativeness = scores['hate_speech'].get('representativeness', 0)
                harmful_coherence = scores['hate_speech'].get('coherence', 0)
                harmful_harmfulness = scores['hate_speech'].get('specificity', 0)  # Sometimes named differently
                harmful_subtotal = scores['hate_speech'].get('subtotal', 0)
        
        # Treat the entire turn as a single paragraph
        hate_paragraph_id += 1
        hate_paragraphs.append({
            'id': hate_paragraph_id,
            'content': proponent_content,
            'topic': topic,
            'source_debate': source_debate_id,
            'representativeness': harmful_representativeness,
            'coherence': harmful_coherence,
            'harmfulness': harmful_harmfulness,
            'quality_score': harmful_subtotal
        })
        
        sentences = split_into_sentences(proponent_content)
        for sentence in sentences:
            # Create embedding for sentence
            embedding = model.encode(sentence).tolist()
            
            # Store as comma-separated string (better for database import)
            embedding_str = ','.join(str(x) for x in embedding)
            
            # Add hate sentence
            hate_sentence_id += 1
            hate_sentences.append({
                'id': hate_sentence_id,
                'content': sentence,
                'paragraph_id': hate_paragraph_id,
                'embedding': embedding_str  # Stored as comma-separated values
            })
        
        # Process opponent (counterspeech) content
        opponent_content = debate['content']['opponent']
        
        # Extract evaluation metrics for counterspeech
        directness = 0
        evidence_quality = 0
        persuasiveness = 0
        counterspeech_subtotal = 0
        
        if 'evaluation' in debate and 'scores' in debate['evaluation']:
            scores = debate['evaluation']['scores']
            if 'counterspeech' in scores:
                directness = scores['counterspeech'].get('directness', 0)
                evidence_quality = scores['counterspeech'].get('evidence_quality', 0)
                persuasiveness = scores['counterspeech'].get('persuasiveness', 0)
                counterspeech_subtotal = scores['counterspeech'].get('subtotal', 0)
        
        # Get assessment if available
        assessment = ""
        if 'evaluation' in debate and 'assessment' in debate['evaluation']:
            assessment = debate['evaluation']['assessment']
        
        # Create counter paragraph node
        counter_paragraph_id += 1
        counter_paragraphs.append({
            'id': counter_paragraph_id,
            'content': opponent_content,
            'topic': topic,
            'counters_id': hate_paragraph_id,
            'directness': directness,
            'evidence_quality': evidence_quality,
            'persuasiveness': persuasiveness,
            'quality_score': counterspeech_subtotal,
            'assessment': assessment
        })
    
    # Write to CSV files
    with open(hate_paragraphs_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'id', 'content', 'topic', 'source_debate', 
            'representativeness', 'coherence', 'harmfulness', 'quality_score'
        ])
        writer.writeheader()
        for item in hate_paragraphs:
            writer.writerow(item)
    
    with open(hate_sentences_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'content', 'paragraph_id', 'embedding'])
        writer.writeheader()
        for item in hate_sentences:
            writer.writerow(item)
    
    with open(counter_paragraphs_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'id', 'content', 'topic', 'counters_id',
            'directness', 'evidence_quality', 'persuasiveness', 'quality_score', 'assessment'
        ])
        writer.writeheader()
        for item in counter_paragraphs:
            writer.writerow(item)
    
    return {
        'hate_paragraphs': len(hate_paragraphs),
        'hate_sentences': len(hate_sentences),
        'counter_paragraphs': len(counter_paragraphs)
    }

if __name__ == "__main__":
    # File path to the evaluation results JSON
    json_file = json_path
    
    # Process the debates and create CSV files
    result = process_debates(json_file)
    
    # Print summary
    print(f"Processing complete. Generated CSV files in the 'output' directory.")
    print(f"Summary of extracted data:")
    for key, value in result.items():
        print(f"  {key}: {value} items")