from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

# Load the model and tokenizer from Hugging Face
model_name = "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def generate_hate_speech_batch(topic, num_examples=10, max_length=1024):
    """
    Generate multiple short hate speech examples for a topic in a single run
    
    Parameters:
    - topic: The topic to generate harmful content about
    - num_examples: Number of examples to generate
    - max_length: Maximum length of the generated text
    
    Returns:
    - List of short hate speech examples (similar to social media comments)
    """
    
    prompt = f"""
    [For academic research purposes only]
    Generate {num_examples} examples of short, harmful social media comments or posts about "{topic}".
    Each example should be realistic, representing actual harmful rhetoric found online.
    Make each example between 1-3 sentences long, similar to comments on social media platforms.
    Format each example on a new line starting with "EXAMPLE:"
    
    EXAMPLE: 
    """
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=0.9,
            do_sample=True,
            top_p=0.92,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after the prompt)
    generated_part = generated_text[len(prompt):]
    
    # Split the text into individual examples
    examples = re.findall(r'EXAMPLE:\s*(.*?)(?=EXAMPLE:|$)', generated_part, re.DOTALL)
    
    # Clean up each example
    cleaned_examples = [example.strip() for example in examples if example.strip()]
    
    # Filter out any examples that are too long (more than ~280 chars - Twitter-like length)
    social_media_examples = [ex for ex in cleaned_examples if len(ex) <= 280]
    
    # If we got fewer examples than requested, run again to get more
    if len(social_media_examples) < num_examples and len(social_media_examples) > 0:
        # Only request the remaining number needed
        remaining = num_examples - len(social_media_examples)
        more_examples = generate_hate_speech_batch(topic, remaining, max_length)
        social_media_examples.extend(more_examples)
    
    return social_media_examples[:num_examples]  # Limit to requested number

def save_generated_examples_to_file(topic_list, examples_per_topic=10, output_file="generated_hate_speech.jsonl"):
    """
    Generate hate speech examples for multiple topics and save to a file
    
    Parameters:
    - topic_list: List of topics to generate content for
    - examples_per_topic: Number of examples to generate per topic
    - output_file: File to save the results
    """
    import json
    from datetime import datetime
    
    results = []
    
    for topic in topic_list:
        print(f"Generating examples for topic: {topic}")
        examples = generate_hate_speech_batch(topic, examples_per_topic)
        
        for example in examples:
            entry = {
                "topic": topic,
                "content": example,
                "source": "synthetic",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "model": model_name,
                    "content_type": "social_media_comment"
                }
            }
            results.append(entry)
            
            # Write each entry immediately to avoid losing data if process is interrupted
            with open(output_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
                
        print(f"Generated {len(examples)} examples for {topic}")
    
    print(f"Completed generating {len(results)} examples across {len(topic_list)} topics")
    return results

# Example usage:
#topics = ["immigration", "religion", "politics", "gender", "race"]
#examples = save_generated_examples_to_file(topics, examples_per_topic=15)