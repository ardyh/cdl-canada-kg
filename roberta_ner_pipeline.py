import os
import json
import glob
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import argparse
import time
import requests
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class RobertaNERPipeline:
    def __init__(self, data_dir, output_dir, model_name="mbastardi24/roBERTa-finetuned-wnut2017", local_model_path=None, timeout=60, max_retries=3):
        """
        Initialize the NER pipeline using Hugging Face's roBERTa model.
        
        Args:
            data_dir (str): Directory containing the JSONL files
            output_dir (str): Directory to save the output files
            model_name (str): Name of the Hugging Face model to use
            local_model_path (str, optional): Path to a local model directory
            timeout (int): Timeout in seconds for model download
            max_retries (int): Maximum number of retries for model download
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load Hugging Face model and tokenizer
        print(f"Loading model: {model_name}")
        
        # Set environment variables to increase timeout
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = str(timeout)
        
        # Try to load the model with retries
        self.tokenizer = None
        self.model = None
        
        for attempt in range(max_retries):
            try:
                if local_model_path and os.path.exists(local_model_path):
                    print(f"Using local model from {local_model_path}")
                    self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                    self.model = AutoModelForTokenClassification.from_pretrained(local_model_path)
                else:
                    print(f"Downloading model from Hugging Face (attempt {attempt+1}/{max_retries})")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForTokenClassification.from_pretrained(model_name)
                break
            except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
                print(f"Timeout error: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # Exponential backoff
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Failed to download the model after multiple attempts.")
                    print("Please try one of the following solutions:")
                    print("1. Check your internet connection")
                    print("2. Download the model manually and use the --local_model_path option")
                    print("3. Use a different model with the --model option")
                    raise
        
        if self.tokenizer is None or self.model is None:
            raise ValueError("Failed to load the model and tokenizer")
        
        # Create NER pipeline
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple"  # Merge tokens with same entity
        )
        
        # Entity index: maps entity text to list of post IDs
        self.entity_index = defaultdict(set)
        # Post data: maps post ID to post data
        self.post_data = {}
        
        # Add spaCy model initialization
        print("Loading spaCy model for entity type classification...")
        import spacy
        self.spacy_nlp = spacy.load("en_core_web_lg")
        
    def find_jsonl_files(self):
        """Find all JSONL files in the data directory."""
        pattern = os.path.join(self.data_dir, "**", "*.jsonl")
        pattern2 = os.path.join(self.data_dir, "**", "reddit-*")
        files = glob.glob(pattern, recursive=True) + glob.glob(pattern2, recursive=True)
        print(f"Found {len(files)} JSONL files")
        return files
    
    def process_file(self, file_path):
        """Process a single JSONL file and extract entities."""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # Parse JSON
                    post = json.loads(line.strip())
                    
                    # Skip if no text
                    if 'text' not in post or not post['text']:
                        continue
                    
                    # Get post ID
                    post_id = post.get('submission_id', post.get('name', None))
                    if not post_id:
                        continue
                    
                    # Store post data
                    self.post_data[post_id] = {
                        'text': post['text'],
                        'title': post.get('title', ''),
                        'permalink': post.get('permalink', ''),
                        'created_utc': post.get('created_utc', ''),
                        'score': post.get('score', 0)
                    }
                    
                    # Process text with Hugging Face NER pipeline
                    entities = self.ner_pipeline(post['text'])
                    
                    # Extract entities and get their types using spaCy
                    for entity in entities:
                        entity_text = entity['word'].strip()
                        
                        # Get entity types using spaCy
                        doc = self.spacy_nlp(entity_text)
                        entity_types = set()
                        if doc.ents:
                            for ent in doc.ents:
                                entity_types.add(ent.label_)
                        
                        # Add post ID to entity index along with its types
                        self.entity_index[entity_text].add((post_id, tuple(entity_types)))
                            
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing post: {e}")
    
    def process_all_files(self):
        """Process all JSONL files."""
        files = self.find_jsonl_files()
        for file_path in tqdm(files, desc="Processing files"):
            self.process_file(file_path)
        
        print(f"Extracted {len(self.entity_index)} entities from {len(self.post_data)} posts")
    
    def save_index_as_json(self):
        """Save the entity index and post data as JSON files."""
        # Create a more readable and analyzable structure
        entity_data = {}
        
        # Process entities
        for entity, post_info in self.entity_index.items():
            # Collect all types and post IDs
            all_types = set()
            post_ids = set()
            
            for post_id, types in post_info:
                post_ids.add(post_id)
                all_types.update(types)
            
            # Create entity entry
            entity_data[entity] = {
                'types': list(all_types),  # Convert set to list for JSON serialization
                'post_count': len(post_ids),
                'posts': [
                    {
                        'id': post_id,
                        'title': self.post_data[post_id]['title'],
                        'text': self.post_data[post_id]['text'],
                        'permalink': self.post_data[post_id]['permalink'],
                        'score': self.post_data[post_id]['score'],
                        'created_utc': self.post_data[post_id]['created_utc']
                    }
                    for post_id in post_ids if post_id in self.post_data
                ]
            }
        
        # Save as JSON
        with open(os.path.join(self.output_dir, 'entity_data.json'), 'w', encoding='utf-8') as f:
            json.dump(entity_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved entity data to {os.path.join(self.output_dir, 'entity_data.json')}")
    
    def run(self):
        """Run the full pipeline."""
        self.process_all_files()
        self.save_index_as_json()


class EntitySearch:
    def __init__(self, index_dir):
        """
        Initialize the entity search.
        
        Args:
            index_dir (str): Directory containing the entity data JSON file
        """
        self.index_dir = index_dir
        
        # Load entity data
        with open(os.path.join(index_dir, 'entity_data.json'), 'r', encoding='utf-8') as f:
            self.entity_data = json.load(f)
        
        print(f"Loaded {len(self.entity_data)} entities")
    
    def search(self, query, entity_type=None, limit=10):
        """
        Search for posts mentioning the query entity.
        
        Args:
            query (str): Entity to search for
            entity_type (str, optional): Filter by entity type
            limit (int, optional): Maximum number of results to return
            
        Returns:
            list: List of posts mentioning the entity
        """
        # Normalize query
        query = query.strip().lower()
        
        # Find matching entities
        matching_entities = []
        for entity in self.entity_data.keys():
            if query in entity.lower():
                # If entity type is specified, check if this entity has that type
                if entity_type:
                    if entity_type in self.entity_data[entity]['types']:
                        matching_entities.append(entity)
                else:
                    matching_entities.append(entity)
        
        # Get posts for matching entities
        results = []
        for entity in matching_entities:
            for post in self.entity_data[entity]['posts']:
                results.append({
                    'entity': entity,
                    'title': post['title'],
                    'text': post['text'],
                    'permalink': post['permalink'],
                    'score': post['score'],
                    'created_utc': post['created_utc']
                })
        
        # Sort by score
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return results[:limit]
    
    def get_entity_types(self):
        """Get all entity types in the index."""
        entity_types = set()
        for entity, data in self.entity_data.items():
            entity_types.update(data['types'])
        return sorted(entity_types)
    
    def get_top_entities(self, entity_type=None, limit=20):
        """
        Get the most frequently mentioned entities.
        
        Args:
            entity_type (str, optional): Filter by entity type
            limit (int, optional): Maximum number of entities to return
            
        Returns:
            list: List of (entity, count) tuples
        """
        entity_counts = []
        
        for entity, data in self.entity_data.items():
            # If entity type is specified, check if this entity has that type
            if entity_type and entity_type not in data['types']:
                continue
            
            entity_counts.append((entity, data['post_count']))
        
        # Sort by count
        entity_counts.sort(key=lambda x: x[1], reverse=True)
        
        return entity_counts[:limit]


def main():
    parser = argparse.ArgumentParser(description='Reddit NER Pipeline with roBERTa')
    parser.add_argument('--data_dir', type=str, default='2025', help='Directory containing the JSONL files')
    parser.add_argument('--output_dir', type=str, default='roberta_ner_output', help='Directory to save the output files')
    parser.add_argument('--model', type=str, default='mbastardi24/roBERTa-finetuned-wnut2017', help='Hugging Face model to use')
    parser.add_argument('--local_model_path', type=str, help='Path to a local model directory')
    parser.add_argument('--timeout', type=int, default=120, help='Timeout in seconds for model download')
    parser.add_argument('--max_retries', type=int, default=3, help='Maximum number of retries for model download')
    parser.add_argument('--search', action='store_true', help='Run in search mode')
    parser.add_argument('--query', type=str, help='Entity to search for')
    parser.add_argument('--entity_type', type=str, help='Filter by entity type')
    parser.add_argument('--top_entities', action='store_true', help='Show top entities')
    
    args = parser.parse_args()
    
    if args.search:
        # Search mode
        search = EntitySearch(args.output_dir)
        
        if args.top_entities:
            # Show top entities
            top_entities = search.get_top_entities(entity_type=args.entity_type)
            print(f"Top {len(top_entities)} entities:")
            for entity, count in top_entities:
                print(f"{entity}: {count} posts")
        elif args.query:
            # Search for entity
            results = search.search(args.query, entity_type=args.entity_type)
            print(f"Found {len(results)} posts mentioning '{args.query}':")
            for i, post in enumerate(results):
                print(f"\n{i+1}. {post.get('title', 'No title')}")
                print(f"   Entity: {post.get('entity', 'Unknown')}")
                print(f"   Score: {post.get('score', 0)}")
                print(f"   Link: {post.get('permalink', 'No link')}")
                print(f"   Text: {post.get('text', 'No text')[:100]}...")
        else:
            # Show entity types
            entity_types = search.get_entity_types()
            print(f"Available entity types: {', '.join(entity_types)}")
    else:
        # Pipeline mode
        pipeline = RobertaNERPipeline(
            args.data_dir, 
            args.output_dir, 
            model_name=args.model,
            local_model_path=args.local_model_path,
            timeout=args.timeout,
            max_retries=args.max_retries
        )
        pipeline.run()


if __name__ == "__main__":
    main() 