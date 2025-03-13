import os
import json
import glob
import spacy
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import pickle
import argparse

class RedditNERPipeline:
    def __init__(self, data_dir, output_dir, model_name="en_core_web_lg"):
        """
        Initialize the NER pipeline.
        
        Args:
            data_dir (str): Directory containing the JSONL files
            output_dir (str): Directory to save the output files
            model_name (str): Name of the spaCy model to use
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load spaCy model
        print(f"Loading spaCy model: {model_name}")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Downloading...")
            os.system(f"python -m spacy download {model_name}")
            self.nlp = spacy.load(model_name)
        
        # Configure the pipeline for better performance
        self.nlp.add_pipe("merge_entities")
        # Disable unnecessary components for better performance
        disabled_pipes = ["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]
        self.nlp.disable_pipes(*[p for p in disabled_pipes if p in self.nlp.pipe_names])
        
        # Entity index: maps entity text to list of post IDs
        self.entity_index = defaultdict(set)
        # Post data: maps post ID to post data
        self.post_data = {}
        
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
                    
                    # Process text with spaCy
                    doc = self.nlp(post['text'])
                    
                    # Extract entities
                    for ent in doc.ents:
                        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'FAC']:
                            # Normalize entity text (lowercase)
                            entity_text = ent.text.strip()
                            # Add post ID to entity index
                            self.entity_index[entity_text].add(post_id)
                            # Also add the entity type to help with filtering later
                            self.entity_index[f"{ent.label_}:{entity_text}"].add(post_id)
                            
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
    
    def save_index(self):
        """Save the entity index and post data in JSON format."""
        # Save combined entity data in JSON format
        output_file = os.path.join(self.output_dir, 'entity_data.json')
        
        # Create output structure
        entity_data = {}
        
        # Process each entity (excluding type prefixes)
        for entity, post_ids in self.entity_index.items():
            if ':' not in entity:  # Skip type-prefixed entries
                # Get entity types
                entity_types = set()
                for key in self.entity_index.keys():
                    if ':' in key:
                        type_part, entity_part = key.split(':', 1)
                        if entity_part == entity:
                            entity_types.add(type_part)
                
                # Create entity entry
                entity_data[entity] = {
                    'types': list(entity_types),
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
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(entity_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved entity data to {output_file}")
    
    def run(self):
        """Run the full pipeline."""
        self.process_all_files()
        self.save_index()


class EntitySearch:
    def __init__(self, index_dir):
        """
        Initialize the entity search.
        
        Args:
            index_dir (str): Directory containing the entity index and post data
        """
        self.index_dir = index_dir
        
        # Load entity index
        with open(os.path.join(index_dir, 'entity_index.pkl'), 'rb') as f:
            self.entity_index = pickle.load(f)
        
        # Load post data
        with open(os.path.join(index_dir, 'post_data.pkl'), 'rb') as f:
            self.post_data = pickle.load(f)
        
        print(f"Loaded {len(self.entity_index)} entities and {len(self.post_data)} posts")
    
    def search(self, query, entity_type=None, limit=10):
        """
        Search for posts mentioning the query entity.
        
        Args:
            query (str): Entity to search for
            entity_type (str, optional): Filter by entity type (PERSON, ORG, etc.)
            limit (int, optional): Maximum number of results to return
            
        Returns:
            list: List of posts mentioning the entity
        """
        # Normalize query
        query = query.strip()
        
        # Find matching entities
        matching_entities = []
        for entity in self.entity_index.keys():
            # Skip entity type prefixes
            if ':' in entity:
                continue
                
            if query.lower() in entity.lower():
                # If entity type is specified, check if this entity has that type
                if entity_type:
                    type_entity = f"{entity_type}:{entity}"
                    if type_entity in self.entity_index:
                        matching_entities.append(entity)
                else:
                    matching_entities.append(entity)
        
        # Get post IDs for matching entities
        post_ids = set()
        for entity in matching_entities:
            post_ids.update(self.entity_index[entity])
        
        # Get post data
        results = []
        for post_id in post_ids:
            if post_id in self.post_data:
                results.append(self.post_data[post_id])
        
        # Sort by score
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return results[:limit]
    
    def get_entity_types(self):
        """Get all entity types in the index."""
        entity_types = set()
        for entity in self.entity_index.keys():
            if ':' in entity:
                entity_type = entity.split(':', 1)[0]
                entity_types.add(entity_type)
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
        
        for entity, post_ids in self.entity_index.items():
            # Skip entity type prefixes
            if ':' in entity:
                continue
                
            # If entity type is specified, check if this entity has that type
            if entity_type:
                type_entity = f"{entity_type}:{entity}"
                if type_entity not in self.entity_index:
                    continue
            
            entity_counts.append((entity, len(post_ids)))
        
        # Sort by count
        entity_counts.sort(key=lambda x: x[1], reverse=True)
        
        return entity_counts[:limit]


def main():
    parser = argparse.ArgumentParser(description='Reddit NER Pipeline')
    parser.add_argument('--data_dir', type=str, default='2025', help='Directory containing the JSONL files')
    parser.add_argument('--output_dir', type=str, default='spacy_ner_output', help='Directory to save the output files')
    parser.add_argument('--model', type=str, default='en_core_web_lg', help='spaCy model to use')
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
                print(f"   Score: {post.get('score', 0)}")
                print(f"   Link: {post.get('permalink', 'No link')}")
                print(f"   Text: {post.get('text', 'No text')[:100]}...")
        else:
            # Show entity types
            entity_types = search.get_entity_types()
            print(f"Available entity types: {', '.join(entity_types)}")
    else:
        # Pipeline mode
        pipeline = RedditNERPipeline(args.data_dir, args.output_dir, model_name=args.model)
        pipeline.run()


if __name__ == "__main__":
    main() 