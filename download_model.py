#!/usr/bin/env python3
"""
Script to download a Hugging Face model locally.
This can be useful if you're having network issues when running the NER pipeline.
"""

import os
import argparse
from huggingface_hub import snapshot_download
import time

def download_model(model_name, output_dir, timeout=300, max_retries=5):
    """
    Download a model from Hugging Face to a local directory.
    
    Args:
        model_name (str): Name of the Hugging Face model
        output_dir (str): Directory to save the model
        timeout (int): Timeout in seconds for each download attempt
        max_retries (int): Maximum number of retries
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading model {model_name} to {output_dir}")
    print(f"This may take a while depending on your internet connection...")
    
    for attempt in range(max_retries):
        try:
            # Set environment variables to increase timeout
            os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = str(timeout)
            
            # Download the model
            snapshot_download(
                repo_id=model_name,
                local_dir=output_dir,
                local_dir_use_symlinks=False,
                revision="main"
            )
            
            print(f"Successfully downloaded model to {output_dir}")
            return True
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Failed to download the model after multiple attempts.")
                print("Please check your internet connection and try again.")
                return False

def main():
    parser = argparse.ArgumentParser(description='Download a Hugging Face model locally')
    parser.add_argument('--model', type=str, default='mbastardi24/roBERTa-finetuned-wnut2017', 
                        help='Hugging Face model to download')
    parser.add_argument('--output_dir', type=str, default='./local_model', 
                        help='Directory to save the model')
    parser.add_argument('--timeout', type=int, default=300, 
                        help='Timeout in seconds for each download attempt')
    parser.add_argument('--max_retries', type=int, default=5, 
                        help='Maximum number of retries')
    
    args = parser.parse_args()
    
    download_model(args.model, args.output_dir, args.timeout, args.max_retries)

if __name__ == "__main__":
    main() 