# --- Cell 3: Download and Prepare the AmazonQA Dataset ---
import os
import json
import pandas as pd
import requests
from tqdm.auto import tqdm

# Define the data directory
DATA_DIR = '/content/drive/MyDrive/ecommerce-support-chatbot/data'  # Update this

# Create the data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Function to download a file with a progress bar
def download_file(url, dest_path):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(dest_path, 'wb') as f, tqdm(
                desc=dest_path.split('/')[-1],
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

# Function to extract Q&A pairs from AmazonQA dataset
def extract_qa_pairs(filepath):
    qa_pairs = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                question = data['questionText']
                for answer in data['answers']:
                    qa_pairs.append({
                        'question': question,
                        'answer': answer['answerText']
                    })
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
    return qa_pairs

# URLs of the dataset files
dataset_urls = {
    'train': 'https://amazon-qa.s3-us-west-2.amazonaws.com/train-qar.jsonl',
    'validation': 'https://amazon-qa.s3-us-west-2.amazonaws.com/val-qar.jsonl',
    'test': 'https://amazon-qa.s3-us-west-2.amazonaws.com/test-qar.jsonl',
}

all_qa_data = []

# Download and process each dataset file
for split, url in dataset_urls.items():
    print(f"Downloading and processing {split} dataset...")
    dest_file = os.path.join(DATA_DIR, f"{split}-qar.jsonl")
    if download_file(url, dest_file):
        qa_data = extract_qa_pairs(dest_file)
        all_qa_data.extend(qa_data)

# Convert to pandas DataFrame for easier handling
qa_df = pd.DataFrame(all_qa_data)

# Display the first few rows of the DataFrame
print(qa_df.head())
