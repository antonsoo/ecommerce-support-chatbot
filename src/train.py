import os
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm
import torch
import json

# Configuration
PROJECT_ROOT = '/content/drive/MyDrive/ecommerce-support-chatbot'  # Update this!
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
FAQ_DIR = os.path.join(DATA_DIR, 'faq')
VECTORSTORE_PATH = os.path.join(PROJECT_ROOT, 'vectorstore')
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
BATCH_SIZE = 1000

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

def load_and_preprocess_data():
    """Loads and preprocesses the AmazonQA dataset."""

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

    # Filter out rows where 'question' or 'answer' is not a string
    qa_df = qa_df.dropna(subset=['question', 'answer'])
    qa_df = qa_df[qa_df.apply(lambda x: isinstance(x['question'], str) and isinstance(x['answer'], str), axis=1)]

    return qa_df

def create_embeddings_and_vectorstore(qa_df, device, cache_dir):
    """Creates embeddings and vector store from the Q&A data."""

    # Use a sentence-transformers model for embeddings
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            cache_folder=cache_dir,
            model_kwargs={'device': device}
        )
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None

    # Create the documents from the DataFrame
    documents = []
    for _, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="Creating documents"):
        doc_content = f"Question: {row['question']}\nAnswer: {row['answer']}"
        metadata = {"source": "AmazonQA"}
        document = Document(page_content=doc_content, metadata=metadata)
        documents.append(document)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # Process text in batches
    texts = []
    for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Splitting documents into chunks"):
        batch = documents[i:i + BATCH_SIZE]
        texts_batch = text_splitter.split_documents(batch)
        texts.extend(texts_batch)

    # Convert text chunks to embeddings
    print("Converting text chunks to embeddings...")
    embedding_vectors = embeddings.embed_documents([text.page_content for text in texts])

    # Ensure embedding vectors are in float32
    embedding_vectors = np.array(embedding_vectors, dtype=np.float32)

    # Create an index using a factory string (this creates an index on CPU first)
    print("Creating FAISS index...")
    embedding_dim = embedding_vectors.shape[1]
    index_cpu = faiss.IndexFlatL2(embedding_dim)  # Example: L2 distance index

    # Move the index to the GPU
    if device == 'cuda':
        print("Moving FAISS index to GPU...")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu) # 0 for the first GPU
    else:
        index = index_cpu

    # Add the embeddings to the index
    index.add(embedding_vectors)

    # Create a FAISS instance for the search index
    db = FAISS.from_texts([t.page_content for t in texts], embeddings)
    db.index = index

    print("FAISS index created.")

    return db

def main():
    """Main function to run the training pipeline."""

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} for embeddings.")

    # Set the cache directory for Hugging Face models
    cache_dir = os.path.join(PROJECT_ROOT, "model")

    # Load and preprocess data
    qa_df = load_and_preprocess_data()

    # Create embeddings and vector store
    db = create_embeddings_and_vectorstore(qa_df, device, cache_dir)

    # Save the vector store
    if db is not None:
        db.save_local(VECTORSTORE_PATH)
        print(f"Vector store saved to {VECTORSTORE_PATH}")

if __name__ == "__main__":
    main()
