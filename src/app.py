import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
import faiss

# Define paths (make sure these match your Colab notebook)
PROJECT_ROOT = 'ecommerce-support-chatbot'
VECTORSTORE_PATH = os.path.join(PROJECT_ROOT, 'vectorstore')
INDEX_PATH = os.path.join(VECTORSTORE_PATH, "faiss_index")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

# Load the LLM
@st.cache_resource
def load_llm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=f'{PROJECT_ROOT}/model/', token=HUGGINGFACE_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",
                                                 torch_dtype=torch.float16,
                                                 load_in_4bit=True,
                                                 use_flash_attention_2=False,
                                                 bnb_4bit_compute_dtype=torch.float16,
                                                 bnb_4bit_quant_type="nf4",
                                                 cache_dir=f'{PROJECT_ROOT}/model/',
                                                 token=HUGGINGFACE_TOKEN)

    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    use_cache=True,
                    device_map="auto",
                    max_new_tokens=512,
                    min_new_tokens=50,
                    top_k=40,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id)

    return pipe

# Load embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load the vector store
@st.cache_resource
def load_vectorstore(embeddings):
    index = faiss.read_index(os.path.join(VECTORSTORE_PATH, "index"))

    # Convert the index to a GPU index if CUDA is available
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # Load the FAISS vector store from disk
    db = FAISS.load_local(VECTORSTORE_PATH, embeddings)

    # Set the loaded index for querying
    db.index = index

    return db

# Create the RAG chain
@st.cache_resource
def create_rag_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever()
    template = """
    You are a helpful customer support chatbot for an e-commerce store.
    Use the following context to answer the question:
    {context}

    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# Streamlit app
st.title("E-commerce Customer Support Chatbot")
st.write("Ask me questions about products, shipping, returns, or anything else related to our store.")

llm_pipeline = load_llm()
llm = pipeline(llm_pipeline)
embeddings = load_embeddings()
vectorstore = load_vectorstore(embeddings)
rag_chain = create_rag_chain(llm, vectorstore)

# Get user input
query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query and rag_chain:
        # Get the response from the RAG chain
        with st.spinner('Processing your request...'):
            try:
                result = rag_chain(query)
                # Display the answer
                st.subheader("Answer:")
                st.write(result['result'])
            except Exception as e:
                st.error(f"An error occurred: {e}")

    else:
        st.warning("Please enter a question.")
