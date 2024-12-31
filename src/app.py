import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
import torch
import os
import faiss

PROJECT_ROOT = '.'  # or your root path
VECTORSTORE_PATH = os.path.join(PROJECT_ROOT, 'vectorstore')
INDEX_PATH = os.path.join(VECTORSTORE_PATH, "faiss_index")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

@st.cache_resource
def load_llm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=f'{PROJECT_ROOT}/model/',
        token=HUGGINGFACE_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True,
        use_flash_attention_2=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        cache_dir=f'{PROJECT_ROOT}/model/',
        token=HUGGINGFACE_TOKEN
    )

    # Lowering max_new_tokens helps keep answers short and to the point
    # Using a lower temperature reduces 'creative' tangents
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_new_tokens=128,
        min_new_tokens=20,
        temperature=0.2,
        top_k=40,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return llm

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def load_vectorstore(_embeddings):
    # Load the FAISS index from disk
    index = faiss.read_index(os.path.join(VECTORSTORE_PATH, "index.faiss"))

    # Load the FAISS vector store from disk
    db = FAISS.load_local(
        VECTORSTORE_PATH,
        _embeddings,
        allow_dangerous_deserialization=True
    )

    # Assign the loaded index for querying
    db.index = index
    return db

@st.cache_resource
def create_rag_chain(_llm, _vectorstore):
    # Create a retriever that fetches top 3 chunks
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})

    # Updated prompt: added a final line asking not to include new questions
    template = """
You are a helpful and enthusiastic customer support chatbot for an e-commerce store.
Use the following context to answer the question.
If the answer is not in the provided context, say that you don't know.
Keep your answer short, polite, and relevant, and format it for readability.
Do not ask any additional questions in your response.

Context:
{context}

Question: {question}
Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

st.title("E-commerce Customer Support Chatbot")
st.write("Ask me questions about products, shipping, returns, or anything else related to our store.")

llm = load_llm()
embeddings = load_embeddings()
vectorstore = load_vectorstore(embeddings)
rag_chain = create_rag_chain(llm, vectorstore)

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query and rag_chain:
        with st.spinner('Processing your request...'):
            try:
                result = rag_chain.invoke(query)
                if isinstance(result, dict) and 'result' in result:
                    answer = result['result']
                else:
                    answer = "Could not generate an answer."

                # 1. Strip leftover special tokens
                for token in ["<s>", "</s>", "[INST]", "[/INST]"]:
                    answer = answer.replace(token, "")

                # 2. If the model tries to add a new question, remove that part
                #    (i.e., anything after "Question:" or "Q:" – you can adapt if needed)
                lower_answer = answer.lower()
                if "question:" in lower_answer:
                    # keep only text before the first 'Question:' (case-insensitive)
                    idx = lower_answer.find("question:")
                    answer = answer[:idx].strip()

                st.subheader("Answer:")
                st.write(answer.strip())

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
