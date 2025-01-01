import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
import torch
import re
import os
import faiss

##############################################################################
# 1) Paths & Setup
##############################################################################
PROJECT_ROOT = '.'  # Adjust if needed
VECTORSTORE_PATH = os.path.join(PROJECT_ROOT, 'vectorstore')
INDEX_PATH = os.path.join(VECTORSTORE_PATH, "faiss_index")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

##############################################################################
# 2) Load the 4-bit Mistral Model & Pipeline (Cached)
##############################################################################
@st.cache_resource
def load_llm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=f"{PROJECT_ROOT}/model/",
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
        cache_dir=f"{PROJECT_ROOT}/model/",
        token=HUGGINGFACE_TOKEN
    )

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        use_cache=True,
        # Slightly higher temp to reduce "I don't know"
        max_new_tokens=80,
        min_new_tokens=20,
        temperature=0.2,
        top_k=40,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return llm

##############################################################################
# 3) Load Embeddings & Vectorstore (Cached)
##############################################################################
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def load_vectorstore(_embeddings):
    index = faiss.read_index(os.path.join(VECTORSTORE_PATH, "index.faiss"))
    db = FAISS.load_local(
        VECTORSTORE_PATH,
        _embeddings,
        allow_dangerous_deserialization=True
    )
    db.index = index
    return db

##############################################################################
# 4) First pass: RetrievalQA (stuff chain)
##############################################################################
def create_stuff_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate(
        template="""
You are a helpful e-commerce customer support bot. 
Use the following context to answer the user's question.
If you aren't fully sure, provide your best estimate or partial answer.

Context:
{context}

Question: {question}
Answer:
""",
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
    return chain

##############################################################################
# 5) Second pass: rewrite into bullet points (up to 3)
##############################################################################
def create_rewrite_chain(llm):
    rewrite_prompt = PromptTemplate(
        template="""
Rewrite the text below into up to 3 concise bullet points. 
Avoid repeating any instructions or disclaimers. 
If you have limited info, give a partial but direct answer.

Raw text:
{answer}
""",
        input_variables=["answer"]
    )
    return LLMChain(llm=llm, prompt=rewrite_prompt)

##############################################################################
# 6) Streamlit App
##############################################################################
st.title("E-commerce Customer Support Chatbot")
st.write("Ask me questions about products, shipping, returns, or anything else related to our store.")

llm = load_llm()
embeddings = load_embeddings()
vectorstore = load_vectorstore(embeddings)

rag_chain = create_stuff_chain(llm, vectorstore)
rewrite_chain = create_rewrite_chain(llm)

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        with st.spinner("Processing your request..."):
            try:
                # 1) Retrieve + answer
                raw_answer = rag_chain.run(query)

                # Remove leftover special tokens
                for token in ["<s>", "</s>", "[INST]", "[/INST]"]:
                    raw_answer = raw_answer.replace(token, "")

                # 2) Rewrite pass
                bullet_answer = rewrite_chain.run({"answer": raw_answer})
                for token in ["<s>", "</s>", "[INST]", "[/INST]"]:
                    bullet_answer = bullet_answer.replace(token, "")

                # ~~~~~ Post-processing ~~~~~
                # Skip lines that mention rewriting or disclaimers
                skip_keywords = [
                    "rewrite the text below",
                    "raw text:",
                    "if you have limited info",
                    "avoid repeating any instructions",
                    "disclaimer",
                ]
                lines = bullet_answer.splitlines()
                cleaned_lines = []
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    lower_line = stripped.lower()
                    if any(kw in lower_line for kw in skip_keywords):
                        continue
                    cleaned_lines.append(stripped)

                # Identify bullet lines
                bullet_pattern = re.compile(r"""^(?:\d+\.\s|[-•]\s?)(.+)$""", re.VERBOSE)
                final_bullets = []
                for line in cleaned_lines:
                    bullet_match = bullet_pattern.match(line)
                    if bullet_match:
                        bullet_text = bullet_match.group(1).strip()
                        # Remove leading "Yes," or "No," if present
                        bullet_text = re.sub(r'^(yes,|no,)\s*', '', bullet_text, flags=re.IGNORECASE).strip()
                        final_bullets.append(bullet_text)

                # Keep up to 3 bullets
                final_bullets = final_bullets[:3]

                # If none found, fallback to the bullet_answer text
                if not final_bullets:
                    fallback = bullet_answer.strip() or raw_answer.strip()
                    final_answer = fallback if fallback else "No bullet points found."
                else:
                    # Re-format each bullet
                    final_answer = ""
                    for i, text in enumerate(final_bullets, start=1):
                        final_answer += f"{i}. {text}\n"
                    final_answer = final_answer.strip()

                st.subheader("Answer:")
                st.write(final_answer)

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
