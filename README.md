# ecommerce-support-chatbot
A chatbot that can answer customer questions about products, shipping, returns, and other common support topics for a hypothetical e-commerce store.

# --- Cell 7: Create the RAG Chain ---
retriever = db.as_retriever()
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)
