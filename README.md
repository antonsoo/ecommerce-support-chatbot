# E-commerce Customer Support Chatbot with Retrieval-Augmented Generation (RAG)

This project implements an intelligent customer support chatbot for an e-commerce store using Retrieval-Augmented Generation (RAG). The chatbot leverages a pre-trained Large Language Model (LLM) combined with a vector database of product information and frequently asked questions (FAQs) to provide accurate and helpful answers to user queries.

## Project Description

The chatbot is designed to enhance the customer support experience by providing instant answers to common questions about products, shipping, returns, and other store policies. It utilizes the following core components:

*   **Retrieval-Augmented Generation (RAG):** Combines the power of a pre-trained LLM with a specialized knowledge base to retrieve relevant information and generate contextually appropriate responses.
*   **Vector Database (FAISS):** Stores embeddings of product information and FAQs, enabling efficient similarity search to find the most relevant context for a given query.
*   **Large Language Model (LLM):** Uses the Mistral-7B-Instruct-v0.2 model from Hugging Face Transformers to generate human-like responses based on the retrieved context and the user's question.
*   **Sentence Transformers:** Employs the `sentence-transformers/all-mpnet-base-v2` model to create high-quality sentence embeddings for efficient information retrieval.
*   **Streamlit:**  A user-friendly web application framework for building and deploying the chatbot interface.
*   **Hugging Face Spaces:** The chatbot is deployed on Hugging Face Spaces for easy accessibility and demonstration.

## Demo

Try out the live chatbot demo here: https://huggingface.co/spaces/antonsoloviev/ecommerce-support-chatbot

If you would like to try the demo, please ask me to re-enable the computing resources on the space hosting the demo.

## Dataset

This project utilizes the **AmazonQA dataset**, a large-scale collection of question-answer pairs related to Amazon products.

**Dataset Source:** [https://github.com/amazon-research/open-question-answering-data/tree/main/AmazonQA](https://github.com/amazon-research/open-question-answering-data/tree/main/AmazonQA)

**Download Instructions:**

The dataset is not included in this repository. You need to download it yourself using the provided script. Please follow these steps:

1. **Install necessary libraries:**
   ```bash
   pip install -r requirements.txt

2. **Run the download script:**
 ```bash
   python src/download_data.py
 ```
This will download the AmazonQA dataset to the data/AmazonQA directory.

--------------------------------

## Project Structure

```
ecommerce-support-chatbot/
├── data/
│   └── AmazonQA/ (Place the downloaded AmazonQA dataset files here)
│   └── faq/ (Optional: Add any additional FAQ documents here)
│   └── HOW_TO_DOWNLOAD_DATA.md (Instructions for downloading the dataset)
├── model/ (Directory for storing the downloaded LLM and sentence-transformer model)
├── vectorstore/ (Directory for storing the FAISS index)
├── notebooks/
│   └── chatbot_development.ipynb (Colab notebook for development and experimentation)
├── src/
│   ├── app.py (Streamlit application script)
│   ├── train.py (Script for creating the vector store)
│   └── download_data.py (Script to download the AmazonQA dataset)
├── README.md
├── requirements.txt
└── .gitignore
```

--------------------------------

## Installation

1. **Install necessary libraries:**
```bash
   git clone https://github.com/antonsoo/ecommerce-support-chatbot
   cd ecommerce-support-chatbot
```

2. **Create a virtual environment (recommended):**
```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Linux/macOS
   .venv\Scripts\activate  # On Windows
```

3. **Install the required packages:**
```bash
   pip install -r requirements.txt
```

4. **Download the dataset:**
Follow the instructions in `data/HOW_TO_DOWNLOAD_DATA.md` to download the AmazonQA dataset and place it in the `data/AmazonQA` directory.

--------------------------------

## Usage

To create the vector store:

1. Make sure you have downloaded the dataset and placed it in the correct directory.

2. Run the training script:
```bash
   streamlit run src/app.py
```
This will provide a temporary public URL to access your app, which can be useful for testing in restricted environments like Google Colab.

--------------------------------

## Deployment:

The app is deployed on Hugging Face Spaces. To deploy your own version:

1. Create a Hugging Face account and a new Space, selecting "Streamlit" as the SDK.

2. Connect your GitHub repository to the Space.

3. Add your Hugging Face token as a secret named HUGGINGFACE_TOKEN in the Space's settings.

4. Hugging Face Spaces will automatically build and deploy your app.

--------------------------------

## Model Details
*   **LLM**: mistralai/Mistral-7B-Instruct-v0.2
*   **Embedding Model**: sentence-transformers/all-mpnet-base-v2
*   **Vector Store**: FAISS

--------------------------------

## Contributing

Contributions to this project are welcome! Feel free to submit pull requests or open issues to suggest improvements or report bugs.
License

This project is licensed under the MIT License - see the LICENSE file for details.

--------------------------------

## Contact

Anton Soloviev - UpWork: https://www.upwork.com/freelancers/~01b9d171164a005062

--------------------------------

## Acknowledgements

*   Hugging Face for the Transformers library, pre-trained models, and the Spaces platform.
*   Streamlit for the easy-to-use web app framework.
*   LangChain for the RAG implementation.
*   The creators of the AmazonQA dataset.
