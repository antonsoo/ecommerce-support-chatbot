
## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/antonsoo/ecommerce-support-chatbot/
    cd ecommerce-support-chatbot
    ```

2. **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate  # On Windows
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download the dataset:**

    Follow the instructions in `data/HOW_TO_DOWNLOAD_DATA.md` to download the AmazonQA dataset and place it in the `data/AmazonQA` directory.

## Usage

**To create the vector store:**

1. Make sure you have downloaded the dataset and placed it in the correct directory.
2. Run the training script:

    ```bash
    python src/train.py
    ```

This will create the FAISS vector store from the dataset and save it to the `vectorstore` directory.

**To run the Streamlit app locally:**

1. Ensure that the vector store has been created by running `train.py`.
2. Run the following command:

    ```bash
    streamlit run src/app.py
    ```

    This will start the Streamlit app, and you will be able to access it in your web browser (usually at `http://localhost:8501`).

**To test locally using ngrok (Optional):**

1. Install `ngrok` following instructions [here](https://ngrok.com/download)
2. Update the `NGROK_AUTH_TOKEN` in the `run_ngrok()` function in the `test_locally.py` file.
3. Run the `test_locally.py` script to start Streamlit and ngrok in separate threads:

    ```bash
    python src/test_locally.py
    ```

This will start Streamlit in the background, start `ngrok`, and provide a temporary public URL to access the app. This is useful for testing in restricted environments like Google Colab.

**Deployment:**

The app is deployed on Hugging Face Spaces. To deploy your own version:

1. Create a Hugging Face account and a new Space, selecting "Streamlit" as the SDK.
2. Connect your GitHub repository to the Space.
3. Add your Hugging Face token as a secret named `HUGGINGFACE_TOKEN` in the Space's settings.
4. Hugging Face Spaces will automatically build and deploy your app.

## Model Details

*   **LLM:** `mistralai/Mistral-7B-Instruct-v0.2`
*   **Embedding Model:** `sentence-transformers/all-mpnet-base-v2`
*   **Vector Store:** FAISS

## Ethical Considerations

*   This project uses a publicly available dataset for research and educational purposes.
*   It is important to be aware of potential biases in the data and to use the chatbot responsibly.
*   The chatbot is intended to be a helpful tool, but it should not be considered a replacement for human customer support agents.

## Future Improvements

*   Experiment with different LLMs and embedding models.
*   Fine-tune the LLM on a domain-specific dataset.
*   Incorporate a mechanism for handling out-of-domain or unsupported questions.
*   Improve the user interface and add more interactive features to the Streamlit app.
*   Evaluate the chatbot's performance using more comprehensive metrics and user studies.

## Acknowledgements

*   [Hugging Face](https://huggingface.co/) for the Transformers library, pre-trained models, and the Spaces platform.
*   [Streamlit](https://streamlit.io/) for the web app framework.
*   [LangChain](https://www.langchain.com/) for the RAG implementation.
*   The creators of the AmazonQA dataset.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
