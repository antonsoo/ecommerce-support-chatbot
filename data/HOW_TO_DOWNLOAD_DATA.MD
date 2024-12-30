## Dataset

This project uses the AmazonQA dataset, a collection of question-answer pairs based on Amazon product data.

**Data Download:**

The dataset is not included in this repository. You need to download it yourself from the original source:

1. **Go to:** [https://github.com/amazon-research/open-question-answering-data/tree/main/AmazonQA](https://github.com/amazon-research/open-question-answering-data/tree/main/AmazonQA)
2. **Download:** Download the following files:
   - `train-qar.jsonl`
   - `val-qar.jsonl`
3. **Place the Files:** Put the downloaded files into the `data/AmazonQA` directory of this project.

**Alternative: Use the Provided Script:**

You can also use the `download_data.py` script (in the `src` directory) to download the data:

```bash
cd src
python download_data.py
