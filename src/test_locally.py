import os
import argparse

def download_kaggle_dataset(dataset_name, destination_folder):
    """
    Downloads a dataset from Kaggle using the Kaggle API.

    Args:
        dataset_name: The name of the Kaggle dataset (e.g., "paultimothymooney/chest-xray-pneumonia").
        destination_folder: The folder where you want to download the dataset.
    """

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Set Kaggle API credentials (make sure you have ~/.kaggle/kaggle.json)
    os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser("~/.kaggle")

    # Download the dataset using the Kaggle API
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_name, path=destination_folder, unzip=True)
        print(f"Dataset '{dataset_name}' downloaded to '{destination_folder}'")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Make sure you have the Kaggle API installed and configured correctly.")
        print("See: https://github.com/Kaggle/kaggle-api for instructions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Kaggle dataset.")
    parser.add_argument("dataset_name", type=str, help="The name of the Kaggle dataset (e.g., paultimothymooney/chest-xray-pneumonia)")
    parser.add_argument("destination_folder", type=str, help="The destination folder for the dataset.")
    args = parser.parse_args()

    download_kaggle_dataset(args.dataset_name, args.destination_folder)
