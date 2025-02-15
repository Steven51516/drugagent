import pandas as pd
import os
import requests

# Set the download directory
download_dir = "../env"


# Define URLs for the datasets
base_url = "https://raw.githubusercontent.com/kexinhuang12345/MolTrans/master/dataset/DAVIS/"
files = ["test.csv", "train.csv", "val.csv"]

# Function to download and save the file
def download_and_save_file(file_name, base_url, save_dir):
    url = f"{base_url}{file_name}"
    save_path = os.path.join(save_dir, file_name)
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(save_path, "wb") as file:
            file.write(response.content)
        print(f"Downloaded and saved: {file_name}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {file_name}: {e}")

# Download each file and save it locally
for file in files:
    download_and_save_file(file, base_url, download_dir)

# Function to process the datasets
def process_dataset(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Select and rename columns
    df = df[["SMILES", "Target Sequence", "Label"]]
    df = df.rename(columns={"SMILES": "Drug", "Target Sequence": "Protein", "Label": "Y"})

    return df

# Process and save the datasets
for file in files:
    input_path = os.path.join(download_dir, file)
    output_path = os.path.join(download_dir, file)
    
    processed_df = process_dataset(input_path)
    processed_df.to_csv(output_path, index=False)
    print(f"Processed and saved: {file}")
