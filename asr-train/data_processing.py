
import os
import re
import pandas as pd
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm
from datasets import Dataset
import gc  # Import garbage collection for resource cleanup

# Directories for caching datasets
train_dataset_cache_dir = "./cached_dataset/train"
val_dataset_cache_dir = "./cached_dataset/val"

import gc
import os
import re
import pandas as pd
import torchaudio
from datasets import Dataset
from tqdm import tqdm
from torchaudio.transforms import Resample

def save_dataset_chunk(dataset, output_dir, dataset_type, chunk_index):
    """
    Save a single dataset chunk to disk.

    Parameters:
    - dataset: The dataset to save.
    - output_dir: Base directory for saving datasets.
    - dataset_type: "train" or "val", to organize saved data into subdirectories.
    - chunk_index: Index of the chunk being saved.

    Returns:
    - None: Saves the dataset chunk to disk.
    """
    chunk_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_path = os.path.join(chunk_dir, f"{dataset_type}_chunk_{chunk_index}")
    dataset.save_to_disk(chunk_path)
    print(f"Saved {dataset_type} chunk {chunk_index} to {chunk_path}")


def cache_dataset(
    csv_name="cv-valid-train.csv",
    base_audio_path="cv-valid-train/",
    output_dir="./cached_dataset",
    debug_limit=None,
    train_frac=0.7,
    target_sample_rate=16000,
    min_input_length=1.0,
    max_input_length=15.0,
    num_workers=4,
    chunk_size=10000  # Number of samples per chunk for processing and saving
):
    """
    Caches datasets for training and evaluation with chunked processing and saving.

    Parameters:
    - csv_name: Name of the CSV file containing metadata.
    - base_audio_path: Base directory for audio files.
    - output_dir: Base directory for saving processed datasets.
    - debug_limit: Limit on the number of rows to process for debugging.
    - train_frac: Fraction of data to use for training.
    - target_sample_rate: Target sampling rate for audio resampling.
    - min_input_length: Minimum allowable duration (in seconds) for audio samples.
    - max_input_length: Maximum allowable duration (in seconds) for audio samples.
    - num_workers: Number of workers for filtering dataset.
    - chunk_size: Number of samples per chunk for processing and saving.

    Returns:
    - None: The processed datasets are saved to disk.
    """
    # Characters to ignore in text normalization
    chars_to_ignore_regex = r'[\,\?\.\!\-\;\:\"]'

    # Text normalization function
    def normalize_text(batch):
        batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower().strip() + " "
        return batch

    # Load the dataset
    print(f"Loading dataset from {csv_name}...")
    df = pd.read_csv(csv_name)

    # Apply debug limit if provided
    if debug_limit is not None:
        df = df.head(debug_limit)

    # Add complete path to the filename column
    df['path'] = df['filename'].apply(lambda x: os.path.join(base_audio_path, x))

    # Filter out rows with empty or NaN text
    df = df[df['text'].notna() & (df['text'].str.strip() != "")]

    # Normalize text
    df = df.apply(normalize_text, axis=1)

    # Split into train and validation sets
    train_df = df.sample(frac=train_frac, random_state=42)
    val_df = df.drop(train_df.index)

    # Preprocess function
    def preprocess_data(row):
        try:
            # Load audio file
            waveform, sr = torchaudio.load(row["path"])

            # Resample to target sampling rate
            resampler = Resample(orig_freq=sr, new_freq=target_sample_rate)
            resampled_waveform = resampler(waveform).squeeze(0).numpy()

            # Calculate duration
            duration = resampled_waveform.shape[0] / target_sample_rate

            return {"speech": resampled_waveform, "text": row["text"], "duration": duration}
        except Exception as e:
            # Log errors and skip problematic files
            print(f"Error processing file {row['path']}: {e}")
            return None

    # Process and save data in chunks
    def process_and_save_in_chunks(df, dataset_type):
        print(f"Processing and saving {dataset_type} data...")
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i + chunk_size]
            print(f"Processing {dataset_type} chunk {i // chunk_size + 1}...")
            chunk_data = [
                result for result in tqdm(map(preprocess_data, chunk_df.to_dict("records"))) if result is not None
            ]

            dataset = Dataset.from_pandas(pd.DataFrame(chunk_data))

            # Sanitize the dataset
            def is_valid_sample(example):
                return len(example["text"]) > 0 and min_input_length <= example["duration"] <= max_input_length

            dataset = dataset.filter(is_valid_sample, num_proc=num_workers)

            # Save the chunk
            save_dataset_chunk(dataset, output_dir, dataset_type, i // chunk_size + 1)

            # Free resources
            del dataset, chunk_data
            gc.collect()

    # Process and save validation dataset
    process_and_save_in_chunks(val_df, "val")

    # Process and save train dataset
    process_and_save_in_chunks(train_df, "train")

    print("Dataset processing completed.")

cache_dataset(chunk_size=10000)