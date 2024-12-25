import pandas as pd
import torchaudio
from datasets import Dataset, load_from_disk, concatenate_datasets

from transformers import TrainerCallback

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Trainer,
    TrainingArguments
)

import evaluate
import numpy as np
import matplotlib.pyplot as plt
import random

from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
import torch

from torchaudio.transforms import Resample
from tqdm import tqdm
import re
import os
import gc  # Import garbage collection for resource cleanup

DEBUG=False # trancate the dataset for debugging
LIMIT_TRAIN_DATASET = False # Limit dataset size to 10% of its length for demo purposes

base_dir = "./cached_dataset"
# Initialize the processor and tokenizer
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

def cache_dataset(
    csv_name="cv-valid-train.csv",
    base_audio_path="cv-valid-train/",
    output_dir=base_dir,
    debug_limit=None,
    train_frac=0.7,
    target_sample_rate=16000,
    min_input_length=1.0,
    max_input_length=15.0,
    num_workers=1, # 4, change it back to 1 due to memory limitation
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
        batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).upper().strip()
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

    # Prepare batch function
    def prepare_batch(batch):
        inputs = processor(
            batch["speech"], 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        labels = processor(
            text=batch["text"], 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).input_ids

        # Replace padding with -100 for ignored tokens in loss calculation
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return inputs


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
            # print(dataset[0].keys())
            # print("Text of the first sample: ", dataset[0]['text'])

            # Sanitize the dataset
            def is_valid_sample(example):
                return len(example["text"]) > 0 and min_input_length <= example["duration"] <= max_input_length

            dataset = dataset.filter(is_valid_sample, num_proc=num_workers)

            # Map the function to datasets
            print("Mapping prepare_batch to datasets...")
            dataset = dataset.map(prepare_batch, batched=True)
            # print('Tokenized text: ', dataset[0]['labels'])

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


def concatenate_cached_datasets(base_dir, dataset_type):
    """
    Concatenates all dataset parts from a specified type ('train' or 'val') into a single dataset.

    Parameters:
    - base_dir: The base directory where cached dataset parts are stored.
    - dataset_type: The type of dataset to concatenate ('train' or 'val').

    Returns:
    - concatenated_dataset: The concatenated dataset.
    """
    dataset_dir = os.path.join(base_dir, dataset_type)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # List all dataset parts
    dataset_parts = [
        os.path.join(dataset_dir, part) for part in os.listdir(dataset_dir) 
        if os.path.isdir(os.path.join(dataset_dir, part))
    ]

    if not dataset_parts:
        raise ValueError(f"No dataset parts found in {dataset_dir}")

    # Load and concatenate all dataset parts
    datasets = [load_from_disk(part) for part in dataset_parts]
    concatenated_dataset = concatenate_datasets(datasets)

    print(f"Concatenated {len(datasets)} parts for {dataset_type} dataset.")
    return concatenated_dataset


# Load dataset
if not os.path.exists(base_dir):
    if DEBUG:
        cache_dataset(debug_limit=1000, chunk_size=10000)
    else:
        cache_dataset(chunk_size=10000)

    
train_dataset = concatenate_cached_datasets(base_dir, "train")
val_dataset = concatenate_cached_datasets(base_dir, "val")
print("Train dataset length:", len(train_dataset))
print("Validation dataset length:", len(val_dataset))

# Set dataset format for PyTorch
train_dataset.set_format(type="torch", columns=["input_values", "labels"])
val_dataset.set_format(type="torch", columns=["input_values", "labels"])


# Limit dataset size to 10% of its length for demo purposes
if LIMIT_TRAIN_DATASET:
    train_subset_size = int(len(train_dataset) * 0.1)
    val_subset_size = int(len(val_dataset) * 0.1)
    
    train_dataset = train_dataset.select(range(train_subset_size))
    val_dataset = val_dataset.select(range(val_subset_size))

# Initialize model
print("Initializing model...")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
# freeze encoder
model.freeze_feature_encoder()

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate input values and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input values
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Convert label_features into a tensor
        labels = [label["input_ids"] for label in label_features]
        labels_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(label, dtype=torch.long) for label in labels],
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )

        # Replace padding with -100 to ignore in loss calculation
        labels_batch[labels_batch == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels_batch

        return batch

# Initialize data collator
data_collator = DataCollatorCTCWithPadding(processor=processor)

# Load evaluation metric
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Replace padding token IDs with -100
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(pred.label_ids, skip_special_tokens=True)

    # Filter out empty references dynamically
    valid_indices = [i for i, ref in enumerate(label_str) if ref.strip()]
    pred_str = [pred_str[i] for i in valid_indices]
    label_str = [label_str[i] for i in valid_indices]

    # Log random predictions for debugging
    if pred_str and label_str:
        random_index = random.randrange(len(pred_str))
        print(f"Sample Prediction[{random_index}]: {pred_str[random_index]}")
        print(f"Sample Reference[{random_index}]: {label_str[random_index]}")
    else:
        raise ValueError("No valid labels to evaluate.")

    wer = metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Setup training arguments
print("Setting up training arguments...")
# Training arguments
training_args = TrainingArguments(
    output_dir="./results",          # Directory to save results
    evaluation_strategy="steps",     # Evaluate at regular intervals
    eval_steps=1000,                  # Evaluate every 100 training steps
    gradient_accumulation_steps=2,
    per_device_train_batch_size=4,   # Adjust as per available GPU memory
    per_device_eval_batch_size=4,    # Adjust as per available GPU memory
    learning_rate=2e-5,              # Learning rate
    num_train_epochs=3,              # Total number of epochs
    weight_decay=0.01,               # Weight decay for regularization
    fp16=True,                       # Mixed precision training
    logging_dir="./logs",            # Logging directory
    save_total_limit=2,              # Limit saved checkpoints
    save_steps=5000,                  # Save checkpoint every 500 steps
    logging_steps=500,                # Log training metrics every 50 steps
    push_to_hub=False,
    report_to=None,  # Disable reporting to external services
    local_rank=-1   # Ensures no distributed training is attempted
)

# Log to CSV
log_file = "./training_log.csv"
with open(log_file, "w") as log:
    log.write("epoch,train_loss,eval_loss,eval_wer\n")

def log_callback(trainer_state, train_loss=None, eval_loss=None, eval_wer=None):
    epoch = trainer_state.epoch
    with open(log_file, "a") as log:
        log.write(f"{epoch},{train_loss},{eval_loss},{eval_wer}\n")

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        train_loss = logs.get("loss", None)
        eval_loss = logs.get("eval_loss", None)
        eval_wer = logs.get("eval_wer", None)
        if state.epoch:
            log_callback(state, train_loss, eval_loss, eval_wer)

    def on_train_begin(self, args, state, control, **kwargs):
        print("Training started")

    def on_train_end(self, args, state, control, **kwargs):
        print("Training completed")

    def on_evaluate(self, args, state, control, **kwargs):
        print("Evaluation complete")

# Initialize Trainer
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor,
    compute_metrics=compute_metrics,
    callbacks=[],  # Disable TensorBoardCallback and any default callbacks
)

trainer.add_callback(LoggingCallback())

# Start training
print("Starting training...")
train_result = trainer.train()
trainer.save_model("./wav2vec2-large-960h-cv")

# Save metrics
metrics = train_result.metrics
metrics["train_samples"] = len(train_dataset)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# Generate and save training plot
metrics = trainer.state.log_history
train_loss = [x.get("loss", None) for x in metrics if "loss" in x]
eval_loss = [x.get("eval_loss", None) for x in metrics if "eval_loss" in x]
eval_wer = [x.get("eval_wer", None) for x in metrics if "eval_wer" in x]

steps = range(len(eval_loss))

plt.figure(figsize=(10, 5))
if train_loss:
    plt.plot(range(len(train_loss)), train_loss, label="Training Loss")
if eval_loss:
    plt.plot(steps, eval_loss, label="Validation Loss")
if eval_wer:
    plt.plot(steps, eval_wer, label="Validation WER")
plt.xlabel("Steps")
plt.ylabel("Metrics")
plt.legend()
plt.title("Training and Validation Metrics")
plt.savefig("./training_plot.png")  # Save plot as PNG
plt.show()
