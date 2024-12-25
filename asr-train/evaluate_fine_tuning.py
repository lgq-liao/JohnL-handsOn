# Evaluate Fine-Tuning Performance Notebook

import pandas as pd
import matplotlib.pyplot as plt
from evaluate import load
import re

# Step 1: Load the Dataset
file_path = "./cv-valid-test/cv-valid-test.csv"  
data = pd.read_csv(file_path)

# Ensure required columns exist
if not {"text", "generated_text"}.issubset(data.columns):
    raise ValueError("Dataset must contain 'text' and 'generated_text' columns.")

# Step 2: Normalize Text
chars_to_ignore_regex = r'[\,\?\.\!\-\;\:"]'  # Characters to ignore during WER computation

def normalize_text(text):
    if isinstance(text, str):
        return re.sub(chars_to_ignore_regex, '', text).lower().strip()
    return ""


data["text"] = data["text"].apply(normalize_text)
data["generated_text"] = data["generated_text"].apply(normalize_text)

# Step 3: Compute WER
wer_metric = load("wer")
wer_score = wer_metric.compute(predictions=data["generated_text"].tolist(), references=data["text"].tolist())

# Log the performance
print(f"Overall WER: {wer_score:.2%}")

# Save results to a file
with open("performance_log.txt", "w") as log_file:
    log_file.write(f"Overall WER: {wer_score:.2%}\n")

# Step 4: Visualization
# Compare length of generated text vs ground truth
lengths = data.apply(lambda row: len(row["generated_text"]) - len(row["text"]), axis=1)

plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=20, alpha=0.7, label="Length Difference (Generated - Ground Truth)")
plt.axvline(0, color='r', linestyle='dashed', linewidth=1, label="Zero Difference")
plt.xlabel("Length Difference")
plt.ylabel("Frequency")
plt.title("Distribution of Length Differences Between Generated and Ground Truth Texts")
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()

# Scatter plot of text lengths
plt.figure(figsize=(10, 6))
plt.scatter(data["text"].apply(len), data["generated_text"].apply(len), alpha=0.5, label="Data Points")
plt.plot([0, max(data["text"].apply(len))], [0, max(data["text"].apply(len))], color="red", linestyle="dashed", label="y=x")
plt.xlabel("Ground Truth Text Length")
plt.ylabel("Generated Text Length")
plt.title("Scatter Plot of Text Lengths")
plt.legend()
plt.grid(alpha=0.75)
plt.show()