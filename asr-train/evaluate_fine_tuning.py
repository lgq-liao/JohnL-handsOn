# Evaluate Fine-Tuning Performance Notebook

import pandas as pd
import matplotlib.pyplot as plt
from evaluate import load
import re

def wer_evaluation(csv_path="./cv-valid-test/cv-valid-test.csv", 
                   ground_txt_col="text",
                   transcription_col="generated_text", 
                   show_plt=False):
    # Step 1: Load the Dataset

    data = pd.read_csv(csv_path)

    # Ensure required columns exist
    if not {ground_txt_col, transcription_col}.issubset(data.columns):
        raise ValueError("Dataset must contain 'text' and 'generated_text' columns.")

    # Step 2: Normalize Text
    chars_to_ignore_regex = r'[\,\?\.\!\-\;\:"]'  # Characters to ignore during WER computation

    def normalize_text(text):
        if isinstance(text, str):
            return re.sub(chars_to_ignore_regex, '', text).lower().strip()
        return ""


    data[ground_txt_col] = data[ground_txt_col].apply(normalize_text)
    data[transcription_col] = data[transcription_col].apply(normalize_text)

    # Step 3: Compute WER
    wer_metric = load("wer")
    wer_score = wer_metric.compute(predictions=data[transcription_col].tolist(), references=data[ground_txt_col].tolist())

    # Log the performance
    if transcription_col =="generated_text":
        print("Orignal model perfromance:")
    else:
        print("Fine-tuned model performance:")

    print(f"\tOverall WER: {wer_score:.2%}")

    # # Save results to a file
    # with open("performance_log.txt", "w") as log_file:
    #     log_file.write(f"Overall WER: {wer_score:.2%}\n")

    if show_plt:
        # Step 4: Visualization
        # Compare length of generated text vs ground truth
        lengths = data.apply(lambda row: len(row[transcription_col]) - len(row[ground_txt_col]), axis=1)

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
        plt.scatter(data[ground_txt_col].apply(len), data[transcription_col].apply(len), alpha=0.5, label="Data Points")
        plt.plot([0, max(data[ground_txt_col].apply(len))], [0, max(data[ground_txt_col].apply(len))], color="red", linestyle="dashed", label="y=x")
        plt.xlabel("Ground Truth Text Length")
        plt.ylabel("Generated Text Length")
        plt.title("Scatter Plot of Text Lengths")
        plt.legend()
        plt.grid(alpha=0.75)
        plt.show()

if __name__ == '__main__':
    wer_evaluation()