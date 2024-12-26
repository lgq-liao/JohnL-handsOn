# Install required libraries
# pip install transformers pandas scikit-learn

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Load model and tokenizer
model_name = "hkunlp/instructor-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

DEBUG = False
THESHOLD=0.4  # Set threshold for similarity

# Load the CSV file
if DEBUG:
    input_file = "detected.txt"
    output_file = "./debug-detected.csv"
else:
    input_file = "../asr-train/cv-valid-dev_ft.csv"
    output_file = "./cv-valid-dev.csv"

df = pd.read_csv(input_file)

# Hot words and embedding computation
hot_words = ['be careful', 'destroy', 'stranger']
hot_word_embeddings = []

for word in hot_words:
    # Tokenize input
    inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        # Use the encoder and process input_ids
        outputs = model.encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        hot_word_embeddings.append(outputs.last_hidden_state.mean(dim=1).numpy())

hot_word_embeddings = torch.tensor(hot_word_embeddings)



# Calculate similarity for each transcription
def check_similarity(text):
    # Ensure text is a valid string
    if not isinstance(text, str):
        return False

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        # Use the encoder to generate embeddings for the transcription
        outputs = model.encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        text_embedding = outputs.last_hidden_state.mean(dim=1).numpy()

    # Compute cosine similarity with each hot word embedding
    for index, hot_word_embedding in enumerate(hot_word_embeddings):
        similarity = cosine_similarity(text_embedding, hot_word_embedding.numpy().reshape(1, -1))
        if DEBUG:
            print(text, hot_words[index], similarity)
        if similarity.max() > THESHOLD:  # Set threshold for similarity
            return True
    return False




# Iterate over rows and calculate similarity
df["similarity"] = df["generated_text_ft"].apply(lambda x: check_similarity(x))

# Save the updated CSV
df.to_csv(output_file, index=False)

print(f"Updated file saved as {output_file}")

