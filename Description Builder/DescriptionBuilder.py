import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch

# Load the dataset
data = pd.read_csv("ssssampled_Swiggy_1_20.csv", dtype={"cuisine": str, "menu": str, "item": str, "restaurant": str}, low_memory=False)

# Group data by restaurant
grouped_data = data.groupby("restaurant")

# Initialize lists to store restaurant names and descriptions
restaurant_names = []
descriptions = []

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Loop through each restaurant group
for restaurant_name, restaurant_group in grouped_data:
    # Concatenate the text from cuisine, menu, and item columns
    text = restaurant_group["cuisine"].fillna("") + " " + restaurant_group["menu"].fillna("") + " " + restaurant_group["item"].fillna("")
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(text)
    
    # Extract top keywords
    feature_names = tfidf_vectorizer.get_feature_names_out()
    top_keywords = [str(feature_names[idx]) for idx in tfidf_matrix.sum(axis=0).argsort()[0, -10:]]
    
    # Convert keywords to a single string
    keywords_str = ", ".join(top_keywords)
    
    # Tokenize input text
    inputs = tokenizer("The top keywords extracted from the dataset are: " + keywords_str + ". Now generate a descriptive summary based on these keywords.", return_tensors="pt", max_length=512, truncation=True)
    
    # Forward pass through BERT model
    outputs = model(**inputs)
    
    # Get the last hidden state
    last_hidden_state = outputs.last_hidden_state
    
    # Compute the mean of the last hidden state across tokens to get a fixed-size sentence embedding
    sentence_embedding = torch.mean(last_hidden_state, dim=1)
    
    # Convert the sentence embedding tensor to a list
    description_embedding = sentence_embedding.tolist()
    
    # Append restaurant name and description to lists
    restaurant_names.append(restaurant_name)
    descriptions.append(description_embedding)

# Create a DataFrame to store restaurant names and descriptions
df = pd.DataFrame({"Restaurant": restaurant_names, "Description": descriptions})

# Save DataFrame to a CSV file
df.to_csv("restaurant_descriptions.csv", index=False)
