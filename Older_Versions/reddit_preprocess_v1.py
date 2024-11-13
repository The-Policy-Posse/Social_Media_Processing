# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:45:21 2024


############### IMPORTANT! ###############
## This script must be run from the reddit_numpy_env environment
## Downgrades numpy to 1.24.3
##########################################

@author: dforc
"""

import pandas as pd
from multiprocessing import Pool, cpu_count
import re
import spacy
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Load necessary NLP data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load spaCy model (make sure it's downloaded in the environment)
nlp = spacy.load("en_core_web_sm")  

# Initialize the sentiment analysis pipeline on CPU
sentiment_analyzer = pipeline("sentiment-analysis")  # Using default CPU settings

# Step 1: Light Cleaning Function for Transformer Sentiment Analysis
def light_clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove hashtags and @mentions
    return text.strip()  # Just trim extra spaces

# Step 2: Full Cleaning and Preprocessing for Non-Transformer NLP Tasks
def preprocess_text(text):
    # Lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text))  # Remove punctuation and extra spaces

    # Lemmatize and remove stop words
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stop_words]
    return ' '.join(tokens)

# Step 3: Parallel Processing Function for Preprocessing
def parallel_preprocess(df, text_column='body', num_processes=None):
    # Default to number of cores if not specified
    num_processes = num_processes or cpu_count()

    # Step 1: Apply light cleaning for transformer analysis
    tqdm.pandas(desc="Light Cleaning Text for Transformer")
    df['cleaned_for_transformer'] = df[text_column].progress_apply(light_clean_text)

    # Step 2: Apply full preprocessing for additional NLP tasks
    tqdm.pandas(desc="Full Preprocessing (Tokenize & Lemmatize)")
    with Pool(num_processes) as pool:
        df['preprocessed_text'] = list(tqdm(pool.imap(preprocess_text, df[text_column]), total=len(df)))

    return df

# Step 4: Sentiment Analysis with Transformer Model on Lightly Cleaned Text
def analyze_sentiment(df, text_column='cleaned_for_transformer'):
    tqdm.pandas(desc="Analyzing Sentiment")
    df['sentiment'] = df[text_column].progress_apply(
        lambda x: sentiment_analyzer(x, truncation=True)[0]['label']  # Truncate long sequences
    )
    return df

# Step 5: Visualization for Sentiment Analysis
def plot_sentiment(df):
    sns.countplot(data=df, x='sentiment')
    plt.title('Sentiment Distribution for Vermont Comments')
    plt.show()

# Main Pipeline Function with Vermont Filtering and Date Retention
def main_pipeline():
    # Load data and filter for Vermont only
    comments_df = pd.read_csv('reddit_comments.csv')
    vermont_df = comments_df[comments_df['state'].str.lower() == 'vermont'].copy()
    print(f"Processing {len(vermont_df)} comments for Vermont.")

    # Retain the original date for later time-based analysis
    vermont_df['created_utc'] = pd.to_datetime(vermont_df['created_utc'], errors='coerce')

    # Parallel preprocessing for Vermont-specific dataset (dual pipeline)
    print("Starting parallel text preprocessing...")
    vermont_df = parallel_preprocess(vermont_df)

    # Sentiment analysis on lightly cleaned text
    print("Starting sentiment analysis on raw-like text...")
    vermont_df = analyze_sentiment(vermont_df)

    # Save processed Vermont data
    vermont_df.to_csv("processed_vermont_comments.csv", index=False)
    print("Processing complete. Data saved to 'processed_vermont_comments.csv'.")

    # Visualize sentiment distribution for Vermont
    plot_sentiment(vermont_df)

# Run the pipeline
if __name__ == "__main__":
    main_pipeline()

    
    