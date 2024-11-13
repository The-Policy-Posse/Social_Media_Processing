# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 00:21:31 2024

@author: dforc

Description:
    This script processes Vermont-specific Reddit comments using multiple NLP 
    transformer models to extract insights such as emotion, bias, sentiment intensity, 
    named entities, and political leanings. It sets up GPU usage, loads data, 
    preprocesses text, and initializes transformer pipelines for various analyses.
"""

import os
import pandas as pd
import numpy as np
import torch
import re
import math
from transformers import pipeline
from tqdm import tqdm
import gc
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from bertopic import BERTopic




########################################################################
## 0. Setup and Configuration
########################################################################

## Set environment variable to handle CUDA launch blocking for debug purposes
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

## Determine if GPU is available, setting device to GPU (0) or CPU (-1)
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

## Set the number of CPU threads for PyTorch (useful for large data)
torch.set_num_threads(8)  # Adjust based on available CPU cores





########################################################################
## 1. Load and Subset the Data
########################################################################

## Load comments dataset, parsing 'created_utc' as datetime for time-based analysis
comments_df = pd.read_csv('reddit_comments.csv', parse_dates=['created_utc'])

## Filter for Vermont-specific comments and create a working DataFrame copy
vermont_df = comments_df[comments_df['state'].str.lower() == 'vermont'].copy()

## Reset index to ensure sequential indexing for Vermont data
vermont_df = vermont_df.reset_index(drop=True)

## Subset to the first 500 rows for testing purposes
# vermont_df = vermont_df.head(500).copy()






########################################################################
## 2. Data Preprocessing
########################################################################

def light_clean_text(text):
    """
    Clean text minimally by removing URLs, mentions, and hashtags, preserving 
    most content while excluding common distractions in social media text.
    """
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove hashtags and mentions
    return text.strip()

## Apply text cleaning to comment body and save to a new column
vermont_df['cleaned_for_transformer'] = vermont_df['body'].apply(light_clean_text)

## Remove rows with empty strings in the cleaned text column
vermont_df = vermont_df[~vermont_df['cleaned_for_transformer'].str.strip().eq('')].reset_index(drop=True)
print(f"After cleaning, {len(vermont_df)} comments remain.")

## Set 'created_utc' as the index to enable time series analysis on comments
vermont_df.set_index('created_utc', inplace=True)






########################################################################
## 3. Initialize Transformers Pipelines
########################################################################

print("Initializing transformer pipelines...")

## Initialize pipeline for emotion classification (uses DistilRoBERTa model)
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    device=device
)

## Initialize pipeline for political bias classification (Twitter-RoBERTa model)
bias_classifier = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=device
)

## Initialize sentiment intensity classifier with BERTweet model
## Note: BERTweet may not always run smoothly on GPU; defaults to CPU if issues arise
sentiment_device = device  # Default to GPU
try:
    sentiment_intensity_classifier = pipeline(
        "text-classification",
        model="finiteautomata/bertweet-base-sentiment-analysis",
        tokenizer="finiteautomata/bertweet-base-sentiment-analysis",
        device=sentiment_device
    )
except Exception as e:
    print("Error initializing sentiment intensity classifier on GPU. Switching to CPU.")
    sentiment_device = -1  # Switch to CPU
    sentiment_intensity_classifier = pipeline(
        "text-classification",
        model="finiteautomata/bertweet-base-sentiment-analysis",
        tokenizer="finiteautomata/bertweet-base-sentiment-analysis",
        device=sentiment_device
    )

## Initialize Named Entity Recognition (NER) pipeline with entity aggregation
ner_classifier = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    device=device,
    aggregation_strategy="simple"
)

## Initialize embedding model for feature extraction (MiniLM for sentence embeddings)
embedding_model = pipeline(
    "feature-extraction",
    model="sentence-transformers/all-MiniLM-L6-v2",
    device=device
)

## Initialize Zero-Shot Classification pipeline for political leaning
political_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)

#############################
## Candidate Labels Set here!
#############################

## Define candidate labels for zero-shot political ideology classification
candidate_labels = ["liberal", "conservative", "libertarian", "socialist", "centrist"]

print("Transformer pipelines initialized successfully.")





########################################################################
## 4. Define Enhanced Batch Processing Functions
########################################################################

def process_in_batches_with_scores(text_data, model, batch_size, **kwargs):
    """
    General batch processing function that captures both labels and confidence scores.
    """
    num_batches = math.ceil(len(text_data) / batch_size)
    labels = []
    scores = []
    for i in tqdm(range(num_batches), desc=f"Processing with {model.model.name_or_path}"):
        batch_text = text_data[i * batch_size: (i + 1) * batch_size]
        try:
            batch_results = model(batch_text, **kwargs)
            for res in batch_results:
                labels.append(res['label'])
                scores.append(res.get('score', None))  # Some models use 'score', others 'confidence'
                
            ## Clear cache and collect garbage
            torch.cuda.empty_cache()
            gc.collect()
            
        ## Error Handling/Prints
        except Exception as e:
            print(f"\nError processing batch {i}")
            print(f"Batch texts: {batch_text}")
            print(f"Exception: {e}")
            raise e
    return labels, scores

def process_in_batches_zero_shot_with_scores(text_data, classifier, candidate_labels, batch_size, **kwargs):
    """
    Batch processing function for Zero-Shot Classification that captures labels and scores.
    """
    num_batches = math.ceil(len(text_data) / batch_size)
    labels = []
    scores = []
    for i in tqdm(range(num_batches), desc="Processing Zero-Shot Classification"):
        batch_text = text_data[i * batch_size: (i + 1) * batch_size]
        try:
            batch_results = classifier(
                batch_text,
                candidate_labels=candidate_labels,
                **kwargs
            )
            for res in batch_results:
                labels.append(res['labels'][0])
                scores.append(res['scores'][0])  ## Assumes scores are in descending order
            ## Clear cache and collect garbage
            torch.cuda.empty_cache()
            gc.collect()
            
        ## Error Handling/Prints    
        except Exception as e:
            print(f"\nError processing batch {i}")
            print(f"Batch texts: {batch_text}")
            print(f"Exception: {e}")
            raise e
    return labels, scores

########################################################################
## 5. Run Analyses in Batches
########################################################################


### || Garbage Collection || ###
## Helper function to ensure garbage collection and cache clearing
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    
    
    
## 5.1 Enhanced Emotion Analysis with Confidence Scores
print("\n--- Running Emotion Analysis ---")
emotion_batch_size = 32
emotion_labels, emotion_scores = process_in_batches_with_scores(
    vermont_df['cleaned_for_transformer'].tolist(),
    emotion_classifier,
    batch_size=emotion_batch_size,
    truncation=True,
    max_length=512
)
vermont_df['emotion'] = emotion_labels
vermont_df['emotion_score'] = emotion_scores

## Save intermediate results
vermont_df.to_pickle('vermont_df_after_emotion.pkl')
print("Emotion analysis results saved.")

## Free up memory
del emotion_labels, emotion_scores
cleanup()


###########################################################


## 5.2 Enhanced Political Bias Detection with Confidence Scores
print("\n--- Running Political Bias Detection ---")
bias_batch_size = 32
bias_labels, bias_scores = process_in_batches_with_scores(
    vermont_df['cleaned_for_transformer'].tolist(),
    bias_classifier,
    batch_size=bias_batch_size,
    truncation=True,
    max_length=512
)
vermont_df['political_bias'] = bias_labels
vermont_df['bias_score'] = bias_scores

## Save intermediate results
vermont_df.to_pickle('vermont_df_after_bias.pkl')
print("Political bias detection results saved.")

# Free up memory
del bias_labels, bias_scores
cleanup()


###########################################################



## 5.3 Enhanced Sentiment Intensity Analysis with Confidence Scores
print("\n--- Running Sentiment Intensity Analysis ---")
sentiment_batch_size = 32
sentiment_labels, sentiment_scores = process_in_batches_with_scores(
    vermont_df['cleaned_for_transformer'].tolist(),
    sentiment_intensity_classifier,
    batch_size=sentiment_batch_size,
    truncation=True,
    max_length=128  # Set max_length to 128 for BERTweet
)
vermont_df['sentiment_intensity'] = sentiment_labels
vermont_df['sentiment_score'] = sentiment_scores

## Save intermediate results
vermont_df.to_pickle('vermont_df_after_sentiment.pkl')
print("Sentiment intensity analysis results saved.")

## Free up memory
del sentiment_labels, sentiment_scores
cleanup()



###########################################################

### || Not Currently Working! || ###
## 5.4 Enhanced Named Entity Recognition (NER) with Entity Types
print("\n--- Running Named Entity Recognition (NER) ---")
ner_batch_size = 32
ner_results = []
num_batches_ner = math.ceil(len(vermont_df) / ner_batch_size)

for i in tqdm(range(num_batches_ner), desc="Processing NER"):
    batch_text = vermont_df['cleaned_for_transformer'].iloc[i * ner_batch_size: (i + 1) * ner_batch_size].tolist()
    try:
        ## **Removed truncation=True and max_length=512 from the NER pipeline call**
        batch_entities = ner_classifier(batch_text)
        ner_results.extend(batch_entities)
        # #Clear cache and collect garbage
        cleanup()
        
    ## Error Handling/Prints    
    except Exception as e:
        print(f"\nError processing NER batch {i}")
        print(f"Batch texts: {batch_text}")
        print(f"Exception: {e}")
        raise e

## Extract entities with their types from NER results
def extract_entities_with_types(ner_output):
    """
    Extract entities and their types from NER output.
    """
    return [(entity['word'], entity['entity_group']) for entity in ner_output]

vermont_df['entities_with_types'] = [extract_entities_with_types(res) if res else [] for res in ner_results]

## Separate entities by type into distinct columns
def separate_entities(entities):
    """
    Separate entities into PERSON, LOCATION, and ORGANIZATION.
    """
    entity_dict = {'PERSON': [], 'LOCATION': [], 'ORGANIZATION': []}
    for word, entity_type in entities:
        if entity_type in entity_dict:
            entity_dict[entity_type].append(word)
    return pd.Series([
        ', '.join(entity_dict['PERSON']) if entity_dict['PERSON'] else '',
        ', '.join(entity_dict['LOCATION']) if entity_dict['LOCATION'] else '',
        ', '.join(entity_dict['ORGANIZATION']) if entity_dict['ORGANIZATION'] else ''
    ])

vermont_df[['persons', 'locations', 'organizations']] = vermont_df['entities_with_types'].apply(separate_entities)

## Save intermediate results
vermont_df.to_pickle('vermont_df_after_ner.pkl')
print("Named entity recognition results saved.")

## Free up memory
del ner_results
cleanup()



###########################################################


## 5.5 Embedding Extraction
print("\n--- Extracting Embeddings for Topic Modeling ---")
embedding_batch_size = 32
embedding_results = []

num_batches_embedding = math.ceil(len(vermont_df) / embedding_batch_size)
for i in tqdm(range(num_batches_embedding), desc=f"Processing with {embedding_model.model.name_or_path}"):
    batch_text = vermont_df['cleaned_for_transformer'].iloc[i * embedding_batch_size: (i + 1) * embedding_batch_size].tolist()
    try:
        batch_embeddings = embedding_model(batch_text, truncation=True, max_length=512)
        ## Flatten embeddings by averaging over tokens
        for emb in batch_embeddings:
            flat_emb = np.mean(emb[0], axis=0)
            embedding_results.append(flat_emb)
        ## Clear cache and collect garbage
        cleanup()
        
    ## Error Handling/Prints    
    except Exception as e:
        print(f"\nError processing embedding batch {i}")
        print(f"Batch texts: {batch_text}")
        print(f"Exception: {e}")
        raise e

## Convert embeddings to a numpy array
embeddings = np.array(embedding_results)

## Save embeddings
np.save('vermont_embeddings.npy', embeddings)
print("Embeddings saved to 'vermont_embeddings.npy'")

# Save DataFrame with embeddings
vermont_df['embeddings'] = embedding_results
vermont_df.to_pickle('vermont_df_after_embeddings.pkl')
print("DataFrame with embeddings saved to 'vermont_df_after_embeddings.pkl'")

## Free up memory
del embedding_results
cleanup()



###########################################################



## 5.6 Enhanced Political Leaning Detection with Confidence Scores
print("\n--- Running Political Leaning Detection ---")
political_batch_size = 32
political_labels, political_scores = process_in_batches_zero_shot_with_scores(
    vermont_df['cleaned_for_transformer'].tolist(),
    political_classifier,
    candidate_labels,
    batch_size=political_batch_size,
    multi_label=False,
    truncation=True,
    max_length=512
)
vermont_df['political_leaning'] = political_labels
vermont_df['political_leaning_score'] = political_scores

## Save intermediate results
vermont_df.to_pickle('vermont_df_after_political.pkl')
print("Political leaning detection results saved.")

## Free up memory
del political_labels, political_scores
cleanup()






########################################################################
## 6. Topic Modeling
########################################################################

## 6.1 KMeans Clustering
print("\n--- Performing Topic Modeling with KMeans Clustering ---")

#### || Set Kmeans Clusters Here || ####
num_clusters = 10  ## Adjust this number to increase/decrease clusters

## Initialize and fit KMeans
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
vermont_df['kmeans_topic'] = kmeans.fit_predict(embeddings)


## Analyzing clusters to interpret topics
print("\nTop terms per cluster using TF-IDF:")


## Use TF-IDF to find top terms in each cluster
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(vermont_df['cleaned_for_transformer'])
terms = tfidf_vectorizer.get_feature_names_out()

## Create a dictionary to hold top terms for each cluster
cluster_terms = {}
for cluster_num in range(num_clusters):
    # Use np.where to get integer positions
    cluster_indices = np.where(vermont_df['kmeans_topic'] == cluster_num)[0]
    cluster_tfidf = tfidf_matrix[cluster_indices]
    # Calculate mean tf-idf score for each term in the cluster
    mean_tfidf = cluster_tfidf.mean(axis=0).A1
    top_indices = mean_tfidf.argsort()[::-1][:10]  # Get indices of top 10 terms
    top_terms = [terms[i] for i in top_indices]
    cluster_terms[cluster_num] = top_terms
    print(f"Cluster {cluster_num}: {top_terms}")
    
    

## Add the top terms to the DataFrame for each comment
def get_cluster_top_terms(cluster_label):
    """
    Retrieve top terms for a given cluster label.
    """
    return cluster_terms.get(cluster_label, [])


vermont_df['kmeans_top_terms'] = vermont_df['kmeans_topic'].apply(get_cluster_top_terms)

## Save intermediate results
vermont_df.to_pickle('vermont_df_after_kmeans.pkl')
print("KMeans clustering results saved.")

## Free up memory
cleanup()



###########################################################





## 6.2 BERTopic
print("\n--- Performing Topic Modeling with BERTopic ---")
topic_model = BERTopic(embedding_model=None, calculate_probabilities=True, verbose=True)
topics, probabilities = topic_model.fit_transform(vermont_df['cleaned_for_transformer'], embeddings)

## Add topics to the dataframe
vermont_df['bertopic_topic'] = topics

## Save BERTopic model
topic_model.save("bertopic_model")
print("BERTopic model saved to 'bertopic_model'")

## Save intermediate results
vermont_df.to_pickle('vermont_df_after_bertopic.pkl')
print("BERTopic results saved.")

## Free up memory
del topics, probabilities, topic_model
cleanup()

########################################################################
## 7. Saving Final Outputs for Analysis
########################################################################

print("\n--- Saving Final Outputs ---")

## Reset index to include 'created_utc' in the DataFrame
vermont_df_reset = vermont_df.reset_index()


## 7.1 Save simple columns to CSV
simple_columns = [
    'cleaned_for_transformer',
    'emotion',
    'emotion_score',
    'political_bias',
    'bias_score',
    'sentiment_intensity',
    'sentiment_score',
    'political_leaning',
    'political_leaning_score',
    'kmeans_topic',
    'bertopic_topic',
    'score',
    'post_id'
]
simple_columns_with_date = ['created_utc'] + simple_columns
vermont_df_reset[simple_columns_with_date].to_csv('vermont_simple_outputs.csv', index=False)
print("Simple outputs saved to 'vermont_simple_outputs.csv'")


## 7.2 Save complex columns to JSON
complex_columns = [
    'cleaned_for_transformer',
    'entities_with_types',
    'persons',
    'locations',
    'organizations',
    'kmeans_top_terms'
]
vermont_df_reset[complex_columns].to_json('vermont_complex_outputs.json', orient='records', lines=True)
print("Complex outputs saved to 'vermont_complex_outputs.json'")

## 7.3 Embeddings already saved earlier as 'vermont_embeddings.npy'

## 7.4 Save the entire DataFrame to a Pickle file
## || Temporary and Optional!! ||
vermont_df.to_pickle('vermont_df_final.pkl')
print("Entire DataFrame saved to 'vermont_df_final.pkl'")

########################################################################
## 8. Prepare for Analysis
########################################################################

print("\nProcessing Complete. Data is saved and ready for analysis.")
