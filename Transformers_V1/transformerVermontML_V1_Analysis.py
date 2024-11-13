# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 02:35:51 2024

@author: dforc
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.manifold import TSNE
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from scipy.stats import chi2_contingency
import matplotlib.dates as mdates
import json

## Set Seaborn style for All Plots
sns.set_style('whitegrid')
sns.set_context("notebook")  ## Options Avail: paper, notebook, talk, poster


########################################################################
### 1. Load the Data
########################################################################

## Load simple outputs from a CSV file
simple_df = pd.read_csv('vermont_simple_outputs.csv', parse_dates=['created_utc'])

## Load complex outputs from JSON file
complex_df = pd.read_json('vermont_complex_outputs.json', orient='records', lines=True)

## Load embeddings from a NumPy file
embeddings = np.load('vermont_embeddings.npy')

## Merge DataFrames on index to align simple and complex outputs
vermont_df = simple_df.merge(complex_df, left_index=True, right_index=True, suffixes=('_simple',
                                                                                      '_complex'))

## Select 'cleaned_for_transformer' column for further analysis
vermont_df['cleaned_for_transformer'] = vermont_df['cleaned_for_transformer_simple']

## Drop redundant columns after merging
vermont_df.drop(columns=['cleaned_for_transformer_simple',
                         'cleaned_for_transformer_complex'], inplace=True)

## Add embeddings as a separate column for later use in clustering or visualization
vermont_df['embeddings'] = list(embeddings)



########################################################################
### 2. Data Preprocessing
########################################################################

## Prepare the data for analysis

## Convert string representations of lists back to actual lists for analysis, if necessary
for col in ['entities_with_types', 'kmeans_top_terms']:
    if col in vermont_df.columns and isinstance(vermont_df[col].iloc[0], str):
        vermont_df[col] = vermont_df[col].apply(ast.literal_eval)

## Set 'created_utc' as the index for time series analysis
vermont_df.set_index('created_utc', inplace=True)



########################################################################
### 3. Define Category Orders and Color Palettes
########################################################################

## Set category orders based on frequency and define color palettes

### 3.1 Define Category Orders based on frequency of occurrence
emotion_order = vermont_df['emotion'].value_counts().index.tolist()
sentiment_order = vermont_df['sentiment_intensity'].value_counts().index.tolist()
bias_order = vermont_df['political_bias'].value_counts().index.tolist()
leaning_order = vermont_df['political_leaning'].value_counts().index.tolist()
kmeans_order = sorted(vermont_df['kmeans_topic'].unique())
bertopic_order = sorted(vermont_df['bertopic_topic'].unique())

### 3.2 Define Color Palettes for each category type
## Custom color dictionaries are used for specific visual themes in the analysis

## Emotion color palette (manually assigned for clarity)
emotion_color_dict = {
    'neutral': 'gray',
    'surprise': 'yellow',
    'disgust': 'green',
    'anger': 'red',
    'joy': 'gold',
    'sadness': 'blue',
    'fear': 'purple'
}

## Sentiment intensity color palette
sentiment_palette = sns.color_palette("coolwarm", len(sentiment_order))
sentiment_color_dict = dict(zip(sentiment_order, sentiment_palette))

## Political bias color palette
bias_palette = sns.color_palette("Set1", len(bias_order))
bias_color_dict = dict(zip(bias_order, bias_palette))

## Political leaning color palette with specific color assignments
leaning_color_dict = {
    'conservative': 'red',
    'liberal': 'blue',
    'centrist': 'yellow',
    'libertarian': 'green',
    'socialist': 'purple'
    ## Add more categories here if additional political leanings added
    ## !! This must be expanded if categories expand !! 
}

## KMeans clusters color palette with integer keys
kmeans_unique = sorted(vermont_df['kmeans_topic'].unique())  # Example: [0,1,2,...,9]
kmeans_palette = sns.color_palette("tab10", len(kmeans_unique))
kmeans_color_dict = dict(zip(kmeans_unique, kmeans_palette))

## BERTopic clusters color palette
bertopic_unique = sorted(vermont_df['bertopic_topic'].unique())
bertopic_palette = sns.color_palette("hsv", 60)  ## Define for top 60 topics




########################################################################
### 4. Exploratory Data Analysis (EDA)
########################################################################

## This section includes helper functions and specific visualizations 
## for exploring the Vermont Reddit data, covering topics such as emotion 
## analysis, entity visualization, and word cloud generation.

## Define standard figure size for consistency in plots
standard_figsize = (10, 6)


########################################################################
### Helper Functions for Plotting and Analysis
########################################################################

def plot_count(data, x, order, palette, title, xlabel, ylabel, remove_legend=True):
    """
    Plot a count plot with consistent styling.
    Supports palette in either dictionary or list format for flexible color options.
    """
    plt.figure(figsize=standard_figsize)
    if isinstance(palette, dict):
        sns.countplot(data=data, x=x, order=order, palette=palette, dodge=False)
    elif isinstance(palette, list):
        sns.countplot(data=data, x=x, order=order, palette=palette, dodge=False)
    else:
        raise ValueError("Palette must be a dictionary or a list of colors.")
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    if remove_legend:
        plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.show()

def plot_box(data, x, y, order, palette, title, xlabel, ylabel):
    """
    Plot a box plot with consistent styling for visualizing score distributions.
    """
    plt.figure(figsize=standard_figsize)
    sns.boxplot(data=data, x=x, y=y, order=order, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_top_entities(entity_series, entity_type, top_n=20):
    """
    Plot the top N entities of a specified type, useful for examining prominent topics.
    """
    all_entities = vermont_df[entity_series].dropna().tolist()
    if not all_entities:
        print(f"\nNo entities found for type: {entity_type}")
        return
    
    # Split comma-separated entities and flatten into a list
    all_entities = [entity.strip() for sublist in all_entities for entity in
                    sublist.split(',') if entity.strip()]
    if not all_entities:
        print(f"\nNo entities found after splitting for type: {entity_type}")
        return
    entity_counts = Counter(all_entities).most_common(top_n)
    
    print(f"\nTop {top_n} {entity_type} Entities:\n", entity_counts)
    
    # Create DataFrame for plotting
    entities_df = pd.DataFrame(entity_counts, columns=['Entity', 'Count'])
    
    plt.figure(figsize=standard_figsize)
    sns.barplot(data=entities_df, x='Entity', y='Count', palette='viridis')
    plt.title(f'Top {top_n} {entity_type} Entities in Vermont Reddit Comments')
    plt.xlabel(f'{entity_type} Entity')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def generate_wordcloud_for_entity(entity_series, entity_type, min_word_length=3):
    """
    Generate a word cloud visualization for a specified entity type.
    """
    all_entities = vermont_df[entity_series].dropna().tolist()
    if not all_entities:
        print(f"\nNo entities found for type: {entity_type}. Skipping word cloud.")
        return
    
    text = ' '.join([entity.strip() for sublist in all_entities for entity in 
                     sublist.split(',') if entity.strip()])
    if not text:
        print(f"\nNo valid text found for type: {entity_type}. Skipping word cloud.")
        return
    
    wordcloud = WordCloud(
        width=800, height=400, background_color='white',
        stopwords=STOPWORDS, max_words=100
    ).generate(text)
    
    plt.figure(figsize=(15, 7.5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {entity_type} Entities')
    plt.tight_layout()
    plt.show()

def format_xaxis(ax, resample_freq, min_date, max_date):
    """
    Format the x-axis of a plot based on resampling frequency (daily, weekly, or monthly).
    """
    if resample_freq == 'D':
        date_format = '%Y-%m-%d'
        locator = mdates.DayLocator()
    elif resample_freq == 'W':
        date_format = '%Y-%m-%d'
        locator = mdates.WeekdayLocator(byweekday=mdates.MO)
    else:  # 'M'
        date_format = '%Y-%m'
        locator = mdates.MonthLocator()

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    ax.set_xlim(min_date, max_date)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    

########################################################################
### 4.1 Emotion Analysis
########################################################################

## Conduct analysis on the emotion data in Vermont Reddit comments

# Count the frequency of each emotion
emotion_counts = vermont_df['emotion'].value_counts()
print("Emotion Counts:\n", emotion_counts)

# Plot the distribution of emotions across all comments
plot_count(
    data=vermont_df,
    x='emotion',
    order=emotion_order,
    palette=emotion_color_dict,
    title='Distribution of Emotions in Vermont Reddit Comments',
    xlabel='Emotion',
    ylabel='Count'
)

# Plot the distribution of emotion confidence scores for each emotion category
plot_box(
    data=vermont_df,
    x='emotion',
    y='emotion_score',
    order=emotion_order,
    palette=emotion_color_dict,
    title='Distribution of Emotion Confidence Scores',
    xlabel='Emotion',
    ylabel='Confidence Score'
)



########################################################################
### 4.2 Sentiment Intensity Analysis
########################################################################

## Analyze the sentiment intensity data in Vermont Reddit comments

# Count the frequency of each sentiment intensity label
sentiment_counts = vermont_df['sentiment_intensity'].value_counts()
print("\nSentiment Intensity Counts:\n", sentiment_counts)

# Plot the distribution of sentiment intensity labels
plot_count(
    data=vermont_df,
    x='sentiment_intensity',
    order=sentiment_order,
    palette=sentiment_color_dict,
    title='Distribution of Sentiment Intensity in Vermont Reddit Comments',
    xlabel='Sentiment Intensity',
    ylabel='Count'
)

# Plot the distribution of sentiment intensity confidence scores
plot_box(
    data=vermont_df,
    x='sentiment_intensity',
    y='sentiment_score',
    order=sentiment_order,
    palette=sentiment_color_dict,
    title='Distribution of Sentiment Intensity Confidence Scores',
    xlabel='Sentiment Intensity',
    ylabel='Confidence Score'
)



########################################################################
### 4.3 Political Bias Analysis
########################################################################

## Examine political bias labels within Vermont Reddit data

# Count the frequency of each political bias label
bias_counts = vermont_df['political_bias'].value_counts()
print("\nPolitical Bias Counts:\n", bias_counts)

# Plot the distribution of political bias labels
plot_count(
    data=vermont_df,
    x='political_bias',
    order=bias_order,
    palette=bias_color_dict,
    title='Distribution of Political Bias in Vermont Reddit Comments',
    xlabel='Political Bias',
    ylabel='Count'
)

# Plot the distribution of political bias confidence scores
plot_box(
    data=vermont_df,
    x='political_bias',
    y='bias_score',
    order=bias_order,
    palette=bias_color_dict,
    title='Distribution of Political Bias Confidence Scores',
    xlabel='Political Bias',
    ylabel='Confidence Score'
)



########################################################################
### 4.4 Political Leaning Analysis
########################################################################

## Assess the political leaning data to understand ideological distributions

# Count the frequency of each political leaning label
leaning_counts = vermont_df['political_leaning'].value_counts()
print("\nPolitical Leaning Counts:\n", leaning_counts)

# Plot the distribution of political leaning labels
plot_count(
    data=vermont_df,
    x='political_leaning',
    order=leaning_order,
    palette=leaning_color_dict,
    title='Distribution of Political Leanings in Vermont Reddit Comments',
    xlabel='Political Leaning',
    ylabel='Count'
)

# Plot the distribution of political leaning confidence scores
plot_box(
    data=vermont_df,
    x='political_leaning',
    y='political_leaning_score',
    order=leaning_order,
    palette=leaning_color_dict,
    title='Distribution of Political Leaning Confidence Scores',
    xlabel='Political Leaning',
    ylabel='Confidence Score'
)



########################################################################
### 4.5 Named Entity Recognition (NER) Analysis
########################################################################

## Explore Named Entity Recognition (NER) results, focusing on Persons, Locations, and Organizations

# Plot the top 20 most frequently mentioned persons
plot_top_entities('persons', 'Person', top_n=20)

# Plot the top 20 most frequently mentioned locations
plot_top_entities('locations', 'Location', top_n=20)

# Plot the top 20 most frequently mentioned organizations
plot_top_entities('organizations', 'Organization', top_n=20)

# Generate word clouds to visualize entity mentions for Persons, Locations, and Organizations
generate_wordcloud_for_entity('persons', 'Person')
generate_wordcloud_for_entity('locations', 'Location')
generate_wordcloud_for_entity('organizations', 'Organization')




########################################################################
### 4.6 Topic Modeling Analysis
########################################################################

## This section analyzes topic modeling results from KMeans and BERTopic
## clustering, focusing on topic distribution and confidence score metrics.



########################################################################
### KMeans Clustering Topics
########################################################################

# Count the number of comments in each KMeans topic cluster
kmeans_counts = vermont_df['kmeans_topic'].value_counts().sort_index()
print("\nKMeans Topic Counts:\n", kmeans_counts)

# Plot the distribution of KMeans topics
plot_count(
    data=vermont_df,
    x='kmeans_topic',
    order=kmeans_order,
    palette=[kmeans_color_dict[k] for k in kmeans_order],  # Generate list of colors
    title='Distribution of KMeans Topics in Vermont Reddit Comments',
    xlabel='KMeans Topic',
    ylabel='Count'
)

# Display the top terms for each KMeans cluster
for cluster_num in kmeans_order:
    cluster_data = vermont_df.loc[vermont_df['kmeans_topic'] == cluster_num, 'kmeans_top_terms']
    if not cluster_data.empty:
        top_terms = cluster_data.iloc[0]
        if isinstance(top_terms, list):
            print(f"\nKMeans Cluster {cluster_num} Top Terms: {', '.join(top_terms)}")
        else:
            print(f"\nKMeans Cluster {cluster_num} Top Terms: Data not in expected format.")
    else:
        print(f"\nNo data found for KMeans Cluster {cluster_num}.")

# Plot the average emotion confidence score per KMeans topic
if 'kmeans_topic' in vermont_df.columns and 'emotion_score' in vermont_df.columns:
    plt.figure(figsize=standard_figsize)
    sns.barplot(
        data=vermont_df,
        x='kmeans_topic',
        y='emotion_score',
        order=kmeans_order,
        palette=[kmeans_color_dict[k] for k in kmeans_order]  # Generate list of colors
    )
    plt.title('Average Emotion Confidence Score per KMeans Topic')
    plt.xlabel('KMeans Topic')
    plt.ylabel('Average Emotion Confidence Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("\nColumns 'kmeans_topic' or 'emotion_score' not found. Skipping plot.")
    
    

########################################################################
### BERTopic Topics
########################################################################

# Count the number of comments in each BERTopic topic
bertopic_counts = vermont_df['bertopic_topic'].value_counts().sort_index()
print("\nBERTopic Topic Counts:\n", bertopic_counts)



########################################################################
### 4.6.1 Limit BERTopic Topics to Top 60
########################################################################

## Focus on the top 60 most frequent BERTopic topics for visualization

# Identify top 60 BERTopic topics based on frequency
top_60_bertopic = bertopic_counts.nlargest(60).index.tolist()

# Create a new column 'bertopic_topic_limited' where topics not in top 60 are labeled as 'Other'
vermont_df['bertopic_topic_limited'] = vermont_df['bertopic_topic'].apply(
    lambda x: x if x in top_60_bertopic else 'Other'
)

# Define limited BERTopic order for plotting
bertopic_order_limited = top_60_bertopic + ['Other']

# Generate a color palette for the 61 categories (60 topics + 'Other')
bertopic_palette_limited = sns.color_palette("hsv", len(bertopic_order_limited))




########################################################################
### 4.6.2 Plot BERTopic Topics Limited to Top 60
########################################################################

# Plot the distribution of limited BERTopic topics
plot_count(
    data=vermont_df,
    x='bertopic_topic_limited',
    order=bertopic_order_limited,
    palette=bertopic_palette_limited,
    title='Distribution of BERTopic Topics in Vermont Reddit Comments (Top 60)',
    xlabel='BERTopic Topic',
    ylabel='Count'
)

# Plot the average sentiment confidence score per limited BERTopic topic
if 'bertopic_topic_limited' in vermont_df.columns and 'sentiment_score' in vermont_df.columns:
    plt.figure(figsize=standard_figsize)
    sns.barplot(
        data=vermont_df,
        x='bertopic_topic_limited',
        y='sentiment_score',
        order=bertopic_order_limited,
        palette=bertopic_palette_limited
    )
    plt.title('Average Sentiment Confidence Score per BERTopic Topic (Top 60)')
    plt.xlabel('BERTopic Topic')
    plt.ylabel('Average Sentiment Confidence Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("\nColumns 'bertopic_topic_limited' or 'sentiment_score' not found. Skipping plot.")
    
    
    

########################################################################
### 4.6.3 Update BERTopic Palette for t-SNE Visualization
########################################################################

# Assign 'Other' as the last color in the BERTopic color dictionary for visualization
bertopic_color_dict_limited = {topic: color for topic, 
                               color in zip(bertopic_order_limited, bertopic_palette_limited)}



########################################################################
### 4.7 Word Clouds for Political Leaning
########################################################################

## Generate word clouds to visualize common terms associated with each 
## political leaning, highlighting distinct vocabulary across leanings.

def generate_wordcloud_for_leaning(leaning_label, min_word_length=3):
    """
    Generate a word cloud for a specified political leaning.
    Filters out words shorter than the minimum word length for clarity.
    """
    # Collect text from the 'cleaned_for_transformer' column for the given leaning
    text = ' '.join(vermont_df[vermont_df['political_leaning'] == leaning_label]['cleaned_for_transformer'])
    # Optionally, filter out short words
    text = ' '.join([word for word in text.split() if len(word) >= min_word_length])
    if not text:
        print(f"\nNo valid text found for type: {leaning_label}. Skipping word cloud.")
        return
    
    # Generate and display the word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=STOPWORDS,
        max_words=100,
        colormap='viridis'  # Use a color map that aligns with the political theme
    ).generate(text)
    
    plt.figure(figsize=(15, 7.5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Political Leaning: {leaning_label}')
    plt.tight_layout()
    plt.show()

# Generate word clouds for each political leaning category in the predefined order
for leaning_label in leaning_order:
    generate_wordcloud_for_leaning(leaning_label)
    
    

########################################################################
### 4.8 Topic Modeling Visualization with t-SNE
########################################################################

## Use t-SNE to visualize high-dimensional embeddings in 2D space, 
## coloring by various categories for comparative insights.

# Reduce dimensionality of embeddings to 2D using t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Create a DataFrame to store 2D t-SNE results along with relevant metadata
vis_df = pd.DataFrame({
    'x': embeddings_2d[:, 0],
    'y': embeddings_2d[:, 1],
    'kmeans_topic': vermont_df['kmeans_topic'].values,
    'bertopic_topic_limited': vermont_df['bertopic_topic_limited'].values,  # Use limited BERTopic topics
    'political_leaning': vermont_df['political_leaning'].values,
    'sentiment_intensity': vermont_df['sentiment_intensity'].values,
    'emotion': vermont_df['emotion'].values,
    'emotion_score': vermont_df['emotion_score'].values,
    'sentiment_score': vermont_df['sentiment_score'].values,
    'political_leaning_score': vermont_df['political_leaning_score'].values
})



########################################################################
### 4.8.1 t-SNE Visualization Colored by Political Leaning
########################################################################

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=vis_df,
    x='x',
    y='y',
    hue='political_leaning',
    palette=leaning_color_dict,
    alpha=0.7,
    edgecolor=None
)
plt.title('t-SNE Visualization of Embeddings Colored by Political Leaning')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Political Leaning', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



########################################################################
### 4.8.2 t-SNE Visualization Colored by Emotion
########################################################################

plt.figure(figsize=standard_figsize)
sns.scatterplot(
    data=vis_df,
    x='x',
    y='y',
    hue='emotion',
    palette=emotion_color_dict,
    alpha=0.7,
    edgecolor=None
)
plt.title('t-SNE Visualization of Embeddings Colored by Emotion')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



########################################################################
### 4.8.3 t-SNE Visualization Colored by KMeans Topic
########################################################################

plt.figure(figsize=standard_figsize)
sns.scatterplot(
    data=vis_df,
    x='x',
    y='y',
    hue='kmeans_topic',
    palette=kmeans_palette,  # Use list of KMeans topic colors
    alpha=0.7,
    edgecolor=None
)
plt.title('t-SNE Visualization of Embeddings Colored by KMeans Topic')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='KMeans Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



########################################################################
### 4.8.4 t-SNE Visualization Colored by BERTopic Topic (Limited to Top 60)
########################################################################

plt.figure(figsize=standard_figsize)
sns.scatterplot(
    data=vis_df,
    x='x',
    y='y',
    hue='bertopic_topic_limited',
    palette=bertopic_color_dict_limited,  # Use limited BERTopic color palette
    alpha=0.7,
    edgecolor=None,
    legend=False  # Remove legend as requested
)
plt.title('t-SNE Visualization of Embeddings Colored by BERTopic Topic (Top 60)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
# plt.legend(title='BERTopic Topic', bbox_to_anchor=(1.05, 1), loc='upper left')  # Removed legend
plt.tight_layout()
plt.show()



########################################################################
### 4.8.5 t-SNE Visualization Colored by Sentiment Confidence Score
########################################################################

plt.figure(figsize=standard_figsize)
sns.scatterplot(
    data=vis_df,
    x='x',
    y='y',
    hue='sentiment_score',
    palette='coolwarm',
    alpha=0.7,
    edgecolor=None
)
plt.title('t-SNE Visualization of Embeddings Colored by Sentiment Confidence Score')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Sentiment Confidence Score', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



########################################################################
### 4.9 Additional Plot: High Confidence Sentiment Word Cloud
########################################################################

## Generate a word cloud to visualize the most common terms in comments
## with high sentiment confidence scores, filtering for sentiment richness.

# Filter comments with high sentiment confidence scores (>= 0.8)
high_conf_sentiment = vermont_df[vermont_df['sentiment_score'] >= 0.8]
if not high_conf_sentiment.empty:
    # Concatenate text from high-confidence comments
    text_high_conf = ' '.join(high_conf_sentiment['cleaned_for_transformer'])
    wordcloud_high_conf = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=STOPWORDS,
        max_words=100
    ).generate(text_high_conf)

    # Display the word cloud
    plt.figure(figsize=(15, 7.5))
    plt.imshow(wordcloud_high_conf, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for High Confidence Sentiment Comments')
    plt.tight_layout()
    plt.show()
else:
    print("\nNo comments with sentiment confidence score >= 0.8 found. Skipping high confidence sentiment word cloud.")



########################################################################
### 4.10 Time Series Analysis
########################################################################

## Perform time series analysis on various aspects of the Vermont Reddit data,
## including emotions, sentiment intensity, and political leaning over time.

# Calculate the date range and select a resampling frequency
min_date = vermont_df.index.min()
max_date = vermont_df.index.max()
date_range_days = (max_date - min_date).days
print(f"Date range: {min_date} to {max_date} ({date_range_days} days)")

# Function to determine the appropriate resampling frequency based on date range
def get_resampling_frequency(date_range_days):
    if date_range_days <= 7:
        return 'D'  # Daily
    elif date_range_days <= 30:
        return 'D'  # Daily
    elif date_range_days <= 90:
        return 'W'  # Weekly
    else:
        return 'M'  # Monthly

resample_freq = get_resampling_frequency(date_range_days)
print(f"Using resampling frequency: {resample_freq}")

# Function to format x-axis based on resampling frequency for consistent date displays
def format_xaxis(ax, resample_freq, min_date, max_date):
    if resample_freq == 'D':
        date_format = '%Y-%m-%d'
        locator = mdates.DayLocator()
    elif resample_freq == 'W':
        date_format = '%Y-%m-%d'
        locator = mdates.WeekdayLocator(byweekday=mdates.MO)
    else:  # 'M'
        date_format = '%Y-%m'
        locator = mdates.MonthLocator()

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    ax.set_xlim(min_date, max_date)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    
    

########################################################################
### 4.10.1 Emotions Over Time
########################################################################

# Resample and count emotions over the selected frequency
emotion_resampled = vermont_df.resample(resample_freq)['emotion'].value_counts().unstack().fillna(0).reindex(columns=emotion_order)

# Plot the resampled emotion counts over time
plt.figure(figsize=(12, 6))
emotion_resampled.plot(kind='line', marker='o', linewidth=2)
ax = plt.gca()

# Apply color palette to each emotion line
for emotion in emotion_order:
    plt.plot([], [], color=emotion_color_dict[emotion], label=emotion)  # Dummy plot for legend

# Format x-axis
format_xaxis(ax, resample_freq, min_date, max_date)

plt.title('Emotion Counts Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot the average emotion confidence score over time
emotion_score_resampled = vermont_df.resample(resample_freq)['emotion_score'].mean()

plt.figure(figsize=standard_figsize)
emotion_score_resampled.plot(kind='line', marker='o', color='purple', linewidth=2)
ax = plt.gca()

# Format x-axis
format_xaxis(ax, resample_freq, min_date, max_date)

plt.title('Average Emotion Confidence Score Over Time')
plt.xlabel('Date')
plt.ylabel('Average Emotion Confidence Score')
plt.tight_layout()
plt.show()




########################################################################
### 4.10.2 Sentiment Intensity Over Time
########################################################################

# Resample and count sentiment intensity values over the selected frequency
sentiment_resampled = vermont_df.resample(resample_freq)['sentiment_intensity'].value_counts().unstack().fillna(0).reindex(columns=sentiment_order)

# Plot the resampled sentiment intensity counts over time
plt.figure(figsize=standard_figsize)
sns.lineplot(data=sentiment_resampled, markers=True, dashes=False, palette=sentiment_color_dict)
ax = plt.gca()

# Format x-axis
format_xaxis(ax, resample_freq, min_date, max_date)

plt.title('Sentiment Intensity Counts Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend(title='Sentiment Intensity', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot the average sentiment confidence score over time
sentiment_score_resampled = vermont_df.resample(resample_freq)['sentiment_score'].mean()

plt.figure(figsize=standard_figsize)
sns.lineplot(data=sentiment_score_resampled, marker='o', color='orange', linewidth=2)
ax = plt.gca()

# Format x-axis
format_xaxis(ax, resample_freq, min_date, max_date)

plt.title('Average Sentiment Confidence Score Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Confidence Score')
plt.tight_layout()
plt.show()




########################################################################
### 4.10.3 Political Leaning Over Time
########################################################################

# Resample and count political leaning values over the selected frequency
leaning_resampled = vermont_df.resample(resample_freq)['political_leaning'].value_counts().unstack().fillna(0).reindex(columns=leaning_order)

# Plot the resampled political leaning counts over time
plt.figure(figsize=standard_figsize)
sns.lineplot(data=leaning_resampled, markers=True, dashes=False, palette=leaning_color_dict)
ax = plt.gca()

# Format x-axis
format_xaxis(ax, resample_freq, min_date, max_date)

plt.title('Political Leaning Counts Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend(title='Political Leaning', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot the average political leaning confidence score over time
leaning_score_resampled = vermont_df.resample(resample_freq)['political_leaning_score'].mean()

plt.figure(figsize=standard_figsize)
sns.lineplot(data=leaning_score_resampled, marker='o', color='green', linewidth=2)
ax = plt.gca()

# Format x-axis
format_xaxis(ax, resample_freq, min_date, max_date)

plt.title('Average Political Leaning Confidence Score Over Time')
plt.xlabel('Date')
plt.ylabel('Average Political Leaning Confidence Score')
plt.tight_layout()
plt.show()





########################################################################
### 4.10.4 Topic Distribution Over Time
########################################################################

# Resample and count KMeans topics over the chosen frequency
kmeans_resampled = vermont_df.resample(resample_freq)['kmeans_topic'].value_counts().unstack().fillna(0).reindex(columns=kmeans_order)

# Plot the resampled KMeans topic counts over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=kmeans_resampled, markers=True, dashes=False, palette=kmeans_color_dict)
ax = plt.gca()

# Format x-axis
format_xaxis(ax, resample_freq, min_date, max_date)

plt.title('KMeans Topics Counts Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend(title='KMeans Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

########################################################################
### 4.10.5 Average Comment Score Over Time
########################################################################

# Resample and calculate the average comment score over the selected frequency
average_score_resampled = vermont_df.resample(resample_freq)['score'].mean()

# Plot the average comment score over time
plt.figure(figsize=standard_figsize)
average_score_resampled.plot(kind='line', marker='o', linewidth=2, color='teal')
ax = plt.gca()

# Format x-axis
format_xaxis(ax, resample_freq, min_date, max_date)

plt.title('Average Comment Score Over Time')
plt.xlabel('Date')
plt.ylabel('Average Score')
plt.tight_layout()
plt.show()

########################################################################
### Additional Plot: Average Political Leaning Confidence vs. Average Score
########################################################################

# Calculate average political leaning confidence and average score for combined plot
combined_resampled = pd.DataFrame({
    'Average Political Leaning Score': vermont_df.resample(resample_freq)['political_leaning_score'].mean(),
    'Average Comment Score': vermont_df.resample(resample_freq)['score'].mean()
})

plt.figure(figsize=standard_figsize)
sns.lineplot(data=combined_resampled, markers=True, dashes=False)
ax = plt.gca()

# Format x-axis
format_xaxis(ax, resample_freq, min_date, max_date)

plt.title('Average Political Leaning Confidence Score vs. Average Comment Score Over Time')
plt.xlabel('Date')
plt.ylabel('Average Values')
plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()





########################################################################
### 4.11 Correlation and Statistical Analysis
########################################################################

## This section examines correlations and dependencies between variables,
## including political leaning, sentiment intensity, and emotion categories.



########################################################################
### 4.11.1 Political Leaning vs. Sentiment Intensity
########################################################################

# Create a crosstab of political leaning vs. sentiment intensity for analysis
leaning_sentiment_crosstab = pd.crosstab(
    vermont_df['political_leaning'],
    vermont_df['sentiment_intensity'],
    rownames=['Political Leaning'],
    colnames=['Sentiment Intensity']
).reindex(index=leaning_order, columns=sentiment_order)

print("\nPolitical Leaning vs. Sentiment Intensity Crosstab:\n", leaning_sentiment_crosstab)

# Plot heatmap for the crosstab
plt.figure(figsize=(12, 8))
sns.heatmap(leaning_sentiment_crosstab, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Heatmap of Political Leaning vs. Sentiment Intensity')
plt.xlabel('Sentiment Intensity')
plt.ylabel('Political Leaning')
plt.tight_layout()
plt.show()

# Additional Analysis: Sentiment Confidence Scores by Political Leaning
plt.figure(figsize=standard_figsize)
sns.boxplot(
    data=vermont_df,
    x='political_leaning',
    y='sentiment_score',
    order=leaning_order,
    palette=leaning_color_dict
)
plt.title('Sentiment Confidence Scores by Political Leaning')
plt.xlabel('Political Leaning')
plt.ylabel('Sentiment Confidence Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



########################################################################
### 4.11.2 Political Leaning vs. Emotion
########################################################################

# Create a crosstab of political leaning vs. emotion categories for analysis
leaning_emotion_crosstab = pd.crosstab(
    vermont_df['political_leaning'],
    vermont_df['emotion'],
    rownames=['Political Leaning'],
    colnames=['Emotion']
).reindex(index=leaning_order, columns=emotion_order)

print("\nPolitical Leaning vs. Emotion Crosstab:\n", leaning_emotion_crosstab)

# Plot heatmap for the crosstab
plt.figure(figsize=(12, 8))
sns.heatmap(leaning_emotion_crosstab, annot=True, fmt='d', cmap='BuPu')
plt.title('Heatmap of Political Leaning vs. Emotion')
plt.xlabel('Emotion')
plt.ylabel('Political Leaning')
plt.tight_layout()
plt.show()

# Additional Analysis: Emotion Confidence Scores by Political Leaning
plt.figure(figsize=standard_figsize)
sns.boxplot(
    data=vermont_df,
    x='political_leaning',
    y='emotion_score',
    order=leaning_order,
    palette=leaning_color_dict
)
plt.title('Emotion Confidence Scores by Political Leaning')
plt.xlabel('Political Leaning')
plt.ylabel('Emotion Confidence Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()





########################################################################
### 4.11.3 Statistical Analysis: Chi-Squared Tests
########################################################################

# Perform a chi-squared test between political leaning and sentiment intensity
chi2_stat, p_val, dof, ex = chi2_contingency(leaning_sentiment_crosstab)
print(f"\nChi-Squared Test between Political Leaning and Sentiment Intensity:")
print(f"Chi2 Statistic: {chi2_stat:.2f}, p-value: {p_val:.4f}")

# Interpret the result
if p_val < 0.05:
    print("There is a significant association between Political Leaning and Sentiment Intensity.")
else:
    print("There is no significant association between Political Leaning and Sentiment Intensity.")

# Perform a chi-squared test between political leaning and emotion
chi2_stat, p_val, dof, ex = chi2_contingency(leaning_emotion_crosstab)
print(f"\nChi-Squared Test between Political Leaning and Emotion:")
print(f"Chi2 Statistic: {chi2_stat:.2f}, p-value: {p_val:.4f}")

# Interpret the result
if p_val < 0.05:
    print("There is a significant association between Political Leaning and Emotion.")
else:
    print("There is no significant association between Political Leaning and Emotion.")
    
    
    

########################################################################
### 4.11.4 Correlation Matrix for Confidence Scores
########################################################################

# Calculate and print the correlation matrix for confidence scores
confidence_scores = vermont_df[['emotion_score', 'bias_score', 'sentiment_score', 'political_leaning_score']]
correlation_matrix = confidence_scores.corr()

print("\nCorrelation Matrix for Confidence Scores:\n", correlation_matrix)

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Confidence Scores')
plt.tight_layout()
plt.show()

# Additional Scatter Plot: Emotion Score vs. Sentiment Score
plt.figure(figsize=standard_figsize)
sns.scatterplot(
    data=vermont_df,
    x='emotion_score',
    y='sentiment_score',
    hue='political_leaning',
    palette=leaning_color_dict,
    alpha=0.7,
    edgecolor=None
)
plt.title('Emotion Confidence Score vs. Sentiment Confidence Score')
plt.xlabel('Emotion Confidence Score')
plt.ylabel('Sentiment Confidence Score')
plt.legend(title='Political Leaning', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()





########################################################################
### 4.12 Correlation and Additional Scatter Plots
########################################################################

# Example of an additional scatter plot with consistent color palette
plt.figure(figsize=standard_figsize)
sns.scatterplot(
    data=vermont_df,
    x='sentiment_score',
    y='emotion_score',
    hue='political_leaning',
    palette=leaning_color_dict,
    alpha=0.7,
    edgecolor=None
)
plt.title('Emotion Confidence Score vs. Sentiment Confidence Score')
plt.xlabel('Emotion Confidence Score')
plt.ylabel('Sentiment Confidence Score')
plt.legend(title='Political Leaning', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()




########################################################################
### 4.13 t-SNE Visualization Colored by Various Categories
########################################################################

### 4.13.1 t-SNE Visualization Colored by Political Leaning

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=vis_df,
    x='x',
    y='y',
    hue='political_leaning',
    palette=leaning_color_dict,
    alpha=0.7,
    edgecolor=None
)
plt.title('t-SNE Visualization of Embeddings Colored by Political Leaning')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Political Leaning', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

### 4.13.2 t-SNE Visualization Colored by Emotion

plt.figure(figsize=standard_figsize)
sns.scatterplot(
    data=vis_df,
    x='x',
    y='y',
    hue='emotion',
    palette=emotion_color_dict,
    alpha=0.7,
    edgecolor=None
)
plt.title('t-SNE Visualization of Embeddings Colored by Emotion')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

### 4.13.3 t-SNE Visualization Colored by KMeans Topic

plt.figure(figsize=standard_figsize)
sns.scatterplot(
    data=vis_df,
    x='x',
    y='y',
    hue='kmeans_topic',
    palette=kmeans_palette,
    alpha=0.7,
    edgecolor=None
)
plt.title('t-SNE Visualization of Embeddings Colored by KMeans Topic')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='KMeans Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

### 4.13.4 t-SNE Visualization Colored by BERTopic Topic (Limited to Top 60)

# Filter to remove unwanted topics, if necessary
filtered_vis_df = vis_df[~vis_df['bertopic_topic_limited'].isin([-1, 'Other'])]

plt.figure(figsize=standard_figsize)
sns.scatterplot(
    data=filtered_vis_df,
    x='x',
    y='y',
    hue='bertopic_topic_limited',
    palette=bertopic_color_dict_limited,
    alpha=0.7,
    edgecolor=None,
    legend=False  # Legend removed as requested
)
plt.title('t-SNE Visualization of Embeddings Colored by BERTopic Topic (Top 60)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.tight_layout()
plt.show()

### 4.13.5 t-SNE Visualization Colored by Sentiment Confidence Score

plt.figure(figsize=standard_figsize)
sns.scatterplot(
    data=vis_df,
    x='x',
    y='y',
    hue='sentiment_score',
    palette='coolwarm',
    alpha=0.7,
    edgecolor=None
)
plt.title('t-SNE Visualization of Embeddings Colored by Sentiment Confidence Score')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Sentiment Confidence Score', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()




########################################################################
### 4.14 Additional Analysis with Confidence Scores
########################################################################

### 4.14.1 Filtering Based on Confidence Scores

# Filter comments with high confidence in sentiment intensity
filtered_df = vermont_df[vermont_df['sentiment_score'] >= 0.7].copy()
print(f"\nFiltered down to {len(filtered_df)} comments with Sentiment Score >= 0.7")

# Plot: Distribution of Sentiment Intensity in Filtered Data
plot_count(
    data=filtered_df,
    x='sentiment_intensity',
    order=sentiment_order,
    palette=sentiment_color_dict,
    title='Distribution of Sentiment Intensity in Filtered Vermont Reddit Comments (Confidence >= 0.7)',
    xlabel='Sentiment Intensity',
    ylabel='Count'
)

### 4.14.2 Weighted Analysis Using Confidence Scores

# Calculate weighted average emotion scores per political leaning
weighted_emotion = vermont_df.groupby('political_leaning').apply(
    lambda x: np.average(x['emotion_score'], weights=x['sentiment_score'])
).reset_index(name='weighted_emotion_score')

print("\nWeighted Average Emotion Score per Political Leaning:\n", weighted_emotion)

# Plot weighted average emotion scores by political leaning
plt.figure(figsize=standard_figsize)
sns.barplot(
    data=weighted_emotion,
    x='political_leaning',
    y='weighted_emotion_score',
    order=leaning_order,
    palette=leaning_color_dict
)
plt.title('Weighted Average Emotion Confidence Score per Political Leaning')
plt.xlabel('Political Leaning')
plt.ylabel('Weighted Average Emotion Confidence Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

### 4.14.3 Correlation Between Confidence Scores

# Calculate and plot the correlation matrix for confidence scores
correlation_matrix = confidence_scores.corr()

print("\nCorrelation Matrix for Confidence Scores:\n", correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Confidence Scores')
plt.tight_layout()
plt.show()

### 4.14.4 Sentiment and Emotion Confidence Scatter Plot

plt.figure(figsize=standard_figsize)
sns.scatterplot(
    data=vermont_df,
    x='sentiment_score',
    y='emotion_score',
    hue='political_leaning',
    palette=leaning_color_dict,
    alpha=0.7,
    edgecolor=None
)
plt.title('Emotion Confidence Score vs. Sentiment Confidence Score')
plt.xlabel('Emotion Confidence Score')
plt.ylabel('Sentiment Confidence Score')
plt.legend(title='Political Leaning', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



########################################################################
### 5. Conclusion
########################################################################

print("\nAnalysis Complete.")



