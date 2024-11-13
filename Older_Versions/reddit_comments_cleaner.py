# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 18:28:05 2024

@author: dforc
"""

# =============================================================================
# ## Imports and Packages
# =============================================================================
import pandas as pd
import re

# =============================================================================
# Notes:
# This script cleans the 'reddit_comments.csv' file for analysis.
# =============================================================================

# Path to the comments data file
comments_file_path = "reddit_comments.csv"

# =============================================================================
# ## Main Function
# =============================================================================
def main():
    # Load the comments data
    comments_data = pd.read_csv(comments_file_path)
    
    # Data cleaning steps
    comments_data = filter_na(comments_data)
    comments_data = gif_detection(comments_data)
    comments_data = clean_text(comments_data)
    comments_data = word_count(comments_data)
    comments_data = character_count(comments_data)
    comments_data = avg_word_length(comments_data)
    
    # Save cleaned data to a new CSV file
    comments_data.to_csv("reddit_comments_cleaned.csv", encoding='utf-8', index=False)
    print("Cleaning complete! Cleaned data saved as 'reddit_comments_cleaned.csv'.")

# =============================================================================
# ## Drop / Filter Missing and NA Function
# =============================================================================
def filter_na(data):
    """Drops rows with no body or author. Fills other missing data with 'NA'."""
    data = data.dropna(subset=['body', 'author'])
    data = data.fillna("NA")
    return data

# =============================================================================
#  ## GIF Detection Function
# =============================================================================
def gif_detection(data):
    """Determines if 'body' contains a shared gif. Creates 'has_gif' column."""
    data['has_gif'] = data['body'].apply(lambda x: "Yes" if "![gif]" in x or "giphy" in x else "No")
    return data

# =============================================================================
# ## 'body' Cleaning Function
# =============================================================================
def clean_text(data):
    """Cleans 'body' text of hyperlinks, usernames, emojis, punctuation."""
    clean_dict = {
        r'\bhttps://[^\s]+\b': '',      # Remove hyperlinks
        r'@[^\s]+': '',                 # Remove usernames
        r'\bRT\b': '',                  # Remove "RT"
        r'[^\w\s]': '',                 # Remove punctuation and emojis
        r'\s+': ' ',                    # Replace multiple spaces with single space
        r'^\s+|\s+$': '',               # Remove leading/trailing spaces
        'amp': 'and'                    # Handle encoding issue
    }
    
    # Apply cleaning dictionary
    data['body'] = data['body'].replace(clean_dict, regex=True)
    
    # Replace empty 'body' entries with "NA"
    data['body'] = data['body'].replace('', "NA", regex=True)
    
    print('\n Cleaned Text Data Preview: \n')
    print(data['body'].head())
    
    return data

# =============================================================================
# ## Clean Text Word Count Function
# =============================================================================
def word_count(data):
    """Counts the number of words in 'body'."""
    data['word_count_clean'] = data['body'].apply(lambda x: 0 if x == "NA" else len(x.split()))
    return data

# =============================================================================
# ## Count Characters Function
# =============================================================================
def character_count(data):
    """Counts the number of characters in 'body'."""
    data['char_count_clean'] = data['body'].apply(lambda x: 0 if x == "NA" else len(x))
    return data

# =============================================================================
# ## Calculate Average Word Length Function
# =============================================================================
def avg_word_length(data):
    """Calculates the average word length in 'body'."""
    data['avg_word_length'] = data.apply(
        lambda row: 0 if row['word_count_clean'] == 0 else round(row['char_count_clean'] / row['word_count_clean'], 3),
        axis=1
    )
    return data

# Entry point for the script
if __name__ == "__main__":
    main()