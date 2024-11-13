# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 20:29:39 2024

@author: dforc
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import math

# Load processed data
vermont_df = pd.read_csv('processed_vermont_comments.csv', parse_dates=['created_utc'])

# Convert 'created_utc' to datetime for resampling
vermont_df['created_utc'] = pd.to_datetime(vermont_df['created_utc'], errors='coerce')
vermont_df = vermont_df.dropna(subset=['created_utc'])  # Drop rows where date parsing failed

# Set up the list to store each plot's image
images = []

# Helper function to save plots to an in-memory image with a fixed size
def save_plot_to_memory():
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', dpi=300, bbox_inches='tight')  # Use bbox_inches to prevent cutting off labels
    buf.seek(0)
    img = Image.open(buf).resize((800, 600))  # Resize all images to the same size for consistency
    images.append(img)
    buf.close()
    plt.close()

# Step 1: Monthly Sentiment Trends (Normalized)
vermont_df['month'] = vermont_df['created_utc'].dt.to_period('M')
monthly_sentiment = vermont_df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
monthly_sentiment_normalized = monthly_sentiment.div(monthly_sentiment.sum(axis=1), axis=0)

# Plot and save to memory
monthly_sentiment_normalized.plot(kind='line', title="Normalized Monthly Sentiment Trends", ylabel="Proportion of Comments", xlabel="Month")
save_plot_to_memory()

# Step 2: Weekly Sentiment Trends (Rolling Average)
vermont_df['week'] = vermont_df['created_utc'].dt.to_period('W')
weekly_sentiment = vermont_df.groupby(['week', 'sentiment']).size().unstack(fill_value=0)
weekly_sentiment_normalized = weekly_sentiment.div(weekly_sentiment.sum(axis=1), axis=0)
weekly_sentiment_rolling = weekly_sentiment_normalized.rolling(window=4, min_periods=1).mean()

# Plot and save to memory
weekly_sentiment_rolling.plot(kind='line', title="4-Week Rolling Average of Normalized Weekly Sentiment Trends", ylabel="Proportion of Comments", xlabel="Week")
save_plot_to_memory()

# Step 3: Day of the Week Sentiment Analysis
vermont_df['day_of_week'] = vermont_df['created_utc'].dt.day_name()
weekday_sentiment = vermont_df.groupby(['day_of_week', 'sentiment']).size().unstack(fill_value=0)
weekday_sentiment = weekday_sentiment.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Plot and save to memory
weekday_sentiment.plot(kind='bar', stacked=True, title="Sentiment by Day of the Week", ylabel="Number of Comments")
plt.xticks(rotation=45)
save_plot_to_memory()

# Additional plot: Pie Chart of Overall Sentiment Distribution
sentiment_counts = vermont_df['sentiment'].value_counts()
sentiment_counts.plot(kind='pie', autopct='%1.1f%%', title="Overall Sentiment Distribution")
save_plot_to_memory()

# Additional plot: Log-Transformed Comment Length by Sentiment
# Fill missing values with an empty string before calculating comment length
vermont_df['comment_length'] = vermont_df['cleaned_for_transformer'].fillna('').apply(len)
vermont_df['log_comment_length'] = vermont_df['comment_length'].apply(lambda x: math.log1p(x))  # log(1 + x) to handle zero values

# Plot the log-transformed comment length by sentiment
sns.boxplot(data=vermont_df, x='sentiment', y='log_comment_length')
plt.title("Log-Transformed Comment Length by Sentiment")
plt.ylabel("Log Comment Length (Standardized)")
save_plot_to_memory()

# Combine saved plots into a single 2-column layout image using Pillow
cols = 2
rows = math.ceil(len(images) / cols)
width, height = images[0].size
combined_image = Image.new('RGB', (width * cols, height * rows), (255, 255, 255))

# Paste each plot image onto the combined canvas in a 2-column format
for index, img in enumerate(images):
    x = (index % cols) * width
    y = (index // cols) * height
    combined_image.paste(img, (x, y))

# Save the final combined image as a single JPEG
combined_image.save("sentiment_analysis_report_combined.jpg", format="jpeg", quality=95)

print("Sentiment analysis report saved as 'sentiment_analysis_report_combined.jpg'.")