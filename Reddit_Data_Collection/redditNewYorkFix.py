# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:24:49 2024

@author: dforc
"""

import asyncio
import asyncpraw
import pandas as pd
import json
import aiohttp
from tqdm.asyncio import tqdm
import os
from dotenv import load_dotenv
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load user agent from .env file
load_dotenv('reddit_env.env')
user_agent = os.getenv('PolicyPosseReddit_UserAgent')

# Load Reddit API keys from JSON file
with open('reddit_api_keys.json') as f:
    reddit_api_keys = json.load(f)

# Constants for rate limiting
MAX_REQUESTS_PER_MINUTE = 60  # Adjust as per Reddit's API rate limits
REQUEST_INTERVAL = 60 / MAX_REQUESTS_PER_MINUTE  # Time between requests per key

# Helper function to initialize Reddit clients
async def initialize_reddit_clients():
    clients = []
    for group in reddit_api_keys.values():
        for key_info in group:
            session = aiohttp.ClientSession()
            reddit = asyncpraw.Reddit(
                client_id=key_info["client_id"],
                client_secret=key_info["api_key"],
                user_agent=user_agent,
                requestor_kwargs={"session": session}
            )
            clients.append((reddit, session))  # Add both Reddit client and session for later closing
    return clients

# Function to retrieve posts asynchronously
async def fetch_posts_for_subreddit(reddit_client, subreddit_name, max_posts=600):
    try:
        subreddit = await reddit_client.subreddit(subreddit_name)
        posts = []
        async for submission in subreddit.top(limit=max_posts, time_filter="year"):
            posts.append({
                'post_id': submission.id,
                'state': subreddit_name,
                'title': submission.title,
                'selftext': submission.selftext,
                'created_utc': submission.created_utc,
                'score': submission.score,
                'url': submission.url,
                'num_comments': submission.num_comments,
                'author': str(submission.author) if submission.author else 'deleted'
            })
        return posts
    except Exception as e:
        print(f"Error fetching posts for {subreddit_name}: {e}")
        return []

# Main function to manage scraping with multiple API keys
async def main(subreddits):
    all_posts = []
    reddit_clients = await initialize_reddit_clients()  # List of (Reddit client, session) tuples
    num_clients = len(reddit_clients)

    # Use asyncio.gather to fetch posts in parallel
    tasks = []
    for reddit_client, _ in reddit_clients:
        task = fetch_posts_for_subreddit(reddit_client, subreddits[0])
        tasks.append(task)

    # Collect results from all clients
    results = await asyncio.gather(*tasks)

    # Combine all posts
    for posts in results:
        all_posts.extend(posts)

    # Remove duplicate posts based on 'post_id'
    all_posts = {post['post_id']: post for post in all_posts}.values()

    # Save all posts to a CSV file
    df = pd.DataFrame(all_posts)
    df.to_csv('reddit_posts.csv', index=False)
    print("Scraping completed and data saved to reddit_posts.csv.")

    # Close all Reddit client sessions
    for _, session in reddit_clients:
        await session.close()

# List of state subreddits to scrape
state_subreddits = ["NewYork"]

# Wrapper function to start the main function in an asyncio event loop
def run_main():
    asyncio.run(main(state_subreddits))

# Execute the wrapper function
if __name__ == "__main__":
    run_main()