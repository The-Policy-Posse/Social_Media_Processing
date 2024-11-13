# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:59:22 2024

@author: dforc

Description:
    This script scrapes posts and comments from state-specific subreddits on Reddit for data analysis purposes. 
    Utilizing multiple API keys, it maintains concurrency and manages rate limits efficiently across requests. 
    The script collects top posts from each subreddit, retrieves their comments, and saves all data to CSV files. 
    Logging is configured to capture any errors encountered during the scraping process.

Segment Overview:

    1. Segment 1: Setup and Load API Keys
       - Loads API keys from a JSON file and organizes them into a list. Each API key is assigned a Reddit client 
         and rate limiter for handling requests while adhering to API rate limits.

    2. Segment 2: Rate Limiter and Concurrency Setup
       - Configures a semaphore to manage the number of concurrent requests, preventing overload on the API.

    3. Segment 3: State Tracking
       - Tracks which subreddits (representing U.S. states) have been fully scraped. Completed states are stored 
         in a file to allow resumption from the last state in case of interruptions.

    4. Segment 4: Data Writing Functions
       - Initializes CSV files with headers for storing posts and comments if they do not already exist. Functions 
         are included to append new data to the files during runtime.

    5. Segment 5: Fetch Top Posts with Rate Limiting and Retry Logic
       - Scrapes the top posts from a specified subreddit within the past year. This segment includes retry logic 
         to handle rate limits and other exceptions, ensuring a robust data collection process.

    6. Segment 6: Fetch Comments for Each Post with Retry Logic
       - Retrieves all comments for a given post, including nested replies. Implements retry logic to manage rate 
         limits and handle potential errors such as server issues or inaccessible posts.

    7. Segment 7: Subreddit Scraping with Enhanced Rate Limit Handling
       - Combines post and comment scraping for each subreddit assigned to a Reddit client. Progress bars display 
         scraping status, and rate limits across clients are managed with shared tracking.

    8. Segment 8: Main Entry Point
       - The main function initializes CSV files, loads completed states, assigns subreddits to Reddit clients, 
         and manages the asynchronous scraping process for each state-specific subreddit. Upon completion, data is 
         saved, and processing time is displayed.

    9. Segment 9: Script Execution
       - The script entry point calls the main asynchronous function when the script is executed directly, initiating 
         the entire scraping process.
"""


import os
import json
import asyncpraw
import asyncio
import pandas as pd
from tqdm.asyncio import tqdm
from asyncio import Semaphore, Lock
import nest_asyncio
from aiolimiter import AsyncLimiter
import time
import random
import logging
from logging.handlers import RotatingFileHandler
from prawcore.exceptions import NotFound, Forbidden, ServerError, TooManyRequests

# Apply nest_asyncio to allow nested event loops (useful in some environments)
nest_asyncio.apply()

# Configure logging to write errors to a file and print critical issues to console
logger = logging.getLogger('reddit_scraper')
logger.setLevel(logging.DEBUG)

# Create handlers
file_handler = RotatingFileHandler('reddit_scraper.log', maxBytes=5*1024*1024, backupCount=2)
file_handler.setLevel(logging.WARNING)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Show warnings and errors on console

# Create formatters and add them to handlers
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Load User Agent from environment or define it directly
USER_AGENT = os.environ.get('PolicyPosseReddit_UserAgent', 'YourAppName/Version')

########################################################################
## Segment 1: Setup and load API keys
########################################################################

def load_reddit_api_keys(env_file):
    """
    Load API keys from a JSON file.
    """
    with open(env_file, 'r') as file:
        api_key_data = json.load(file)
    api_keys = []
    for group_keys in api_key_data.values():
        api_keys.extend(group_keys)
    return api_keys

def initialize_reddit_clients(api_keys):
    """
    Create a Reddit instance and rate limiter for each API key.
    """
    clients = []
    for key_pair in api_keys:
        reddit = asyncpraw.Reddit(
            client_id=key_pair['client_id'],
            client_secret=key_pair['api_key'],
            user_agent=USER_AGENT,
            check_for_async=False  # Important for compatibility
        )
        rate_limiter = AsyncLimiter(95, 60)  # Limit each key to 95 requests per 60 seconds
        clients.append({'reddit': reddit, 'rate_limiter': rate_limiter, 'key_pair': key_pair})
    return clients

def assign_subreddits_to_clients(subreddits, clients):
    """
    Distribute subreddits evenly among Reddit clients.
    """
    for idx, subreddit in enumerate(subreddits):
        client = clients[idx % len(clients)]
        if 'subreddits' not in client:
            client['subreddits'] = []
        client['subreddits'].append(subreddit)
    return clients

########################################################################
## Segment 2: Rate limiter and concurrency setup
########################################################################

# Concurrency control
CONCURRENT_REQUESTS = 2  # Further reduced to prevent overwhelming the API
semaphore = Semaphore(CONCURRENT_REQUESTS)

########################################################################
## Segment 3: State Tracking
########################################################################

COMPLETED_STATES_FILE = 'completed_states.txt'

def load_completed_states():
    """
    Load the list of completed states from a file.
    """
    if not os.path.exists(COMPLETED_STATES_FILE):
        return set()
    with open(COMPLETED_STATES_FILE, 'r') as file:
        completed = {line.strip().lower() for line in file if line.strip()}
    return completed

def save_completed_state(state):
    """
    Append a completed state to the completed states file.
    """
    with open(COMPLETED_STATES_FILE, 'a') as file:
        file.write(f"{state.lower()}\n")

########################################################################
## Segment 4: Data Writing Functions
########################################################################

import csv

def initialize_csv_files():
    """
    Initialize the CSV files with headers if they don't exist.
    """
    if not os.path.exists('reddit_posts.csv'):
        with open('reddit_posts.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=[
                'post_id', 'state', 'title', 'selftext', 'created_utc',
                'score', 'url', 'num_comments', 'author'
            ])
            writer.writeheader()
    
    if not os.path.exists('reddit_comments.csv'):
        with open('reddit_comments.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=[
                'post_id', 'state', 'comment_id', 'body', 'created_utc',
                'score', 'author'
            ])
            writer.writeheader()

async def write_posts_to_csv(posts):
    """
    Append posts to the reddit_posts.csv file.
    """
    if not posts:
        return
    df = pd.DataFrame(posts)
    df = df.astype(str)  # Ensure all data is string to prevent encoding issues
    df.to_csv('reddit_posts.csv', mode='a', index=False, header=False)

async def write_comments_to_csv(comments):
    """
    Append comments to the reddit_comments.csv file.
    """
    if not comments:
        return
    df = pd.DataFrame(comments)
    df = df.astype(str)  # Ensure all data is string to prevent encoding issues
    df.to_csv('reddit_comments.csv', mode='a', index=False, header=False)

########################################################################
## Segment 5: Fetch top posts with rate limiting and retry logic
########################################################################

async def fetch_posts(client, subreddit_name, total_limit=600, batch_size=100, max_retries=3):
    """
    Fetch top posts from a subreddit using a single Reddit client.
    This function fetches posts in batches to reduce the number of API calls.
    """
    reddit = client['reddit']
    rate_limiter = client['rate_limiter']
    posts = []
    posts_fetched = 0

    try:
        async with rate_limiter:
            subreddit = await reddit.subreddit(subreddit_name)

        while posts_fetched < total_limit:
            try:
                async with rate_limiter:
                    submissions = subreddit.top(time_filter='year', limit=batch_size)

                async for submission in submissions:
                    post_data = {
                        'post_id': submission.id,
                        'state': subreddit_name,
                        'title': submission.title,
                        'selftext': submission.selftext,
                        'created_utc': submission.created_utc,
                        'score': submission.score,
                        'url': submission.url,
                        'num_comments': submission.num_comments,
                        'author': str(submission.author) if submission.author else 'deleted'
                    }
                    posts.append(post_data)
                    posts_fetched += 1
                    await asyncio.sleep(0.67)  # Delay between post fetches

                    if posts_fetched >= total_limit:
                        break

            except TooManyRequests:
                wait_time = 5 + random.uniform(0, 1)
                logger.warning(f"Rate limit hit for r/{subreddit_name} on post fetch. Retrying in {wait_time:.2f} seconds.")
                await asyncio.sleep(wait_time)

            except Exception as e:
                logger.error(f"Error fetching posts from r/{subreddit_name}: {type(e).__name__}: {e}")
                break

    except Exception as e:
        logger.error(f"Error initializing subreddit {subreddit_name}: {type(e).__name__}: {e}")

    # Introduce a cooldown after fetching posts to allow rate limits to reset
    await asyncio.sleep(60)  # 1-minute cooldown
    return posts

########################################################################
## Segment 6: Scrape comments with request tracking and backoff
########################################################################

async def fetch_comments_with_delay(client, post_id, subreddit_name, max_retries=5):
    """
    Fetch comments for a specific post with a delay between each call
    to ensure each API key does not exceed 90 requests per minute.
    """
    reddit = client['reddit']
    rate_limiter = client['rate_limiter']
    comments_data = []
    attempt = 0  # To keep track of retry attempts

    while attempt < max_retries:
        try:
            # Fetch the submission details
            async with rate_limiter:
                submission = await reddit.submission(id=post_id)
                await asyncio.sleep(0.67)  # 0.67-second delay to keep within 90 requests per minute

            # Fetch all comments for the submission
            await submission.comments.replace_more(limit=None)
            all_comments = submission.comments.list()

            for comment in tqdm(all_comments, desc=f"Fetching comments for post {post_id}", leave=False):
                comments_data.append({
                    'post_id': post_id,
                    'state': subreddit_name,
                    'comment_id': comment.id,
                    'body': comment.body,
                    'created_utc': comment.created_utc,
                    'score': comment.score,
                    'author': str(comment.author) if comment.author else 'deleted'
                })
                await asyncio.sleep(0.67)  # Delay after each comment fetch

            # Exit loop if fetching is successful
            break

        except TooManyRequests as e:
            # Handle rate limit errors with exponential backoff
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Rate limit hit for post {post_id} in r/{subreddit_name}. Retrying in {wait_time:.2f} seconds.")
            await asyncio.sleep(wait_time)
            attempt += 1  # Increment retry counter

        except Exception as e:
            logger.error(f"Error fetching comments for post {post_id} in r/{subreddit_name}: {e}")
            break

    return comments_data

########################################################################
## Segment 7: Subreddit scraping with enhanced rate limit handling
########################################################################

async def scrape_subreddits_for_client(client, post_limit=600, completed_states=set(), rate_limited_clients=None, rate_limited_lock=None, total_clients=0):
    """
    Process assigned subreddits for a specific Reddit client.
    """
    reddit = client['reddit']
    rate_limiter = client['rate_limiter']
    subreddits = client.get('subreddits', [])
    for subreddit_name in subreddits:
        if subreddit_name.lower() in completed_states:
            print(f"Skipping already completed subreddit: r/{subreddit_name}", flush=True)
            continue
        print(f"Fetching top posts from r/{subreddit_name}", flush=True)
        posts = await fetch_posts(client, subreddit_name, total_limit=post_limit)
        await write_posts_to_csv(posts)
        
        # Create comment-fetching tasks with semaphore to control concurrency
        comment_tasks = [
            fetch_comments_with_delay(
                client,
                post['post_id'],
                subreddit_name,
                max_retries=5
            ) for post in posts
        ]
        
        # Use tqdm to show progress
        for task in tqdm(asyncio.as_completed(comment_tasks), total=len(comment_tasks), desc=f"Fetching comments for r/{subreddit_name}"):
            comments = await task
            await write_comments_to_csv(comments)
        
        # Mark the subreddit as completed only after all posts and comments are written
        save_completed_state(subreddit_name)
        
        # Fancy Console Output
        print("\n" + "*" * 50)
        print(f"****  {subreddit_name.upper()} COMPLETED  ****")
        print("*" * 50 + "\n")
        
        print(f"Completed scraping subreddit: r/{subreddit_name}", flush=True)
    
    # Close the Reddit instance to prevent session issues
    await reddit.close()

########################################################################
## Segment 8: Main Entry Point
########################################################################

async def main():
    """
    Core function to load API keys, initialize Reddit clients, and execute the scraping process.
    """
    # Initialize CSV files
    initialize_csv_files()
    
    # Load completed states
    completed_states = load_completed_states()
    
    # Load API keys and initialize clients
    api_keys = load_reddit_api_keys('reddit_api_keys.env')
    clients = initialize_reddit_clients(api_keys)
    
    # Define state subreddits
    state_subreddits = [
        'alabama', 'alaska', 'arizona', 'arkansas', 'california',
        'colorado', 'connecticut', 'delaware', 'florida', 'georgia',
        'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas',
        'kentucky', 'louisiana', 'maine', 'maryland', 'massachusetts',
        'michigan', 'minnesota', 'mississippi', 'missouri', 'montana',
        'nebraska', 'nevada', 'newhampshire', 'newjersey', 'newmexico',
        'newyork', 'northcarolina', 'northdakota', 'ohio', 'oklahoma',
        'oregon', 'pennsylvania', 'rhodeisland', 'southcarolina', 'southdakota',
        'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington',
        'westvirginia', 'wisconsin', 'wyoming'
    ]
    
    # Assign subreddits to clients
    clients = assign_subreddits_to_clients(state_subreddits, clients)
    
    # Total number of clients
    total_clients = len(clients)
    
    print("Starting the scraping process...", flush=True)
    
    # Shared set and lock to track rate-limited clients
    rate_limited_clients = set()
    rate_limited_lock = Lock()
    
    # Create tasks for each client
    tasks = [
        scrape_subreddits_for_client(
            client,
            post_limit=600,
            completed_states=completed_states,
            rate_limited_clients=rate_limited_clients,
            rate_limited_lock=rate_limited_lock,
            total_clients=total_clients
        ) for client in clients
    ]
    
    # Run tasks concurrently
    await asyncio.gather(*tasks)
    
    print("Scraping completed.", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
    
    
    
    
    
    
    