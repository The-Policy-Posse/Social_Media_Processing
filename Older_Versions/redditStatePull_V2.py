# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 23:46:02 2024

@author: dforc
"""

import os
import json
import time
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import List, Dict
from dotenv import load_dotenv
import praw
from praw.models import MoreComments
from tqdm import tqdm
import logging
import signal
import sys
import random

# Configure logging
logging.basicConfig(
    filename='reddit_scraper.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Load environment variables from .env file for USER_AGENT
load_dotenv('reddit_env.env')

# Constants
POSTS_CSV = 'reddit_posts.csv'
COMMENTS_CSV = 'reddit_comments.csv'
PROGRESS_FILE = 'completed_states.json'
MAX_POSTS_PER_SUBREDDIT = 600
TIME_FILTER = 'year'  # Top posts from the past year
THREADS = 12  # Total number of concurrent threads (4 groups × 3 keys each)
DELAY_PER_REQUEST = 0.7  # Seconds to wait between requests per client
API_KEYS_FILE = 'reddit_api_keys.json'  # JSON file containing API keys
BATCH_SIZE = 100  # Number of posts per batch for writing to CSV
MAX_MORE_COMMENTS = None  # Fetch all comments

# List of 50 U.S. state subreddits
STATE_SUBREDDITS = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
    'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
    'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
    'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
    'NewHampshire', 'NewJersey', 'NewMexico', 'NewYork', 'NorthCarolina', 'NorthDakota',
    'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'RhodeIsland', 'SouthCarolina',
    'SouthDakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
    'WestVirginia', 'Wisconsin', 'Wyoming'
]

# Global variable to hold completed states for signal handling
completed_states = {}

# Function to initialize CSV files with headers if they don't exist
def initialize_csv():
    if not os.path.exists(POSTS_CSV):
        with open(POSTS_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'post_id', 'state', 'title', 'selftext', 'created_utc',
                'score', 'url', 'num_comments', 'author'
            ])
            writer.writeheader()
    if not os.path.exists(COMMENTS_CSV):
        with open(COMMENTS_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'post_id', 'state', 'comment_id', 'body',
                'created_utc', 'score', 'author'
            ])
            writer.writeheader()

# Function to load completed states from progress file
def load_completed_states() -> Dict[str, Dict[str, int]]:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Initialize with all states not completed
        return {subreddit: {'posts': 0, 'comments': 0} for subreddit in STATE_SUBREDDITS}

# Function to save completed states to progress file
def save_completed_states(completed_states: Dict[str, Dict[str, int]]):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(completed_states, f, indent=4)

# Function to load API keys from JSON file
def load_api_keys() -> List[List[Dict[str, str]]]:
    if not os.path.exists(API_KEYS_FILE):
        raise FileNotFoundError(f"API keys file '{API_KEYS_FILE}' not found.")
    with open(API_KEYS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    groups = []
    for i in range(1, 5):  # reddit_group_1 to reddit_group_4
        group_key = f'reddit_group_{i}'
        group = data.get(group_key, [])
        if not group:
            logging.warning(f"{group_key} is empty or not found in '{API_KEYS_FILE}'.")
            print(f"Warning: {group_key} is empty or not found in '{API_KEYS_FILE}'.")
        groups.append(group)
    return groups  # List of groups, each group is a list of dicts with client_id and api_key

# Function to load USER_AGENT from environment variables
def load_user_agent() -> str:
    USER_AGENT = os.environ.get('PolicyPosseReddit_UserAgent')
    if not USER_AGENT:
        raise EnvironmentError("USER_AGENT not found in system environment variables. Please set 'PolicyPosseReddit_UserAgent'.")
    return USER_AGENT

# Class to manage Reddit clients and rate limiting
class RedditClientManager:
    def __init__(self, api_groups: List[List[Dict[str, str]]], user_agent: str, delay_per_request: float):
        self.api_groups = api_groups
        self.user_agent = user_agent
        self.delay_per_request = delay_per_request  # Delay in seconds between requests per client
        self.lock = threading.Lock()
        self.group_clients = []  # List of lists of tuples (Reddit instance, client_id)
        self.group_indices = []  # To keep track of next client per group
        self.last_request_times = []  # To track last request time per client
        self.initialize_clients()

    def initialize_clients(self):
        for group in self.api_groups:
            clients = []
            for api in group:
                reddit = praw.Reddit(
                    client_id=api['client_id'],
                    client_secret=api['api_key'],
                    user_agent=self.user_agent
                )
                clients.append((reddit, api['client_id']))  # Store tuple (Reddit instance, client_id)
            self.group_clients.append(clients)
            self.group_indices.append(0)  # Initialize client index for the group
            self.last_request_times.append([0 for _ in group])  # Initialize last request times

    def get_next_client(self, group_index: int) -> (praw.Reddit, str, int):
        with self.lock:
            clients = self.group_clients[group_index]
            if not clients:
                raise ValueError(f"No clients available in group {group_index}")
            reddit_instance, client_id = clients[self.group_indices[group_index]]
            client_idx = self.group_indices[group_index]
            # Log the usage
            logging.info(f"Using API key {client_id} for group {group_index}")
            # Update index for next time
            self.group_indices[group_index] = (self.group_indices[group_index] + 1) % len(clients)
            return reddit_instance, client_id, client_idx

    def enforce_delay(self, group_index: int, client_index: int):
        with self.lock:
            last_time = self.last_request_times[group_index][client_index]
            current_time = time.time()
            elapsed = current_time - last_time
            if elapsed < self.delay_per_request:
                wait_time = self.delay_per_request - elapsed
                time.sleep(wait_time)
            # Update last request time
            self.last_request_times[group_index][client_index] = time.time()

# Function to write a batch of posts to the posts CSV
def write_posts_batch(posts_batch: List[Dict[str, str]]):
    if not posts_batch:
        return
    with threading.Lock():
        with open(POSTS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'post_id', 'state', 'title', 'selftext', 'created_utc',
                'score', 'url', 'num_comments', 'author'
            ])
            writer.writerows(posts_batch)

# Function to write a batch of comments to the comments CSV
def write_comments_batch(comments_batch: List[Dict[str, str]]):
    if not comments_batch:
        return
    with threading.Lock():
        with open(COMMENTS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'post_id', 'state', 'comment_id', 'body',
                'created_utc', 'score', 'author'
            ])
            writer.writerows(comments_batch)

# Function to handle rate limiting with exponential backoff and jitter
def handle_rate_limit(retry_count: int):
    # Exponential backoff: 2^retry_count seconds plus jitter
    base_delay = 2 ** retry_count
    jitter = random.uniform(0, 1)
    delay = base_delay + jitter
    print(f"Rate limited. Sleeping for {delay:.2f} seconds.")
    logging.info(f"Rate limited. Sleeping for {delay:.2f} seconds.")
    time.sleep(delay)

# Function to fetch comments for a single submission
def fetch_comments(submission, subreddit_name, comments_batch: List[Dict[str, str]], client_manager: RedditClientManager, group_index: int, client_idx: int, retry_count=0, max_more_comments: int = None):
    try:
        # Enforce delay before making API requests
        client_manager.enforce_delay(group_index, client_idx)
        
        # Replace all MoreComments objects to fetch all comments
        submission.comments.replace_more(limit=max_more_comments)
        
        # Enforce delay after replace_more to respect rate limits
        client_manager.enforce_delay(group_index, client_idx)
        
        for comment in submission.comments.list():
            comment_data = {
                'post_id': submission.id,
                'state': subreddit_name,
                'comment_id': comment.id,
                'body': comment.body,
                'created_utc': datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).isoformat(),
                'score': comment.score,
                'author': str(comment.author) if comment.author else 'deleted'
            }
            comments_batch.append(comment_data)
    except praw.exceptions.APIException as api_exc:
        if api_exc.error_type == 'RATELIMIT':
            if retry_count < 5:
                logging.warning(f"Rate limit hit for group {group_index}, client {client_idx} when processing post {submission.id}.")
                handle_rate_limit(retry_count)
                fetch_comments(submission, subreddit_name, comments_batch, client_manager, group_index, client_idx, retry_count + 1, max_more_comments)
            else:
                print(f"Max retries exceeded for comments of post {submission.id} in {subreddit_name}. Skipping.")
                logging.error(f"Max retries exceeded for comments of post {submission.id} in {subreddit_name}. Skipping.")
        else:
            print(f"APIException for {subreddit_name}: {api_exc}")
            logging.error(f"APIException for {subreddit_name}: {api_exc}")
    except Exception as e:
        print(f"Error fetching comments for post {submission.id} in {subreddit_name}: {e}")
        logging.error(f"Error fetching comments for post {submission.id} in {subreddit_name}: {e}")

# Function to process a single subreddit (state)
def process_subreddit(subreddit_name: str, group_index: int, client_manager: RedditClientManager, completed_states: Dict[str, Dict[str, int]]):
    try:
        reddit_client, client_id, client_idx = client_manager.get_next_client(group_index)
        subreddit = reddit_client.subreddit(subreddit_name)
        
        # Collect all posts in memory
        posts_batch = []
        posts = []
        with tqdm(total=MAX_POSTS_PER_SUBREDDIT, desc=f'Fetching posts for {subreddit_name}', unit='posts') as pbar:
            try:
                for submission in subreddit.top(time_filter=TIME_FILTER, limit=MAX_POSTS_PER_SUBREDDIT):
                    post_data = {
                        'post_id': submission.id,
                        'state': subreddit_name,
                        'title': submission.title,
                        'selftext': submission.selftext,
                        'created_utc': datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat(),
                        'score': submission.score,
                        'url': submission.url,
                        'num_comments': submission.num_comments,
                        'author': str(submission.author) if submission.author else 'deleted'
                    }
                    posts_batch.append(post_data)
                    posts.append(submission)
                    pbar.update(1)
            except Exception as e:
                print(f"Error fetching posts for {subreddit_name}: {e}")
                logging.error(f"Error fetching posts for {subreddit_name}: {e}")
                raise e

        # Collect all comments in memory
        comments_batch = []
        with tqdm(total=len(posts), desc=f'Fetching comments for {subreddit_name}', unit='posts') as pbar_comments:
            with ThreadPoolExecutor(max_workers=3) as comment_executor:  # 3 threads per group
                futures = [
                    comment_executor.submit(
                        fetch_comments,
                        submission,
                        subreddit_name,
                        comments_batch,
                        client_manager,
                        group_index,
                        client_idx
                    )
                    for submission in posts
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                        pbar_comments.update(1)
                    except Exception as exc:
                        print(f"Exception during comment fetching: {exc}")
                        logging.error(f"Exception during comment fetching: {exc}")

        # After collecting all comments, write to CSV
        write_posts_batch(posts_batch)
        logging.info(f"Wrote {len(posts_batch)} posts for {subreddit_name} using API key {client_id}.")
        
        write_comments_batch(comments_batch)
        logging.info(f"Wrote {len(comments_batch)} comments for {subreddit_name} using API key {client_id}.")

        # Mark the state as complete
        completed_states[subreddit_name]['posts'] = MAX_POSTS_PER_SUBREDDIT
        completed_states[subreddit_name]['comments'] = len(comments_batch)
        save_completed_states(completed_states)
        print(f"Completed fetching posts and comments for {subreddit_name}")
        logging.info(f"Completed fetching posts and comments for {subreddit_name} using API key {client_id}.")

    except Exception as e:
        print(f"Failed to process subreddit {subreddit_name}: {e}")
        logging.error(f"Failed to process subreddit {subreddit_name}: {e}")

# Function to handle graceful shutdown
def signal_handler(sig, frame):
    print('Interrupt received, saving progress and exiting...')
    save_completed_states(completed_states)
    sys.exit(0)

# Main function
def main():
    global completed_states
    # Set up signal handling for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    initialize_csv()
    completed_states = load_completed_states()
    try:
        api_groups = load_api_keys()
    except FileNotFoundError as e:
        print(e)
        logging.error(e)
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing API keys JSON: {e}")
        logging.error(f"Error parsing API keys JSON: {e}")
        return

    try:
        user_agent = load_user_agent()
    except EnvironmentError as e:
        print(e)
        logging.error(e)
        return

    # Initialize RedditClientManager with delay_per_request
    client_manager = RedditClientManager(api_groups, user_agent, delay_per_request=DELAY_PER_REQUEST)

    # Assign each group to handle certain states
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        future_to_subreddit = {}
        group_indices = list(range(len(api_groups)))  # 0 to 3

        # Filter out completed states
        states_to_process = [
            state for state in STATE_SUBREDDITS
            if completed_states[state]['posts'] < MAX_POSTS_PER_SUBREDDIT or completed_states[state]['comments'] == 0
        ]

        # Initialize progress bars for overall progress
        overall_pbar = tqdm(total=len(states_to_process), desc='Overall Progress', unit='states')

        for subreddit in states_to_process:
            # Assign group in a round-robin fashion
            group_index = group_indices.pop(0)
            group_indices.append(group_index)

            future = executor.submit(
                process_subreddit,
                subreddit,
                group_index,
                client_manager,
                completed_states
            )
            future_to_subreddit[future] = subreddit

        for future in as_completed(future_to_subreddit):
            subreddit = future_to_subreddit[future]
            try:
                future.result()
            except Exception as exc:
                print(f"{subreddit} generated an exception: {exc}")
                logging.error(f"{subreddit} generated an exception: {exc}")
            finally:
                overall_pbar.update(1)

        overall_pbar.close()

    print("Scraping completed for all states.")
    logging.info("Scraping completed for all states.")

if __name__ == '__main__':
    main()