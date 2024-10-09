"""
## config.py
## Policy Posse Reddit API Credentials Configuration

## 10/8/24

This script is responsible for loading environment variables required for
database access and Reddit API credentials. These variables are sourced 
from a .env file using the dotenv library.

The .env file must contain the following environment variables:
- Database credentials: DB_USER, DB_PASSWORD, DB_NAME, DB_HOST, DB_PORT
- Reddit API credentials: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

@author: dforc
"""

# =============================================================================
# ## Imports
# =============================================================================
from dotenv import load_dotenv
import os


# =============================================================================
# ## Load Environment Variables (From creds.env)
# =============================================================================
load_dotenv(dotenv_path="creds.env")


# =============================================================================
# ## Database Credentials
# =============================================================================
## These are used to connect to the PostgreSQL database
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')


# =============================================================================
# ## Reddit API Credentials
# =============================================================================
## These are used to interact with the Reddit API for scraping and data collection purposes
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')
