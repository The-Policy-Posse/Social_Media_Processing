# -*- coding: utf-8 -*-
"""
## Policy Posse Reddit API Logger
## Logging Setup for Reddit Scraping Application

This script sets up logging for the Reddit scraping application. It configures both
console and file logging, ensuring detailed logs are written to a rotating file while
higher-level logs are printed to the console. The log messages include timestamps,
log levels, and messages, making it easier to trace application activity and debug issues.

The rotating file handler ensures that log files don't grow too large, with a limit of
10 MB per log file and up to 5 backup log files being retained.

@author: dforc
"""


# =============================================================================
# ## Imports
# =============================================================================
import logging
from logging.handlers import RotatingFileHandler 


# =============================================================================
# ## Create Logger
# =============================================================================
## Create a logger object for the Reddit scraper
logger = logging.getLogger('reddit_scraper')  
logger.setLevel(logging.DEBUG)


# =============================================================================
# ## Create Handlers
# =============================================================================
## Create a console handler to log messages to the terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  ## Set to INFO for general messages, change to DEBUG for more details

## Create a rotating file handler to log messages to a file with a maximum size and backup
## 10 MB max per file
file_handler = RotatingFileHandler('reddit_scraper.log', maxBytes=10*1024*1024, backupCount=5)
file_handler.setLevel(logging.DEBUG)


# =============================================================================
# ## Create Formatters and Add to Handlers
# =============================================================================
## Define the log format with timestamp, log level, and message
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

## Add the formatter to both the console and file handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


# =============================================================================
# ## Add Handlers to Logger
# =============================================================================
## Attach the handlers (console and file) to the logger object
logger.addHandler(console_handler)
logger.addHandler(file_handler)
