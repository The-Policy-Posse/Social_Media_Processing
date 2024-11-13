# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 02:35:09 2024

@author: dforc
"""

import os
import re
import aiohttp
import asyncio
from aiohttp import ClientSession
from tqdm import tqdm
import pandas as pd
import nest_asyncio
import time

nest_asyncio.apply()

# Load the data
posts_df = pd.read_csv('reddit_posts.csv')

# Define the image extensions and apply cleaning
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
posts_df['url'] = posts_df['url'].apply(lambda x: re.sub(r"\\+", "/", str(x)).strip())

# Filter for image URLs
posts_df['image_url'] = posts_df['url'].apply(lambda x: x if x.lower().endswith(image_extensions) else None)
image_urls = posts_df[['post_id', 'image_url']].dropna().values  # Get (post_id, image_url) pairs

# Create a directory for downloaded images
os.makedirs("post_images", exist_ok=True)

# Asynchronous function to download images with retry logic and backoff for rate limits
async def download_image(post_id, url, session, pbar, retries=8, initial_delay=1):
    image_name = os.path.join("post_images", f"{post_id}.jpg")  # Save with post_id as filename
    delay = initial_delay
    for attempt in range(retries):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(image_name, "wb") as f:
                        f.write(await response.read())
                    pbar.update(1)
                    return  # Successful download
                elif response.status == 429:  # Rate limit
                    print(f"Rate limit hit for {url}, retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay *= 2  # Increase delay for the next attempt
                else:
                    print(f"Failed to download {url} with status {response.status}, skipping.")
                    break
        except Exception as e:
            print(f"Error downloading {url}: {e}, retrying in {delay} seconds...")
            await asyncio.sleep(delay)
            delay *= 2  # Increase delay for the next attempt
    pbar.update(1)  # Update progress even if all attempts fail

# Main async function to manage the download process
async def download_all_images(post_id_url_pairs):
    async with ClientSession() as session:
        with tqdm(total=len(post_id_url_pairs), desc="Downloading Images") as pbar:
            tasks = [download_image(post_id, url, session, pbar) for post_id, url in post_id_url_pairs]
            await asyncio.gather(*tasks)

# Run the download process
asyncio.run(download_all_images(image_urls))