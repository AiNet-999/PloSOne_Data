!pip install vaderSentiment
!mkdir /content/2016-brexit
!tar -xvf /content/drive/MyDrive/2016-brexit.tar -C /content/2016-brexit

import os
import subprocess
import json
import csv
import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

folder_path = '/content/2015-refugeeswelcome'

for filename in os.listdir(folder_path):
    if filename.endswith(".bz2"):
        file_path = os.path.join(folder_path, filename)
        folder_name = filename[:-4]
        output_folder = os.path.join(folder_path, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        output_file_path = os.path.join(output_folder, folder_name)
        with open(output_file_path, 'wb') as output_file:
            subprocess.run(['bzip2', '-dc', file_path], stdout=output_file)

print("Decompression complete.")

folder_path = '/content/2015-refugeeswelcome/2015-refugeeswelcome'
output_csv = '/content/2015-refugeeswelcome.csv'
analyzer = SentimentIntensityAnalyzer()

def clean_tweet(tweet):
    tweet = re.sub(r"(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet

def analyze_sentiment_vader(tweet):
    vader_scores = analyzer.polarity_scores(tweet)
    compound_score = vader_scores['compound']
    if compound_score > 0:
        return 1
    elif compound_score < 0:
        return -1
    else:
        return 0

def process_file(file_path):
    total_tweets = 0
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    tweet = item.get('text', None)
                    if tweet:
                        cleaned_tweet = clean_tweet(tweet)
                        sentiment = analyze_sentiment_vader(cleaned_tweet)
                        total_tweets += 1
                        if sentiment == 1:
                            positive_count += 1
                        elif sentiment == -1:
                            negative_count += 1
                        else:
                            neutral_count += 1
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return None
    if total_tweets == 0:
        return None
    if neutral_count > positive_count and neutral_count > negative_count:
        overall_sentiment = 0
    elif positive_count > negative_count:
        overall_sentiment = 1
    else:
        overall_sentiment = -1
    return overall_sentiment

subfolders = sorted([f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))])

with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Date", "Overall Sentiment"])
    for folder_name in subfolders:
        folder_full_path = os.path.join(folder_path, folder_name)
        file_path = os.path.join(folder_full_path, folder_name)
        overall_sentiment = process_file(file_path)
        if overall_sentiment is not None:
            csv_writer.writerow([folder_name, overall_sentiment])

print(f"Daily sentiment has been saved to {output_csv}")
