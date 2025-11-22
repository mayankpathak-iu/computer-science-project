# Importing all libraries
import time
from dotenv import load_dotenv
import os
import re
import tweepy
import nltk
import requests
import spacy
from rake_nltk import Rake
from typing import List, Dict, Any

############################### Step 1 - Accepting Tweet from user and fetching Tweet ######################################

# Loading the env variables (API tokens)
r'''
load_dotenv()  

bearer_token = os.getenv("bearer_token")
client = tweepy.Client(bearer_token, wait_on_rate_limit=True)

def extract_tweet_id(url):
    match = re.search(r"status/(\d+)", url)
    return match.group(1) if match else None

# Asking tweet input from the user

while True:

    url = input("Please Enter the tweet link: ")
    tweetid = extract_tweet_id(url)

    if url.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    if not tweetid:
        print("\nError: That doesn't look like a valid tweet link.")
        print("Please paste the full URL (e.g., https://twitter.com/user/status/12345)\n")
        continue  # Ask for the link again

    try:

        # This endpoint/method returns the information about the Tweet(s)
        response = client.get_tweets(tweetid, tweet_fields=["created_at"])

        if response.data:
            print(response,response.data[0].created_at)
            tweet_created_date = response.data[0].created_at

        else:
            # The ID format was correct, but the tweet is unavailable
            # (e.g., deleted, private, or just a wrong ID)
            print("\nError: Tweet not found. It may be deleted, private, or the link is incorrect.\n")
            continue

    except Exception as e:
            # Catch any other unexpected errors (e.g., network issues)
            print(f"\nAn unexpected error occurred: {e}\n")
            # Ask for the link again


############################### STEP 2 - Creating two search queries for Google News #############################

'''

def clean_query_text(text):
    text = re.sub(r"[^0-9A-Za-z\s\-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dedupe_words_preserve_order(text):
    seen = set()
    result_words = []
    for w in text.split():
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            result_words.append(w)
    return " ".join(result_words)



def generate_search_queries(tweet_text):
    """
    Generate up to `max_queries` optimized search queries from tweet text.
    Uses:
      - Named Entities (PERSON, ORG, GPE, LOC, EVENT)
      - Noun phrases
      - RAKE keywords
    """
    nltk.download('stopwords')
    nltk.download('punkt_tab')      
    # Load spaCy English model
    queries = []
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(tweet_text)

    # Method 1: Extract Named Entities (people, orgs, places, events)
    entities = [
        ent.text
        for ent in doc.ents
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "EVENT"]
    ]


    # Method 2: Extract Noun Phrases
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]

    # Method 3: Extract Keywords using RAKE
    rake = Rake()
    rake.extract_keywords_from_text(tweet_text)
    keywords = rake.get_ranked_phrases()[:5]

    if entities and keywords:
        raw_q1 = " ".join(entities[:2]) + " " + " ".join(keywords[:1])
        q1 = dedupe_words_preserve_order(clean_query_text(raw_q1))
        queries.append(q1)
    
    # Query 3: Top keywords
    if keywords:
        raw_q2 = " ".join(keywords[:3])
        q2 = dedupe_words_preserve_order(clean_query_text(raw_q2))
        queries.append(q2)

    return queries



txt = "Court finds Nnamdi Kanu, leader of the Indigenous People of Biafra (IPOB), guilty of inciting violence during the #EndSARS protests, which led to the killing of security personnel and destruction of government properties in Lagos"
lst = generate_search_queries(txt)
print(lst)


############################### STEP 3 - Google news API  #################################################################

news_bearer_token = os.getenv("bearer_token")
