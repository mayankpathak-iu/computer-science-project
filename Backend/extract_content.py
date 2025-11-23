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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

############################### Step 1 - Accepting Tweet from user and fetching Tweet ######################################
load_dotenv()  

# Loading the env variables (API tokens)
r'''

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
search_queries = generate_search_queries(txt)
#print(search_queries)


############################### STEP 3 - Google news API  #################################################################

news_bearer_token = os.getenv("news_bearer_token")
SERPAPI_ENDPOINT = "https://serpapi.com/search"

# Hardcoded config
DEFAULT_HL = "en"               # language
DEFAULT_GL = "us"               # country
DEFAULT_MAX_RESULTS = 5         # top results per query
DEFAULT_RETRIES = 3             # retry attempts
DEFAULT_BACKOFF = 2.0           # seconds between retries


def serpapi_google_news_search(query):
    """
    Minimal SerpAPI Google News search with:
    - Hardcoded config
    - Built-in rate limit handling
    - Retry logic
    - Clean output (returns [] on any failure)
    """

    params = {
        "engine": "google_news",
        "api_key": news_bearer_token,
        "q": query,
        "hl": DEFAULT_HL,
        "gl": DEFAULT_GL,
        "tbs": "sbd:1",              # sort by date (1 = newest first)
        "output": "json"
    }

    for attempt in range(1, DEFAULT_RETRIES + 1):
        try:
            response = requests.get(
                SERPAPI_ENDPOINT,
                params=params,
                timeout=15
            )

            # HTTP-level error (non-200)
            response.raise_for_status()

            data = response.json()

            # ----- SerpAPI-level errors -----
            if "error" in data:
                message = data["error"].lower()

                # Monthly rate-limit exceeded
                if "exceeded monthly searches" in message:
                    print("SerpAPI monthly rate limit exceeded.")
                    return []

                # Invalid key, invalid query, bad parameters
                print(f"SerpAPI Error: {data['error']}")
                return []

            # Successful request
            results = data.get("news_results", []) or []
            return results[:DEFAULT_MAX_RESULTS]

        except requests.exceptions.HTTPError as e:
            print(f"[HTTP Error] {e}")
            return []

        except requests.exceptions.Timeout:
            print(f"[Timeout] attempt {attempt}/{DEFAULT_RETRIES}")
            time.sleep(DEFAULT_BACKOFF)

        except requests.exceptions.RequestException as e:
            print(f"[Network Error] {e} — retrying...")
            time.sleep(DEFAULT_BACKOFF)

        except Exception as e:
            print(f"[Unexpected Error] {e}")
            return []

    print("Too many failures. Returning empty list.")
    return []

def normalize_search_api_response(raw_articles):
    """
    Takes a list of raw SerpAPI google_news results (news_results)
    and normalizes them into a consistent structure.
    """
    normalized = []
    for r in raw_articles:
        normalized.append({
            "title": r.get("title", "") or "",
            "link": r.get("link", "") or "",
            "date": r.get("date", "") or "",
            "source": r.get("source", "") or "",
            "raw": r,   # keep if you need more fields later
        })
    return normalized


def dedupe_articles_by_link(articles):
    """
    Deduplicate articles based on their 'link'.
    """
    seen = set()
    deduped = []
    for a in articles:
        link = a.get("link")
        if not link:
            continue
        if link in seen:
            continue
        seen.add(link)
        deduped.append(a)
    return deduped


def select_top_k_by_tfidf(tweet_text, articles, k=3):
    """
    Rank articles by TF-IDF cosine similarity between:
      - tweet_text (claim)
      - article titles
    and return top-k.
    """
    if not articles:
        return []

    docs = [tweet_text] + [a["title"] for a in articles]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(docs)

    tweet_vec = tfidf[0:1]
    article_vecs = tfidf[1:]

    sims = cosine_similarity(tweet_vec, article_vecs)[0]

    for a, s in zip(articles, sims):
        a["similarity"] = float(s)

    sorted_articles = sorted(articles, key=lambda x: x["similarity"], reverse=True)
    result = []
    for a in sorted_articles[:k]:
        result.append({
            "title": a["title"],
            "link": a["link"],
            "similarity": a["similarity"]
        })
    return result[:k]


gnews_response = []

for q in search_queries:
    raw_results = serpapi_google_news_search(q)
    normalized = normalize_search_api_response(raw_results)
    gnews_response.extend(normalized)

deduped_list = dedupe_articles_by_link(gnews_response)

top_3_similar_headlines = select_top_k_by_tfidf(txt,deduped_list)
print(top_3_similar_headlines)


