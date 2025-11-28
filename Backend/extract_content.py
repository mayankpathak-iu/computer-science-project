# Importing all libraries
import time
import numpy as np
from dotenv import load_dotenv
import os
import sys
import re
import tweepy
import requests
import spacy
from rake_nltk import Rake
from typing import List, Dict, Any
from newspaper import Article
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

load_dotenv()  
bearer_token = os.getenv("bearer_token")
client = tweepy.Client(bearer_token, wait_on_rate_limit=True)
news_bearer_token = os.getenv("news_bearer_token")
SERPAPI_ENDPOINT = "https://serpapi.com/search"

# Ensure project root is on sys.path (works locally and in deployment)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from models.model import classify_stance_nli

start = time.time()

############################### Step 1 - Accepting tweet from user and fetching Tweet ######################################

def extract_tweet_id(url):

    """
    This function extracts the tweet id from the URL
    
    :param url: https://twitter.com/user/status/12345 (12345 is the Id)

    :Return: if match exists or None 
    """
    match = re.search(r"status/(\d+)", url)
    return match.group(1) if match else None

# Asking tweet input from the user
def get_tweets(client, tweetid, max_retries=3, sleep_seconds=60):

    """
    Calls X API safely with rate-limit handling.
    Retries automatically when hitting-TooManyRequests.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return client.get_tweets(
                tweetid,
                tweet_fields=["created_at", "text"]
            )

        except tweepy.TooManyRequests as e:
            print(f"[rate-limit] Twitter API rate limit hit (attempt {attempt}/{max_retries}).")

            retry_after = getattr(e, "retry_after", None)
            wait = retry_after or sleep_seconds

            print(f"Sleeping {wait} sec...")
            time.sleep(wait)

        except tweepy.TweepyException as e:
            print(f"[tweepy-error] {e}")
            return None

        except Exception as e:
            print(f"[unexpected-error] {e}")
            return None

    print("[rate-limit] Failed after maximum retries.")
    return None


print("Now running the rest of the pipeline...")

############################### STEP 2 - Creating search queries with different keywords to hit Google News API #############################

def clean_query_text(text):

    """
    A reuable function to clean a text
    
    :param text
    :Returns processed text
    """

    text = text or ""
    text = re.sub(r"[^A-Za-z0-9\.\?\!\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def dedupe_words_preserve_order(text):

    """
    This function ensure the text has unique keywords.
    
    :param text: Description
    :Returns: Text without duplicates
    """
    seen = set()
    result_words = []
    for w in text.split():
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            result_words.append(w)
    return " ".join(result_words)

def generate_search_queries(tweet_text, max_queries=3):
    
    """
    Generate optimized search-engine queries from a tweet/claim for retrieving
    relevant news articles.

    This function extracts key semantic elements from the input text using
    three complementary NLP strategies:

    1. Named Entity Recognition (NER) via spaCy
       - Extracts important real-world entities such as persons, organizations,
         locations, and events.

    2. Keyword extraction via RAKE (Rapid Automatic Keyword Extraction)
       - Identifies statistically relevant multi-word keyword phrases.

    3. Query construction and cleanup
       - Combines entities and keywords to form concise, high-signal queries.
       - Removes duplicate words and special characters.
       - Preserves keyword order to maintain semantic meaning.

    The output queries are designed to:
    - Maximize recall while searching news APIs.
    - Reduce noise from irrelevant results.
    - Capture both entity-centric and concept-centric queries.

    params: The raw tweet or claim text.

    Returns: List: A list of 1–3 cleaned and optimized search queries
                   suitable for use with search APIs (Google News).

    Example:
        Input:
            "Germany announced a ban on AI-generated videos in political campaigns"

        Output:["Germany AI generated videos ban","AI generated political campaigns"]

    Notes: Results may vary depending on tweet length and entity presence.
    """
   
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(tweet_text)

    queries = []

    # Named Entities
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    gpes = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]

    # Pick main person if exists
    main_person = persons[0] if persons else None

    # Noun Phrases
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]

    # Filter event-like phrases
    event_candidates = [
        p for p in noun_phrases
        if any(x in p.lower() for x in ["death", "sentence", "crime", "protest", "crackdown", "attack", "ban", "violence"])
    ]

    # RAKE Keywords (cleaned)
    rake = Rake()
    rake.extract_keywords_from_text(tweet_text)
    raw_keywords = rake.get_ranked_phrases()[:6]

    def keep_nouns_only(phrase):
        d = nlp(phrase)
        return " ".join([t.text for t in d if t.pos_ in ("NOUN", "PROPN") and t.is_alpha])

    clean_keywords = []
    for p in raw_keywords:
        k = keep_nouns_only(p)
        if k:
            clean_keywords.append(k)

    # --------- Build queries ---------

    # Query 1: person + strongest event phrase
    if main_person and event_candidates:
        q1 = f"{main_person} {event_candidates[0]}"
        queries.append(clean_query_text(q1))

    # Query 2: person + clean keyword
    if main_person and clean_keywords:
        q2 = f"{main_person} {clean_keywords[0]}"
        queries.append(clean_query_text(q2))

    # Query 3: person + year + event keyword
    year = next((d for d in dates if d.isdigit()), None)
    if main_person and year and event_candidates:
        q3 = f"{main_person} {year} {event_candidates[0]}"
        queries.append(clean_query_text(q3))

    # Fallback Queries if we still need more
    if len(queries) < max_queries and gpes and main_person:
        queries.append(clean_query_text(f"{main_person} {gpes[0]}"))

    if len(queries) < max_queries and clean_keywords:
        queries.append(clean_query_text(clean_keywords[0]))

    # Deduplicate and trim
    final = []
    for q in queries:
        if q not in final:
            final.append(q)

    return final[:max_queries]

############################### STEP 3 - Using search queries on Google news API  #################################################################

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

            # SerpAPI-level errors
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
    Takes a list of raw SerpAPI Google news response (news_results)
    and normalizes them into a consistent structure.
    :params: SerpAPI response
    :Return: Formatted list
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
    This function checks for common articles which needs to be removed. 
    (Different search queries can result in same articles)
    Deduplicate articles based on their 'link'.
    :params: Article list
    :Return: Unique list of Articles
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
    :params: original claim, unique list of articles, number of top similar articles
    :Return: Top K articles with similarity scores
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

############################### STEP 4 - Article body collection (Evidence)  #################################################################

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

def fetch_article_body(url):

    """
    Download and extract main article text from a URL.
    :params:URL of top K articles
    :Return: Body of article
    """

    if not url:
        return ""

    try:
        article = Article(url)
        article.download()
        article.parse()
        text = (article.text or "").strip()
        if not text:
            print(f"[warn] Empty article text for URL: {url}")
            return ""
        return text
    except Exception as e:
        print(f"[error] Failed to fetch article body for {url}: {e}")
        return ""
    
def split_sentences(text):

    """
    Function to tokenize each sentence. We split the sentence based on '.'
    :param text: Description
    :Return: tokenized text
    """
    text = (text or "").strip()
    if not text:
        return []
    return sent_tokenize(text)

def sbert_encode(texts):

    """
    This function encodes a list of texts with SBERT.
    :params: tokenized sentences
    :Return: A 2D numpy array: (n_texts, embedding_dim)
    """
    return sbert_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def extract_top_sentences_sbert(tweet_text, articles, top_k_sentences=3):

    """
    For each article, compute SBERT similarity between:
      - tweet_text
      - each sentence in article['body']
    Keep top_m_per_article sentences (and their similarity).
    Adds 'evidence_sentences' list to each article:
    :params: claim, articles, and number of top similar sentences
    Return: the modified article list with similar sentences
    """
    results = []

    # Encode claim once
    claim_emb = sbert_encode([tweet_text])[0]

    for a in articles:
        title = a.get("title", "").strip()
        link = a.get("link", "").strip()
        body = (a.get("body") or "").strip()

        # If no body extracted — skip or return empty evidence
        if not body:
            results.append({
                "title": title,
                "link": link,
                "evidence": []
            })
            continue

        # 1. Split the body into sentences
        sentences = split_sentences(body)
        if not sentences:
            results.append({
                "title": title,
                "link": link,
                "evidence": []
            })
            continue

        # 2. SBERT encode all sentences
        sent_embs = sbert_encode(sentences)

        # 3. Similarity (since embeddings normalized)
        sims = sent_embs @ claim_emb

        # 4. Pair sentences + scores
        pairs = [
            {"sentence": s, "similarity": float(sim)}
            for s, sim in zip(sentences, sims)
        ]

        # 5. Sort by similarity descending
        pairs_sorted = sorted(pairs, key=lambda x: x["similarity"], reverse=True)

        # 6. Keep only top-K
        evidence = pairs_sorted[:top_k_sentences]

        results.append({
            "title": title,
            "link": link,
            "evidence": evidence
        })

    return results

###################### STEP 5 classifying claims using NLI model ##############################################

def attach_stance_to_evidence(claim_text, articles):

    """
    Runs an NLI model on each evidence sentence to determine whether it
    supports, refutes, or is neutral toward the claim.

    For every evidence sentence, this function:
    - Compares it with the claim using a pretrained NLI model.
    - Assigns a stance label (supports / refutes / neutral).
    - Stores confidence scores for each class.

    params:
        articles: List of articles with extracted evidence sentences.
        claim: The tweet or claim text.
        tokenizer: Tokenizer for the NLI model.
        model: Pretrained NLI model.

    Returns:
        Articles with stance labels and probability scores added to
        each evidence sentence.
    """

    enriched = []

    for art in articles:
        new_art = {
            "title": art.get("title"),
            "link": art.get("link"),
            "evidence": []
        }

        for ev in art.get("evidence", []):
            sentence = ev.get("sentence", "")
            if not sentence:
                continue

            # Call NLI model here
            stance, probs = classify_stance_nli(claim_text, sentence)

            new_art["evidence"].append({
                "sentence": sentence,
                "similarity": ev.get("similarity"),
                "stance": stance,
                "nli_probs": probs,
            })

        enriched.append(new_art)
    return enriched

def summarize_article_stance(
    articles_with_sentence_stance,
    support_thresh: float = 0.6,
    refute_thresh: float = 0.6,
    margin: float = 0.1,):

    """
    Given a list of articles where each article has sentence-level stance info,
    compute an article-level stance.

    Returns:
        List of articles, each extended with:
          - article_stance: 'supports' | 'refutes' | 'neutral' | 'mixed' | 'no_evidence'
          - best_support_score: max entailment prob across evidence (or 0.0)
          - best_refute_score: max contradiction prob across evidence (or 0.0)
    """
    summarized = []

    for art in articles_with_sentence_stance:
        evidence = art.get("evidence") or []

        best_support = 0.0
        best_refute = 0.0
        has_any_probs = False

        # We keep evidence as-is, just read from it
        for ev in evidence:
            probs = ev.get("nli_probs")
            if not probs:
                continue

            has_any_probs = True
            ent = float(probs.get("entailment", 0.0))
            contra = float(probs.get("contradiction", 0.0))

            if ent > best_support:
                best_support = ent
            if contra > best_refute:
                best_refute = contra

        # Decide article-level stance
        if not evidence or not has_any_probs:
            article_stance = "no_evidence"
        else:
            if (
                best_support >= support_thresh
                and best_support >= best_refute + margin
            ):
                article_stance = "supports"
            elif (
                best_refute >= refute_thresh
                and best_refute >= best_support + margin
            ):
                article_stance = "refutes"
            else:
                # Some signal but conflicting / weak → mixed/neutral
                if best_support >= support_thresh or best_refute >= refute_thresh:
                    article_stance = "mixed"
                else:
                    article_stance = "neutral"

        # Build output article: preserve original fields + add stance summary
        new_art = {k: v for k, v in art.items()}  # shallow copy
        new_art["article_stance"] = article_stance
        new_art["best_support_score"] = best_support
        new_art["best_refute_score"] = best_refute
        new_art.pop("body", None)
        summarized.append(new_art)

    return summarized

def aggregate_claim_verdict(
    articles_with_article_stance,
    claim_text: str,
    support_thresh: float = 0.7,
    refute_thresh: float = 0.7,
    margin: float = 0.15,):

    """
    Aggregate article-level stances into a single claim-level verdict.

    Args:
        articles_with_article_stance: list of article dicts that already contain:
            - article_stance
            - best_support_score
            - best_refute_score
        claim_text: the original tweet / claim string
        support_thresh: min support score to call claim likely true
        refute_thresh: min refute score to call claim likely false
        margin: required gap between best support and best refute

    Returns:
        dict with:
            - claim: original claim_text
            - verdict: 'likely_true' | 'likely_false' | 'uncertain'
            - global_best_support
            - global_best_refute
            - counts of article stances
            - articles: original list with stances (unchanged)
    """
    if not articles_with_article_stance:
        return {
            "claim": claim_text,
            "verdict": "uncertain",
            "reason": "no_articles_found",
            "global_best_support": 0.0,
            "global_best_refute": 0.0,
            "num_supporting_articles": 0,
            "num_refuting_articles": 0,
            "num_neutral_articles": 0,
            "num_mixed_articles": 0,
            "num_no_evidence_articles": 0,
            "articles": [],
        }

    global_best_support = 0.0
    global_best_refute = 0.0

    num_supporting = 0
    num_refuting = 0
    num_neutral = 0
    num_mixed = 0
    num_no_evidence = 0

    for art in articles_with_article_stance:
        stance = art.get("article_stance", "neutral")
        bs = float(art.get("best_support_score", 0.0))
        br = float(art.get("best_refute_score", 0.0))

        if bs > global_best_support:
            global_best_support = bs
        if br > global_best_refute:
            global_best_refute = br

        if stance == "supports":
            num_supporting += 1
        elif stance == "refutes":
            num_refuting += 1
        elif stance == "neutral":
            num_neutral += 1
        elif stance == "mixed":
            num_mixed += 1
        elif stance == "no_evidence":
            num_no_evidence += 1
        else:
            # Unknown label – treat as neutral
            num_neutral += 1

    # Decide claim-level verdict
    if (
        global_best_refute >= refute_thresh
        and global_best_refute >= global_best_support + margin
    ):
        verdict = "likely_false"
        reason = "strong_refuting_evidence"
    elif (
        global_best_support >= support_thresh
        and global_best_support >= global_best_refute + margin
    ):
        verdict = "likely_true"
        reason = "strong_supporting_evidence"
    else:
        verdict = "uncertain"
        reason = "no_clear_winner"

    return {
        "claim": claim_text,
        "verdict": verdict,
        "reason": reason,
        "global_best_support": global_best_support,
        "global_best_refute": global_best_refute,
        "num_supporting_articles": num_supporting,
        "num_refuting_articles": num_refuting,
        "num_neutral_articles": num_neutral,
        "num_mixed_articles": num_mixed,
        "num_no_evidence_articles": num_no_evidence,
    }

def run_pipeline_from_tweet_text(url, top_k_articles = 3):

    # 1) validate & extract tweet id

    tweetid = extract_tweet_id(url)
    if not tweetid:
        raise ValueError("Invalid tweet link. Expected something like https://x.com/user/status/12345")

    # 2) call X API safely
    response = get_tweets(client, tweetid)

    if not response or not response.data:
        raise ValueError("Tweet not found or rate limited")

    tweet_obj = response.data[0]
    tweet_created_date = tweet_obj.created_at
    tweet_body = tweet_obj.text
    
    # 3) generate queries
    search_queries = generate_search_queries(tweet_body)

    # 4) search and normalize
    gnews_response = []
    for q in search_queries:
        raw_results = serpapi_google_news_search(q)
        normalized = normalize_search_api_response(raw_results)
        gnews_response.extend(normalized)

    # 5) dedupe
    deduped_list = dedupe_articles_by_link(gnews_response)

    # 6) top-k similar by TF-IDF
    top_similar_articles = select_top_k_by_tfidf(
        tweet_body,
        deduped_list,
        k=top_k_articles,
    )

    # 7) fetch bodies
    for a in top_similar_articles:
        url = a.get("link")
        body = clean_query_text(fetch_article_body(url))
        a["body"] = body

    # 8) SBERT sentence extraction
    articles_with_evidence = extract_top_sentences_sbert(
        tweet_body,
        top_similar_articles,
        top_k_sentences=3,
    )

    # 9) NLI stance per sentence
    articles_with_sentence_stance = attach_stance_to_evidence(
        tweet_body,
        articles_with_evidence,
    )

    # 10) article-level stance
    articles_with_article_stance = summarize_article_stance(
        articles_with_sentence_stance
    )
    # 10) claim-level verdict
    final_result = aggregate_claim_verdict(
        articles_with_article_stance,
        tweet_body,
    )
    return final_result

url = "https://x.com/DeusXMachina14/status/1990469787395432560"  # or any tweet

result = run_pipeline_from_tweet_text(url,3)

end = time.time()
print('--------------------------------------------------------------------------------')

print(f"Verdict: {result['verdict']} | "f"Support Score: {result['global_best_support']:.3f} | "f"Refute Score: {result['global_best_refute']:.3f}")

print('--------------------------------------------------------------------------------')

print(f"Pipeline execution time: {end - start:.2f} seconds")
