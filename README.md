# Fake News Detection & Claim Verification System

A real-time automated fact verification system that analyzes social media claims by retrieving evidence from trusted news sources and using Natural Language Inference to determine claim validity.

---

## Overview

This system takes a tweet URL, extracts the claim, searches for relevant news articles, and uses machine learning to verify whether the claim is supported or contradicted by current evidence. Unlike traditional approaches that rely solely on pattern matching, this system actively retrieves and analyzes real-world evidence to make informed decisions.

**Key Features:**
- Real-time evidence retrieval from Google News
- Semantic similarity matching using sentence embeddings
- Natural Language Inference for claim-evidence reasoning
- Explainable results with source citations
- Modular architecture for easy extension

---

## 🏗️ System Architecture

### **1. Claim Extraction**
The system accepts a tweet URL and extracts:
- Tweet text (the claim to verify)
- Posting timestamp
- Author metadata

### **2. Query Generation**
Smart query generation using multiple NLP techniques:
- **Named Entity Recognition** (spaCy) — extracts people, places, organizations
- **Keyword Extraction** (RAKE) — identifies key terms
- **Noun Phrase Detection** — captures important concepts

This multi-query approach increases the chance of finding relevant evidence.

### **3. News Article Retrieval**
Articles are fetched from Google News via SerpAPI:
- Automatic deduplication
- Date filtering (prioritizes recent articles)
- Source normalization
- Robust error handling

### **4. Article Processing**
Each article is parsed using `newspaper3k`:
- Extracts headline and body text
- Filters out ads and irrelevant content
- Captures publication timestamp

### **5. Evidence Sentence Selection**
The system uses **Sentence-BERT** (`all-MiniLM-L6-v2`) to:
- Convert claim and article sentences into semantic embeddings
- Calculate cosine similarity scores
- Select the most relevant sentences as evidence

Only sentences with high semantic similarity to the claim are kept for verification.

### **6. Natural Language Inference**
Each evidence sentence is analyzed using **RoBERTa-large-MNLI**:
- Determines if evidence **supports** the claim (entailment)
- Determines if evidence **contradicts** the claim (contradiction)
- Identifies if evidence is **neutral** or irrelevant

### **7. Article-Level Stance Classification**
Each article receives a stance label based on its strongest evidence:
- **SUPPORTS** — evidence confirms the claim
- **REFUTES** — evidence contradicts the claim  
- **NEUTRAL** — evidence is related but doesn't confirm or deny
- **MIXED** — contains both supporting and contradicting evidence
- **NO_EVIDENCE** — no relevant information found

### **8. Final Verdict**
All article stances are aggregated to produce the final verdict:
- ✅ **Likely True** — strong supporting evidence found
- ❌ **Likely False** — strong contradicting evidence found
- ⚠️ **Uncertain** — conflicting or insufficient evidence

---

## 🚀 Getting Started

### **Prerequisites**
- Python 3.10 or higher
- API keys for Twitter and SerpAPI
- At least 4GB RAM (8GB recommended for optimal performance)

### **Installation**

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

**2. Create a virtual environment**
```bash
# Using conda
conda create -n fakenews python=3.10
conda activate fakenews

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download required NLP models**
```python
# Run in Python
import nltk
import spacy

nltk.download('punkt')
nltk.download('stopwords')
spacy.cli.download('en_core_web_sm')
```

**5. Set up environment variables**

Create a `.env` file in the project root:
```env
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
SERPAPI_KEY=your_serpapi_key_here
```

**How to get API keys:**
- **Twitter:** Visit [Twitter Developer Portal](https://developer.twitter.com/)
- **SerpAPI:** Sign up at [SerpAPI](https://serpapi.com/) (100 free searches/month)

### **Running the System**

```bash
# Run the main verification pipeline
python Backend/main.py

```

---

## 📦 Requirements

Create a `requirements.txt` file with the following dependencies:

```txt
# Core ML and NLP
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.2
spacy>=3.6.0
nltk>=3.8.1

# Text processing
rake-nltk>=1.0.6
newspaper3k>=0.2.8
beautifulsoup4>=4.12.0
lxml>=4.9.2
lxml_html_clean>=0.1.0

# API and web
tweepy>=4.14.0
requests>=2.31.0
python-dotenv>=1.0.0

# Data processing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Utils
regex>=2023.6.3
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Project Structure

```
fake-news-detection/
│
├── Backend/
│   ├── main.py                 # main data processing pipeline
│   ├── app.py                  # Web app using FastAPI
│
├── models/
│   ├── model.py                 # Sentence embedding and ranking, RoBERTa NLI inference
│   ├── __init__.py              # init file to import modules
│
│
├── Frontend/                      
│    ├── index.html              # contains all the frontend/UI
│                                   
├── data/
│   └── FEVER_train.json         # Sample test data
│
├── .env                         # API keys (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Verification Logic

### **Article Stance Classification**

For each article, the system calculates:
```python
E = max entailment score across all sentences
C = max contradiction score across all sentences
```

Classification rules:
- **SUPPORTS** if `E ≥ 0.6` and `E ≥ C + 0.1`
- **REFUTES** if `C ≥ 0.6` and `C ≥ E + 0.1`
- **MIXED** if both scores exceed thresholds
- **NEUTRAL** if scores are weak but sentence is relevant
- **NO_EVIDENCE** if no relevant sentences found

### **Final Verdict Aggregation**

```python
best_contradiction = max contradiction score across all articles
best_entailment = max entailment score across all articles
```

Decision rules:
| Condition | Verdict |
|-----------|---------|
| `best_contradiction ≥ 0.7` | ❌ Likely False |
| `best_entailment ≥ 0.7` | ✅ Likely True |
| Weak or conflicting signals | ⚠️ Uncertain |

---

## Technical Details

### **Why Sentence-BERT?**
- Captures semantic meaning beyond keyword matching
- Fast inference (much faster than full transformers)
- Pre-trained on diverse text pairs
- Excellent at measuring text similarity

### **Why RoBERTa-MNLI?**
- State-of-the-art performance on natural language inference
- Trained on 433k human-annotated sentence pairs
- Handles complex reasoning (negation, paraphrasing, contradiction)
- Better than keyword-based or simple similarity approaches

### **Model Sizes**
- `all-MiniLM-L6-v2`: ~80 MB
- `roberta-large-mnli`: ~1.4 GB
- Total model storage: ~1.5 GB

---

## Use Cases

1. **Individual Users** — Verify suspicious social media posts before sharing
2. **Journalists** — Quick fact-checking during news reporting
3. **Researchers** — Study misinformation spread patterns
4. **Educators** — Teach media literacy with real examples
5. **Platform Moderators** — Flag potentially misleading content

---

## System Assumptions

| Assumption | Rationale |
|------------|-----------|
| English-only input | NLP models are English-trained |
| News articles are generally reliable | Major outlets have editorial standards |
| Recent articles are more relevant | Prioritizes current context |
| Some claims are inherently unverifiable | Breaking news, opinions, predictions |
| Multiple sources reduce bias | Aggregating diverse perspectives |

---

## Configuration

You can adjust system behavior by modifying these parameters:

```python
# In your configuration file
CONFIG = {
    'max_articles_per_query': 5,           # Articles to retrieve per search query
    'similarity_threshold': 0.7,           # Min similarity for evidence selection
    'entailment_threshold': 0.6,           # Min score for "supports" classification
    'contradiction_threshold': 0.6,        # Min score for "refutes" classification
    'verdict_confidence_threshold': 0.7,   # Min score for definitive verdict
    'max_evidence_sentences': 3,           # Top sentences per article
}
```

---

## Strengths

- **Evidence-Based** — Uses real news sources, not just pattern recognition
- **Explainable** — Shows which sentences support or contradict the claim
- **Real-Time** — Retrieves current information, not limited to training data
- **Modular** — Easy to swap components (different models, search APIs, etc.)
- **No Training Required** — Uses pre-trained models, works out of the box
- **Scalable** — Can process multiple claims in parallel

---

## Limitations

1. **Language** — Currently English-only
2. **Paywalls** — Cannot access articles behind paywalls
3. **API Costs** — SerpAPI has rate limits and costs
4. **Breaking News** — Very recent events may lack coverage
5. **Computational Resources** — Large models require significant RAM
6. **Sarcasm & Satire** — May struggle with non-literal language
7. **Opinion Claims** — Cannot verify subjective statements

---

## Future Improvements

- [ ] **Source Credibility Scoring** — Weight evidence by publisher reputation
- [ ] **Multilingual Support** — Extend to Spanish, French, German, etc.
- [ ] **Claim Decomposition** — Break complex claims into verifiable sub-claims
- [ ] **Temporal Reasoning** — Better handling of time-sensitive claims
- [ ] **Visual Analysis** — Verify images and videos in tweets
- [ ] **User Feedback Loop** — Learn from user corrections
- [ ] **Stance Detection Training** — Fine-tune models on fact-checking datasets
- [ ] **Caching Layer** — Store results for frequently checked claims

---

## References

- [Sentence-BERT Paper](https://arxiv.org/abs/1908.10084)
- [RoBERTa Paper](https://arxiv.org/abs/1907.11692)
- [MNLI Dataset](https://cims.nyu.edu/~sbowman/multinli/)
- [FEVER Dataset](https://fever.ai/)

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**
