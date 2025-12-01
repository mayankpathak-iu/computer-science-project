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


## 🎥 Demo (System Walkthrough & Results)

The following videos demonstrate the complete working of the Fake News Detection & Claim Verification System, including real-time evidence retrieval, sentence ranking, and NLI-based reasoning.

---

### 📺 Demo 1 — Full End-to-End Walkthrough
<a href="https://youtu.be/ozSngJRwNhw" target="_blank">
  <img src="https://img.youtube.com/vi/ozSngJRwNhw/maxresdefault.jpg" alt="Demo 1 - System Walkthrough" width="720">
</a>

### 📺 Demo 2 — Reasoning Breakdown & Output Explanation
<a href="https://youtu.be/kHO6tnqeQws" target="_blank">
  <img src="https://img.youtube.com/vi/kHO6tnqeQws/maxresdefault.jpg" alt="Demo 2 - Reasoning & Verdict" width="720">
</a>

Covers:
✔ Article-level stance calculation  
✔ Support vs contradiction scores  
✔ Confidence thresholds  
✔ Edge cases and ambiguous scenarios  
✔ Interpretation of output  

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


⸻

📦 requirements.txt

torch
transformers
sentence-transformers
spacy
nltk
rake-nltk
newspaper3k
beautifulsoup4
lxml
lxml_html_clean
requests
python-dotenv
tweepy
numpy
pandas
scikit-learn
regex
tqdm


⸻

📂 Project Structure

Backend/
├── main.py
├── app.py
models/
├── model.py
Frontend/
└── index.html
requirements.txt
README.md


⸻

✅ Strengths
	•	Evidence-driven
	•	Explainable reasoning
	•	Real-time fact checking
	•	No training required
	•	Neural semantic reasoning
	•	Scalable API design

⸻

⚠️ Limitations
	•	English only
	•	Requires news coverage
	•	No access to paywalled articles
	•	Depends on external APIs
	•	NLI confidence ≠ factual certainty
	•	Opinions cannot be verified

⸻

🔭 Future Work
	•	Multilingual support
	•	Claim decomposition
	•	Credibility scoring
	•	Multimodal fact checking
	•	LLM-driven reasoning layer
	•	Evidence summarization
	•	Domain weighting

⸻

📚 References
	•	Sentence-BERT – https://arxiv.org/abs/1908.10084
	•	RoBERTa – https://arxiv.org/abs/1907.11692
	•	MNLI – https://cims.nyu.edu/~sbowman/multinli
	•	FEVER – https://fever.ai

⸻

⭐ If this system helped you, consider starring the repo.

---