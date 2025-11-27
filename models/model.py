import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Global cache for the SBERT model
_SBERT_MODEL = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Which SBERT model to use (small + fast, good default)
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_NAME = "roberta-large-mnli"

_NLI_MODEL = None
_NLI_TOKENIZER = None

def get_sbert_model():
    """
    - First call: loads the model and moves it to the right device.
    - Later calls: just return the same model (no reload).
    """
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        _SBERT_MODEL = SentenceTransformer(SBERT_MODEL_NAME, device=str(_DEVICE))
    return _SBERT_MODEL

def sbert_encode(texts):
    """
    Encode a string or list of strings into SBERT embeddings.

    Args:
        texts: str OR list[str]

    Returns:
        torch.Tensor:
          - shape (D,) if input is a single string
          - shape (N, D) if input is a list of N strings
    """
    model = get_sbert_model()

    # Normalize to list
    is_str = isinstance(texts, str)
    if is_str:
        texts = [texts]

    emb = model.encode(texts, convert_to_tensor=True)

    if is_str:
        # for a single string, return a 1D vector instead of (1, D)
        return emb[0]
    return emb

def load_nli_model():
    """
    Loads the NLI model and tokenizer once and caches them.
    """
    global _NLI_MODEL, _NLI_TOKENIZER

    if _NLI_MODEL is None or _NLI_TOKENIZER is None:
        print("Loading NLI model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.to(_DEVICE)      # move to CPU/GPU
        model.eval()

        _NLI_MODEL = model
        _NLI_TOKENIZER = tokenizer

    return _NLI_TOKENIZER, _NLI_MODEL

def _nli_label_probs(logits):
    """
    Convert raw logits from MNLI model to:
    - stance label: 'supports' | 'refutes' | 'neutral'
    - probs dict with keys: 'entailment', 'contradiction', 'neutral'

    For roberta-large-mnli, label order is:
        0: contradiction
        1: neutral
        2: entailment
    """
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    contradiction_prob, neutral_prob, entailment_prob = probs

    # Pick stance based on which probability is highest
    if entailment_prob >= max(contradiction_prob, neutral_prob):
        stance = "supports"
    elif contradiction_prob >= max(entailment_prob, neutral_prob):
        stance = "refutes"
    else:
        stance = "neutral"

    return stance, {
        "entailment": float(entailment_prob),
        "contradiction": float(contradiction_prob),
        "neutral": float(neutral_prob),
    }

def classify_stance_nli(claim_text: str, evidence_sentence: str):
    """
    Run NLI on a single (claim, evidence) pair.

    Args:
        claim_text: the tweet / claim string
        evidence_sentence: one sentence from an article

    Returns:
        stance: 'supports' | 'refutes' | 'neutral'
        probs: dict with keys 'entailment', 'contradiction', 'neutral'
    """
    tokenizer, model = load_nli_model()

    # Tokenize as a pair: (claim, evidence)
    inputs = tokenizer(
        claim_text,
        evidence_sentence,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(_DEVICE)

    # Forward pass (no gradients needed)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape: (1, 3)

    # Convert logits → stance + probabilities
    return _nli_label_probs(logits)


