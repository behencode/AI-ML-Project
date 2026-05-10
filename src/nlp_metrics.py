import warnings
import ssl

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import rouge_score
    from rouge_score import rouge_scorer
    _NLTK_AVAILABLE = True
except ImportError:
    warnings.warn("nltk or rouge_score not installed. NLP metrics will be 0.0.")
    _NLTK_AVAILABLE = False

_DOWNLOAD_ATTEMPTED = False

def _ensure_nltk_data() -> None:
    global _DOWNLOAD_ATTEMPTED
    if not _NLTK_AVAILABLE or _DOWNLOAD_ATTEMPTED:
        return
    _DOWNLOAD_ATTEMPTED = True
    
    try:
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading NLTK wordnet...")
            nltk.download('wordnet', quiet=True)
            
        try:
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            print("Downloading NLTK omw-1.4...")
            nltk.download('omw-1.4', quiet=True)
    except Exception as exc:
        print(f"Warning: Failed to download NLTK data: {exc}")


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute BLEU score for a single sentence pair.
    
    Args:
        reference: The gold truth string.
        hypothesis: The generated string.
        
    Returns:
        BLEU score (0.0 to 1.0).
    """
    if not _NLTK_AVAILABLE or not reference or not hypothesis:
        return 0.0
    _ensure_nltk_data()
    
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    # Use smoothing function to avoid 0.0 scores for short sentences
    smoothie = SmoothingFunction().method1
    
    # For short sentences, limit the max n-gram order if necessary, but
    # default sentence_bleu handles it okay with smoothing.
    score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
    return float(score)


def compute_rouge(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 score for a single sentence pair.
    
    Args:
        reference: The gold truth string.
        hypothesis: The generated string.
        
    Returns:
        ROUGE-L F1 score (0.0 to 1.0).
    """
    if not _NLTK_AVAILABLE or not reference or not hypothesis:
        return 0.0
    _ensure_nltk_data()
        
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return float(scores['rougeL'].fmeasure)


def compute_meteor(reference: str, hypothesis: str) -> float:
    """Compute METEOR score for a single sentence pair.
    
    Args:
        reference: The gold truth string.
        hypothesis: The generated string.
        
    Returns:
        METEOR score (0.0 to 1.0).
    """
    if not _NLTK_AVAILABLE or not reference or not hypothesis:
        return 0.0
    _ensure_nltk_data()
        
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    try:
        score = meteor_score([ref_tokens], hyp_tokens)
        return float(score)
    except Exception:
        return 0.0
