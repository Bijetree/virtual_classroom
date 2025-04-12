import difflib
import re
import numpy as np
import pdfplumber
import mammoth
import html
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from Levenshtein import ratio as levenshtein_ratio
from collections import Counter
from functools import lru_cache
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
# Load NLP model
#bert_model = SentenceTransformer("paraphrase-MiniLM-L12-v2")

# Initialize utilities
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Precompile regex for efficiency
special_chars_regex = re.compile(r"[^\w\s'-]")

# Preprocessing function
def preprocess_text(text):
    """Preprocess text: lowercasing, remove special chars, tokenization, lemmatization, and stemming."""
    try:
        text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode("utf-8")
        text = special_chars_regex.sub('', text)
        tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in word_tokenize(text) if word not in stop_words]
        return " ".join(tokens)
    except Exception:
        return ""

# Text extraction functions
def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    try:
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages]).strip()
    except Exception:
        return ""

def extract_text_from_docx(file):
    """Extract text from a DOCX file."""
    try:
        return mammoth.extract_raw_text(file).value.strip()
    except Exception:
        return ""

# Similarity computations
def jaccard_similarity(text1, text2):
    """Calculates Jaccard Similarity."""
    words1, words2 = set(text1.split()), set(text2.split())
    return len(words1 & words2) / len(words1 | words2) if words1 and words2 else 0.0

@lru_cache(maxsize=100)
def get_bert_embedding(text):
    """Gets cached BERT embeddings."""
    return bert_model.encode(text)

def bert_similarity(text1, text2):
    """Computes semantic similarity using BERT."""
    try:
        return cosine_similarity([get_bert_embedding(text1)], [get_bert_embedding(text2)])[0][0]
    except Exception:
        return 0.0

# Find matching phrases
def find_matching_phrases(text1, text2, n=3, threshold=0.6):
    """Finds matching n-grams between texts using Jaccard similarity."""
    def get_ngrams(text, n):
        words = text.split()
        return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
    
    ngrams1, ngrams2 = Counter(get_ngrams(text1, n)), Counter(get_ngrams(text2, n))
    return [(phrase, phrase) for phrase in set(ngrams1) & set(ngrams2) if difflib.SequenceMatcher(None, phrase, phrase).ratio() >= threshold]

# Highlight matching phrases
def highlight_text(original, submitted, matching_phrases):
    """Highlights matching phrases in both texts."""
    highlight_style = '<span style="background-color: yellow; color: black;">{}</span>'
    for phrase, _ in matching_phrases:
        escaped = html.escape(phrase)
        highlighted = highlight_style.format(escaped)
        original = re.sub(rf"\b{re.escape(phrase)}\b", highlighted, original, flags=re.IGNORECASE)
        submitted = re.sub(rf"\b{re.escape(phrase)}\b", highlighted, submitted, flags=re.IGNORECASE)
    return original, submitted

# Compare two texts
def compare_texts(text1, text2):
    #from sentence_transformers import SentenceTransformer
    #bert_model = SentenceTransformer("paraphrase-MiniLM-L12-v2")
    """Compares texts and returns similarity score with highlighted differences."""
    if not text1 or not text2:
        return 0.0, text1, text2

    text1_clean, text2_clean = preprocess_text(text1), preprocess_text(text2)
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text1_clean, text2_clean])
        tfidf_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    except Exception:
        tfidf_similarity = 0.0

    jaccard_sim, lev_similarity, bert_sim = jaccard_similarity(text1_clean, text2_clean), levenshtein_ratio(text1_clean, text2_clean), bert_similarity(text1_clean, text2_clean)
    final_score = (tfidf_similarity * 0.3) + (jaccard_sim * 0.2) + (lev_similarity * 0.2) + (bert_sim * 0.3)
    highlighted_text1, highlighted_text2 = highlight_text(text1, text2, find_matching_phrases(text1, text2))
    return round(final_score * 100, 2), highlighted_text1, highlighted_text2
