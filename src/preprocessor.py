"""
Data Preprocessing Module
Handles text cleaning, normalization, tokenization, and spell correction.
"""

import re
import string
import unicodedata
from typing import List, Tuple

NLP = None  # module-level; populated lazily by _load_spacy()

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from spellchecker import SpellChecker
    SPELLCHECK_AVAILABLE = True
except ImportError:
    SPELLCHECK_AVAILABLE = False


def _ensure_nltk_data():
    """Download required NLTK data if not present."""
    if not NLTK_AVAILABLE:
        return
    packages = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


def _load_spacy():
    global NLP
    if NLP is None and SPACY_AVAILABLE:
        try:
            NLP = spacy.load("en_core_web_sm")
        except Exception:
            try:
                from spacy.cli import download
                download("en_core_web_sm")
                NLP = spacy.load("en_core_web_sm")
            except Exception:
                NLP = None
    return NLP


class TextPreprocessor:
    """
    Handles the full text preprocessing pipeline for the AES system.
    Steps: normalization → tokenization → spell correction → stop word handling → lemmatization
    """

    def __init__(self):
        _ensure_nltk_data()
        self.nlp = _load_spacy()
        self.spell = SpellChecker() if SPELLCHECK_AVAILABLE else None
        if NLTK_AVAILABLE:
            try:
                self.stop_words = set(stopwords.words('english'))
                self.lemmatizer = WordNetLemmatizer()
            except Exception:
                self.stop_words = set()
                self.lemmatizer = None
        else:
            self.stop_words = set()
            self.lemmatizer = None

    def normalize(self, text: str) -> str:
        """Step 1: Clean and normalize raw essay text."""
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        # Fix common encoding artifacts
        text = text.replace('\u2019', "'").replace('\u2018', "'")
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2014', ' - ').replace('\u2013', '-')
        # Normalize whitespace
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        # Remove non-ASCII except common punctuation
        text = ''.join(c if (c.isascii() or c in '""''–—') else ' ' for c in text)
        return text.strip()

    def get_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences."""
        if NLTK_AVAILABLE:
            try:
                return [s.strip() for s in sent_tokenize(text) if s.strip()]
            except Exception:
                pass
        # Fallback: simple split on punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def get_words(self, text: str, remove_punctuation: bool = True) -> List[str]:
        """Tokenize text into words."""
        if remove_punctuation:
            text_clean = text.translate(str.maketrans('', '', string.punctuation))
        else:
            text_clean = text
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text_clean)
                return [t for t in tokens if t.strip()]
            except Exception:
                pass
        return text_clean.lower().split()

    def get_spelling_errors(self, text: str) -> Tuple[List[str], int]:
        """Detect spelling errors using PySpellChecker."""
        words = self.get_words(text.lower())
        alpha_words = [w for w in words if w.isalpha() and len(w) > 2]
        if self.spell and alpha_words:
            misspelled = list(self.spell.unknown(alpha_words))
            return misspelled, len(misspelled)
        return [], 0

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stop words — used for neural pathway focus on content words."""
        return [t for t in tokens if t.lower() not in self.stop_words]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens to their base forms."""
        if self.nlp:
            doc = self.nlp(' '.join(tokens))
            return [token.lemma_ for token in doc]
        if self.lemmatizer:
            return [self.lemmatizer.lemmatize(t) for t in tokens]
        return tokens

    def get_paragraphs(self, text: str) -> List[str]:
        """Split essay into paragraphs."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) <= 1:
            # Try single newline
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        return paragraphs

    def process(self, text: str) -> dict:
        """
        Full preprocessing pipeline.
        Returns a dict with normalized text, sentences, words, paragraphs, etc.
        """
        normalized = self.normalize(text)
        sentences = self.get_sentences(normalized)
        words = self.get_words(normalized)
        words_lower = [w.lower() for w in words if w.isalpha()]
        content_words = self.remove_stopwords(words_lower)
        lemmas = self.lemmatize(words_lower)
        paragraphs = self.get_paragraphs(normalized)
        misspelled, error_count = self.get_spelling_errors(normalized)

        return {
            'original': text,
            'normalized': normalized,
            'sentences': sentences,
            'words': words,
            'words_lower': words_lower,
            'content_words': content_words,
            'lemmas': lemmas,
            'paragraphs': paragraphs,
            'misspelled_words': misspelled,
            'spelling_error_count': error_count,
        }
