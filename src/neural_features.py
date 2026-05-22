"""
Neural Features Module (Deep Perception Pathway)
Extracts semantic embeddings via DeBERTa-v3 with graceful TF-IDF fallback.
"""

import numpy as np
from typing import Optional, Tuple

TRANSFORMERS_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


_TOKENIZER = None
_MODEL = None
_FALLBACK_VECTORIZER = None
_FALLBACK_SVD = None
_DEBERTA_LOADED = False
_DEBERTA_TRIED = False


def _try_load_deberta(model_name: str = "microsoft/deberta-v3-small") -> bool:
    """Attempt to load DeBERTa-v3. Returns True if successful."""
    global _TOKENIZER, _MODEL, _DEBERTA_LOADED, _DEBERTA_TRIED
    if _DEBERTA_TRIED:
        return _DEBERTA_LOADED
    _DEBERTA_TRIED = True
    if not TRANSFORMERS_AVAILABLE:
        return False
    try:
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _MODEL = AutoModel.from_pretrained(model_name)
        _MODEL.eval()
        _DEBERTA_LOADED = True
        return True
    except Exception:
        _DEBERTA_LOADED = False
        return False


def _get_deberta_embedding(text: str, max_length: int = 512) -> Optional[np.ndarray]:
    """Extract CLS embedding from DeBERTa-v3."""
    if not _DEBERTA_LOADED or _TOKENIZER is None or _MODEL is None:
        return None
    try:
        # Truncate long essays for performance
        inputs = _TOKENIZER(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        with torch.no_grad():
            outputs = _MODEL(**inputs)
        # Use mean pooling over last hidden state
        last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden_size)
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        mean_pool = (last_hidden * attention_mask).sum(1) / attention_mask.sum(1)
        embedding = mean_pool.squeeze().cpu().numpy()
        return embedding
    except Exception:
        return None


def _get_tfidf_embedding(text: str, n_components: int = 128) -> np.ndarray:
    """Fallback: TF-IDF + LSA embedding when DeBERTa unavailable."""
    global _FALLBACK_VECTORIZER, _FALLBACK_SVD
    if _FALLBACK_VECTORIZER is None and SKLEARN_AVAILABLE:
        # Warm-up corpus — large enough to give SVD at least n_components features
        dummy = [
            "this is a test essay about writing and language skills",
            "automated essay scoring uses natural language processing techniques",
            "students write essays to demonstrate their knowledge and understanding",
            "grammar vocabulary sentence fluency organisation content readability",
            "education technology communication digital literacy critical thinking",
            "research study analysis evaluation assessment measurement performance",
        ]
        _FALLBACK_VECTORIZER = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                                sublinear_tf=True)
        tfidf_mat = _FALLBACK_VECTORIZER.fit_transform(dummy)
        n_features = tfidf_mat.shape[1]
        safe_components = min(n_components, n_features - 1)
        _FALLBACK_SVD = TruncatedSVD(n_components=safe_components, random_state=42)
        _FALLBACK_SVD.fit(tfidf_mat)
    try:
        tfidf_vec = _FALLBACK_VECTORIZER.transform([text])
        embedding = _FALLBACK_SVD.transform(tfidf_vec).flatten()
        # Pad or truncate to exactly n_components
        if len(embedding) < n_components:
            embedding = np.pad(embedding, (0, n_components - len(embedding)))
        return embedding[:n_components]
    except Exception:
        return np.zeros(n_components)


class NeuralFeatureExtractor:
    """
    Manages the neural (deep perception) pathway of the HFC-AES system.
    Attempts DeBERTa-v3 first; falls back to TF-IDF + LSA if unavailable.
    """

    def __init__(self, use_deberta: bool = True):
        self.use_deberta = use_deberta
        self.deberta_available = False
        if use_deberta:
            self.deberta_available = _try_load_deberta()

    @property
    def embedding_dim(self) -> int:
        if self.deberta_available:
            return 768  # DeBERTa-v3-small hidden size
        return 128  # TF-IDF + LSA fallback

    @property
    def model_name(self) -> str:
        if self.deberta_available:
            return "DeBERTa-v3-small"
        return "TF-IDF + LSA (Fallback)"

    def get_embedding(self, text: str) -> np.ndarray:
        """Get essay embedding from active model."""
        if self.deberta_available:
            emb = _get_deberta_embedding(text)
            if emb is not None:
                return emb
        return _get_tfidf_embedding(text)

    def compute_semantic_features(self, text: str) -> dict:
        """
        Compute high-level semantic features from the embedding and text.
        Returns both the embedding and derived scalar metrics.
        """
        embedding = self.get_embedding(text)

        # Derive semantic scalar features from embedding statistics
        features = {
            'embedding_norm': float(np.linalg.norm(embedding)),
            'embedding_mean': float(np.mean(embedding)),
            'embedding_std': float(np.std(embedding)),
            'embedding_max': float(np.max(embedding)),
            'embedding_min': float(np.min(embedding)),
        }

        # Sentence-level coherence: compare adjacent sentence embeddings
        coherence_score = self._compute_coherence(text)
        features['semantic_coherence'] = coherence_score

        return {
            'embedding': embedding,
            'scalar_features': features,
            'model_used': self.model_name,
        }

    def _compute_coherence(self, text: str, max_sentences: int = 20) -> float:
        """
        Compute coherence as average cosine similarity between adjacent sentences.
        Higher = more logically connected essay.
        """
        try:
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
            sentences = sentences[:max_sentences]
            if len(sentences) < 2:
                return 0.5  # neutral if not enough sentences

            embeddings = [self.get_embedding(s) for s in sentences]
            similarities = []
            for i in range(len(embeddings) - 1):
                a, b = embeddings[i], embeddings[i + 1]
                norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
                if norm_a > 0 and norm_b > 0:
                    sim = np.dot(a, b) / (norm_a * norm_b)
                    similarities.append(float(sim))
            if similarities:
                return round(float(np.mean(similarities)), 4)
            return 0.5
        except Exception:
            return 0.5

    def get_attention_highlights(self, text: str, top_n: int = 5) -> list:
        """
        Return top-N sentences that likely contribute most to score.
        Uses embedding norm as a proxy for semantic richness.
        """
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        if not sentences:
            return []
        try:
            scored = []
            for sent in sentences[:30]:  # cap for performance
                emb = self.get_embedding(sent)
                score = float(np.linalg.norm(emb))
                scored.append((score, sent))
            scored.sort(reverse=True)
            return [s for _, s in scored[:top_n]]
        except Exception:
            return sentences[:top_n]
