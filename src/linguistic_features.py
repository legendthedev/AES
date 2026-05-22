"""
Handcrafted Linguistic Features Module (Shallow Perception Pathway)
Extracts explicit, measurable linguistic markers from essays for the HFC-AES system.
"""

import re
import math
import string
from collections import Counter
from typing import Dict, List, Any

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


# Academic/sophisticated word list (high-frequency academic vocabulary)
ACADEMIC_WORDS = {
    'analyze', 'analyse', 'argument', 'assessment', 'assume', 'authority', 'available',
    'benefit', 'concept', 'consistent', 'constitutional', 'context', 'contract', 'create',
    'data', 'definition', 'derived', 'distribution', 'economic', 'environment', 'established',
    'estimate', 'evident', 'export', 'factors', 'financial', 'formula', 'function', 'identified',
    'income', 'indicate', 'individual', 'interpretation', 'involved', 'issues', 'labour', 'legal',
    'legislation', 'major', 'method', 'occurs', 'percent', 'period', 'policy', 'principle',
    'procedure', 'process', 'required', 'research', 'response', 'role', 'section', 'significant',
    'similar', 'source', 'specific', 'structure', 'theory', 'variables', 'evidence', 'however',
    'therefore', 'furthermore', 'moreover', 'consequently', 'subsequently', 'demonstrate',
    'emphasize', 'illustrate', 'suggest', 'conclude', 'examine', 'investigate', 'evaluate',
    'implement', 'justify', 'maintain', 'obtain', 'participate', 'perceive', 'perspective',
    'predominant', 'significant', 'sufficient', 'underlying', 'validate', 'phenomenon',
    'hypothesis', 'methodology', 'paradigm', 'conceptual', 'empirical', 'substantial',
    'correlation', 'framework', 'distinction', 'implications', 'comprehensive', 'systematic',
}

# Discourse connectors / transition words
DISCOURSE_MARKERS = {
    'however', 'furthermore', 'moreover', 'therefore', 'consequently', 'additionally',
    'nevertheless', 'nonetheless', 'meanwhile', 'subsequently', 'thus', 'hence',
    'although', 'despite', 'whereas', 'while', 'on the other hand', 'in contrast',
    'in conclusion', 'to summarize', 'in addition', 'as a result', 'for example',
    'for instance', 'in fact', 'indeed', 'similarly', 'likewise', 'conversely',
    'alternatively', 'finally', 'firstly', 'secondly', 'thirdly', 'in summary',
    'to conclude', 'to begin with', 'above all', 'especially', 'particularly',
}


def _load_spacy_model():
    """Lazy-load spaCy model."""
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


class LinguisticFeatureExtractor:
    """
    Extracts handcrafted linguistic features for the shallow perception pathway.
    Features span: lexical, syntactic, grammatical/mechanical, and structural dimensions.
    """

    def __init__(self):
        self.nlp = _load_spacy_model() if SPACY_AVAILABLE else None

    # ─── LEXICAL FEATURES ────────────────────────────────────────────────────

    def word_count(self, words: List[str]) -> int:
        return len(words)

    def unique_word_count(self, words: List[str]) -> int:
        return len(set(w.lower() for w in words))

    def lexical_density(self, words: List[str]) -> float:
        """Type-Token Ratio (TTR): unique/total words."""
        if not words:
            return 0.0
        return round(self.unique_word_count(words) / len(words), 4)

    def corrected_ttr(self, words: List[str]) -> float:
        """CTTR: accounts for essay length bias."""
        n = len(words)
        v = self.unique_word_count(words)
        if n == 0:
            return 0.0
        return round(v / math.sqrt(2 * n), 4)

    def academic_word_ratio(self, words: List[str]) -> float:
        """Proportion of academic vocabulary words."""
        if not words:
            return 0.0
        academic_count = sum(1 for w in words if w.lower() in ACADEMIC_WORDS)
        return round(academic_count / len(words), 4)

    def avg_word_length(self, words: List[str]) -> float:
        """Average character length per word."""
        if not words:
            return 0.0
        return round(sum(len(w) for w in words) / len(words), 2)

    def hapax_legomena_ratio(self, words: List[str]) -> float:
        """Ratio of words that appear only once (lexical richness)."""
        if not words:
            return 0.0
        freq = Counter(w.lower() for w in words)
        hapax = sum(1 for v in freq.values() if v == 1)
        return round(hapax / len(freq), 4) if freq else 0.0

    def discourse_marker_count(self, text: str) -> int:
        """Count of discourse/transition markers used."""
        text_lower = text.lower()
        count = 0
        for marker in DISCOURSE_MARKERS:
            count += len(re.findall(r'\b' + re.escape(marker) + r'\b', text_lower))
        return count

    # ─── SYNTACTIC FEATURES ──────────────────────────────────────────────────

    def sentence_count(self, sentences: List[str]) -> int:
        return len(sentences)

    def avg_sentence_length(self, sentences: List[str], words: List[str]) -> float:
        """Average words per sentence."""
        if not sentences:
            return 0.0
        return round(len(words) / len(sentences), 2)

    def sentence_length_variance(self, sentences: List[str]) -> float:
        """Variance in sentence length (higher = more variety)."""
        if len(sentences) < 2:
            return 0.0
        lengths = [len(s.split()) for s in sentences]
        mean = sum(lengths) / len(lengths)
        variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
        return round(math.sqrt(variance), 2)  # std dev

    def complex_sentence_ratio(self, text: str, sentences: List[str]) -> float:
        """Proportion of sentences with subordinate clauses."""
        subordinators = r'\b(although|because|since|when|while|if|unless|until|after|before|though|whereas|whether|that|which|who|whom|whose)\b'
        if not sentences:
            return 0.0
        complex_count = sum(1 for s in sentences if re.search(subordinators, s.lower()))
        return round(complex_count / len(sentences), 4)

    def pos_tag_features(self, text: str) -> Dict[str, float]:
        """Extract POS-based features using spaCy."""
        features = {
            'noun_ratio': 0.0,
            'verb_ratio': 0.0,
            'adj_ratio': 0.0,
            'adv_ratio': 0.0,
            'passive_ratio': 0.0,
        }
        if not self.nlp or not text:
            return features
        try:
            # Limit for performance
            doc = self.nlp(text[:5000])
            tokens = [t for t in doc if not t.is_space]
            if not tokens:
                return features
            n = len(tokens)
            nouns = sum(1 for t in tokens if t.pos_ in ('NOUN', 'PROPN'))
            verbs = sum(1 for t in tokens if t.pos_ == 'VERB')
            adjs = sum(1 for t in tokens if t.pos_ == 'ADJ')
            advs = sum(1 for t in tokens if t.pos_ == 'ADV')
            # Passive: aux + past participle pattern
            passives = sum(1 for t in tokens if t.dep_ == 'auxpass')
            features.update({
                'noun_ratio': round(nouns / n, 4),
                'verb_ratio': round(verbs / n, 4),
                'adj_ratio': round(adjs / n, 4),
                'adv_ratio': round(advs / n, 4),
                'passive_ratio': round(passives / max(verbs, 1), 4),
            })
        except Exception:
            pass
        return features

    # ─── GRAMMATICAL / MECHANICAL FEATURES ──────────────────────────────────

    def spelling_error_density(self, error_count: int, word_count: int) -> float:
        """Spelling errors per 100 words."""
        if word_count == 0:
            return 0.0
        return round((error_count / word_count) * 100, 2)

    def punctuation_density(self, text: str, word_count: int) -> float:
        """Punctuation marks per 100 words."""
        if word_count == 0:
            return 0.0
        punct_count = sum(1 for c in text if c in string.punctuation)
        return round((punct_count / word_count) * 100, 2)

    def capitalization_errors(self, sentences: List[str]) -> int:
        """Count sentences not starting with a capital letter."""
        errors = 0
        for s in sentences:
            s = s.strip()
            if s and s[0].isalpha() and not s[0].isupper():
                errors += 1
        return errors

    def repeated_word_ratio(self, words: List[str], window: int = 5) -> float:
        """Ratio of adjacent repeated content words (sign of repetitiveness)."""
        if len(words) < 2:
            return 0.0
        stopwords_simple = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'and', 'or', 'is', 'are', 'was', 'were'}
        content = [w.lower() for w in words if w.lower() not in stopwords_simple]
        repeats = 0
        for i in range(window, len(content)):
            if content[i] in content[max(0, i-window):i]:
                repeats += 1
        return round(repeats / max(len(content) - window, 1), 4)

    # ─── READABILITY FEATURES ────────────────────────────────────────────────

    def readability_scores(self, text: str) -> Dict[str, float]:
        """Compute standard readability indices."""
        scores = {
            'flesch_reading_ease': 0.0,
            'flesch_kincaid_grade': 0.0,
            'gunning_fog': 0.0,
            'smog_index': 0.0,
            'automated_readability_index': 0.0,
        }
        if not TEXTSTAT_AVAILABLE or not text:
            return scores
        try:
            scores['flesch_reading_ease'] = round(textstat.flesch_reading_ease(text), 2)
            scores['flesch_kincaid_grade'] = round(textstat.flesch_kincaid_grade(text), 2)
            scores['gunning_fog'] = round(textstat.gunning_fog(text), 2)
            scores['smog_index'] = round(textstat.smog_index(text), 2)
            scores['automated_readability_index'] = round(textstat.automated_readability_index(text), 2)
        except Exception:
            pass
        return scores

    # ─── STRUCTURAL FEATURES ─────────────────────────────────────────────────

    def paragraph_count(self, paragraphs: List[str]) -> int:
        return len(paragraphs)

    def avg_sentences_per_paragraph(self, paragraphs: List[str], sentences: List[str]) -> float:
        if not paragraphs:
            return 0.0
        return round(len(sentences) / len(paragraphs), 2)

    def has_intro_conclusion(self, paragraphs: List[str]) -> Dict[str, bool]:
        """Heuristic check for intro and conclusion paragraphs."""
        intro_signals = [
            # Classic essay signals
            'this essay', 'this paper', 'in this', 'this study', 'i will', 'we will',
            'this report', 'the purpose', 'the aim', 'this work discusses',
            # Academic / research article variants
            'this article', 'this research', 'this work', 'this chapter',
            'this document', 'this analysis', 'this review', 'this thesis',
            'the following', 'will discuss', 'will explore', 'will examine',
            'will investigate', 'will argue', 'will present', 'will demonstrate',
            'it is important', 'it has been', 'there is a', 'there are',
            'the concept', 'the topic', 'this topic', 'the question',
            'background', 'introduction', 'overview', 'context of',
            'has become', 'have become', 'plays a', 'play a', 'is one of',
        ]
        conclusion_signals = [
            # Classic essay signals
            'in conclusion', 'to conclude', 'in summary', 'to summarize',
            'in closing', 'finally', 'overall', 'in brief', 'this essay has',
            # Academic / research article variants
            'to sum up', 'in short', 'in essence', 'as discussed',
            'as shown', 'as demonstrated', 'as explored', 'as examined',
            'it can be concluded', 'it is clear that', 'it is evident',
            'the study has', 'this paper has', 'this research has',
            'findings suggest', 'findings indicate', 'results show',
            'results indicate', 'results demonstrate', 'the results',
            'therefore', 'thus it', 'hence', 'recommend', 'future research',
            'looking ahead', 'moving forward', 'going forward',
            'based on the', 'given the', 'the above', 'taken together',
            'this highlights', 'this shows', 'this suggests', 'this indicates',
        ]
        has_intro = has_conclusion = False
        if paragraphs:
            first_para = paragraphs[0].lower()
            has_intro = any(sig in first_para for sig in intro_signals)
            last_para = paragraphs[-1].lower()
            has_conclusion = any(sig in last_para for sig in conclusion_signals)
        return {'has_intro': has_intro, 'has_conclusion': has_conclusion}

    def paragraph_length_balance(self, paragraphs: List[str]) -> float:
        """Coefficient of variation of paragraph lengths (lower = more balanced)."""
        if len(paragraphs) < 2:
            return 0.0
        lengths = [len(p.split()) for p in paragraphs]
        mean = sum(lengths) / len(lengths)
        if mean == 0:
            return 0.0
        std = math.sqrt(sum((l - mean) ** 2 for l in lengths) / len(lengths))
        return round(std / mean, 4)  # CV

    # ─── MAIN EXTRACTION METHOD ──────────────────────────────────────────────

    def extract(self, processed: dict) -> Dict[str, Any]:
        """
        Extract all linguistic features from a preprocessed essay dict.
        Returns a flat dict of feature_name → value.
        """
        text = processed['normalized']
        words = processed['words_lower']
        sentences = processed['sentences']
        paragraphs = processed['paragraphs']
        spell_errors = processed['spelling_error_count']

        # Lexical
        wc = self.word_count(words)
        uw = self.unique_word_count(words)
        ld = self.lexical_density(words)
        cttr = self.corrected_ttr(words)
        awr = self.academic_word_ratio(words)
        awl = self.avg_word_length(words)
        hapax = self.hapax_legomena_ratio(words)
        dm_count = self.discourse_marker_count(text)

        # Syntactic
        sc = self.sentence_count(sentences)
        asl = self.avg_sentence_length(sentences, words)
        slv = self.sentence_length_variance(sentences)
        csr = self.complex_sentence_ratio(text, sentences)
        pos = self.pos_tag_features(text)

        # Grammatical / Mechanical
        sed = self.spelling_error_density(spell_errors, wc)
        pd = self.punctuation_density(text, wc)
        cap_errors = self.capitalization_errors(sentences)
        rep_ratio = self.repeated_word_ratio(words)

        # Readability
        readability = self.readability_scores(text)

        # Structural
        pc = self.paragraph_count(paragraphs)
        aspp = self.avg_sentences_per_paragraph(paragraphs, sentences)
        ic = self.has_intro_conclusion(paragraphs)
        plb = self.paragraph_length_balance(paragraphs)

        features = {
            # Lexical
            'word_count': wc,
            'unique_word_count': uw,
            'lexical_density': ld,
            'corrected_ttr': cttr,
            'academic_word_ratio': awr,
            'avg_word_length': awl,
            'hapax_legomena_ratio': hapax,
            'discourse_marker_count': dm_count,

            # Syntactic
            'sentence_count': sc,
            'avg_sentence_length': asl,
            'sentence_length_variance': slv,
            'complex_sentence_ratio': csr,
            **pos,

            # Grammatical
            'spelling_error_count': spell_errors,
            'spelling_error_density': sed,
            'punctuation_density': pd,
            'capitalization_errors': cap_errors,
            'repeated_word_ratio': rep_ratio,

            # Readability
            **readability,

            # Structural
            'paragraph_count': pc,
            'avg_sentences_per_paragraph': aspp,
            'has_intro': int(ic['has_intro']),
            'has_conclusion': int(ic['has_conclusion']),
            'paragraph_length_balance': plb,
        }
        return features
