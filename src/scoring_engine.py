"""
Feature Fusion and Scoring Engine
Combines linguistic and neural features to produce a final essay grade.
Uses a calibrated rule-based scoring model aligned with ASAP rubric standards.
"""

import math
import numpy as np
from typing import Dict, Any, Tuple


# ASAP Dataset score ranges per prompt
PROMPT_SCORE_RANGES = {
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60),
}

# Ideal essay characteristics for calibration
IDEAL_WORD_COUNT = {1: 400, 2: 400, 3: 180, 4: 180, 5: 180, 6: 180, 7: 300, 8: 700}
IDEAL_SENTENCE_COUNT = {1: 25, 2: 25, 3: 12, 4: 12, 5: 12, 6: 12, 7: 18, 8: 45}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class ScoringEngine:
    """
    Hybrid Feature Fusion Scoring Engine.

    Maps the fused vector of linguistic + neural features onto the ASAP rubric scale.
    Each trait is scored independently, then combined into a final holistic grade.
    """

    def __init__(self, prompt_id: int = 1):
        self.prompt_id = prompt_id
        self.score_min, self.score_max = PROMPT_SCORE_RANGES.get(prompt_id, (0, 10))
        self.score_range = self.score_max - self.score_min
        self.ideal_words = IDEAL_WORD_COUNT.get(prompt_id, 350)
        self.ideal_sentences = IDEAL_SENTENCE_COUNT.get(prompt_id, 20)

    def _score_grammar_mechanics(self, feats: Dict) -> float:
        """Score grammar and mechanical accuracy (0–1)."""
        # Spelling error density: 0 errors = perfect, >10/100 words = poor
        # (was /5.0 — too strict; even 3 errors/100 words halved the score)
        sed = feats.get('spelling_error_density', 0.0)
        cap_errors = feats.get('capitalization_errors', 0)
        sc = max(feats.get('sentence_count', 1), 1)

        grammar_score = clamp(1.0 - (sed / 10.0), 0.0, 1.0)
        # Require >50% of sentences to have cap errors before heavy penalty
        cap_score = clamp(1.0 - (cap_errors / (sc * 1.5)), 0.0, 1.0)
        return round((grammar_score * 0.7 + cap_score * 0.3), 4)

    def _score_vocabulary(self, feats: Dict) -> float:
        """Score vocabulary richness and sophistication (0–1)."""
        cttr = feats.get('corrected_ttr', 0.3)
        awr  = feats.get('academic_word_ratio', 0.05)   # ideal ≥ 0.07  (was 0.08)
        awl  = feats.get('avg_word_length', 4.0)         # ideal ≈ 6.0   (was 5.5)
        hapax = feats.get('hapax_legomena_ratio', 0.4)   # ideal ≥ 0.50  (was 0.65)

        cttr_score  = clamp(cttr / 0.45, 0.0, 1.0)          # was /0.6
        awr_score   = clamp(awr / 0.07,  0.0, 1.0)          # was /0.12 — 7% academic words = full marks
        awl_score   = clamp((awl - 3.0) / 3.0, 0.0, 1.0)   # was /4.0  — avg len 6.0 now hits max
        hapax_score = clamp(hapax / 0.50, 0.0, 1.0)         # was /0.65

        return round((cttr_score * 0.35 + awr_score * 0.25 + awl_score * 0.2 + hapax_score * 0.2), 4)

    def _score_sentence_fluency(self, feats: Dict) -> float:
        """Score sentence fluency and syntactic variety (0–1)."""
        asl = feats.get('avg_sentence_length', 15.0)     # ideal 12–30
        slv = feats.get('sentence_length_variance', 5.0)  # ideal > 3 (was > 4, but /10 was too strict)
        csr = feats.get('complex_sentence_ratio', 0.3)    # ideal 0.25–0.5

        # Penalise both very short and very long average sentence length
        if asl < 5:
            asl_score = 0.3
        elif asl <= 30:
            asl_score = 0.6 + (asl - 5) / 25 * 0.4
        else:
            asl_score = clamp(1.0 - (asl - 30) / 30, 0.4, 1.0)

        slv_score = clamp(slv / 5.5, 0.0, 1.0)    # was /10.0 — std-dev of 5.5 now hits full marks
        csr_score = clamp(csr / 0.40, 0.0, 1.0)   # was /0.5  — 40% complex sentences = full marks

        return round((asl_score * 0.4 + slv_score * 0.3 + csr_score * 0.3), 4)

    def _score_organization(self, feats: Dict) -> float:
        """Score essay organization and structure (0–1)."""
        pc          = feats.get('paragraph_count', 1)
        has_intro   = feats.get('has_intro', 0)
        has_conclusion = feats.get('has_conclusion', 0)
        dm          = feats.get('discourse_marker_count', 0)
        plb         = feats.get('paragraph_length_balance', 1.0)
        sc          = max(feats.get('sentence_count', 1), 1)

        # Ideal: 3–7 paragraphs (slightly wider window than before)
        if pc < 2:
            pc_score = 0.3
        elif pc <= 7:
            pc_score = 0.6 + (pc - 2) / 5 * 0.4
        else:
            pc_score = clamp(1.0 - (pc - 7) / 8, 0.55, 1.0)

        # Structural fallback: academic/journal writing rarely opens with "This essay…"
        # If the essay has 3+ paragraphs, grant structural credit even without keyword match.
        intro_credit      = bool(has_intro)      or (pc >= 3)
        conclusion_credit = bool(has_conclusion) or (pc >= 3)

        # Discourse marker density (normalised by sentence count)
        dm_score = min(dm / max(sc * 0.20, 1.0), 1.0)

        # Rebalanced: intro/conclusion each 30% (was 40%), DM raised to 40%
        structure_score = (int(intro_credit) * 0.30
                           + int(conclusion_credit) * 0.30
                           + dm_score * 0.40)
        balance_score = clamp(1.0 - plb, 0.0, 1.0)

        return round((pc_score * 0.30 + structure_score * 0.50 + balance_score * 0.20), 4)

    def _score_content_development(self, feats: Dict, neural_feats: Dict) -> float:
        """Score content depth and semantic development (0–1)."""
        wc = feats.get('word_count', 0)
        ideal = self.ideal_words
        coherence = neural_feats.get('scalar_features', {}).get('semantic_coherence', 0.5)
        rep_ratio = feats.get('repeated_word_ratio', 0.3)

        # Length adequacy
        if wc < 50:
            length_score = 0.1
        elif wc <= ideal:
            length_score = 0.4 + (wc / ideal) * 0.6
        else:
            # Mild penalty for excessive length
            length_score = clamp(1.0 - ((wc - ideal) / (ideal * 2)), 0.7, 1.0)

        coherence_score = clamp(coherence, 0.0, 1.0)
        rep_score = clamp(1.0 - rep_ratio * 2, 0.0, 1.0)

        return round((length_score * 0.4 + coherence_score * 0.4 + rep_score * 0.2), 4)

    def _score_readability(self, feats: Dict) -> float:
        """Score overall readability (0–1) based on textstat metrics.

        Academic / journal text typically has Flesch Reading Ease of 20–55
        (harder to read) which is *appropriate* for the audience.  The old
        calibration rewarded FRE 60–80 (newspaper/magazine level), which
        systematically penalised university-level writing.
        """
        fre = feats.get('flesch_reading_ease', 50.0)
        fkg = feats.get('flesch_kincaid_grade', 8.0)

        if fre == 0 and fkg == 0:
            return 0.65  # no readability data — give neutral score

        # FRE: reward academic range 20–75; don't penalise difficult prose
        if fre < 10:
            fre_score = 0.30
        elif fre <= 75:
            fre_score = 0.50 + (fre - 10) / 65 * 0.50
        elif fre <= 90:
            fre_score = 1.0
        else:
            fre_score = clamp(1.0 - (fre - 90) / 30, 0.55, 1.0)

        # FKG: ideal grade-level range 8–16 for university essays (was 8–14)
        if fkg < 5:
            fkg_score = 0.45
        elif fkg <= 16:
            fkg_score = 0.50 + (fkg - 5) / 11 * 0.50
        else:
            fkg_score = clamp(1.0 - (fkg - 16) / 12, 0.50, 1.0)

        return round((fre_score * 0.5 + fkg_score * 0.5), 4)

    def fuse_and_score(
        self,
        linguistic_feats: Dict[str, Any],
        neural_feats: Dict[str, Any],
        prompt_id: int = None
    ) -> Dict[str, Any]:
        """
        Fuse linguistic and neural features into a final essay score.
        Returns trait scores, final grade, confidence, and feature vector.
        """
        if prompt_id is not None and prompt_id != self.prompt_id:
            self.prompt_id = prompt_id
            self.score_min, self.score_max = PROMPT_SCORE_RANGES.get(prompt_id, (0, 10))
            self.score_range = self.score_max - self.score_min
            self.ideal_words = IDEAL_WORD_COUNT.get(prompt_id, 350)

        # ── Trait Scores (each 0–1) ──────────────────────────────────────────
        traits = {
            'Grammar & Mechanics': self._score_grammar_mechanics(linguistic_feats),
            'Vocabulary Sophistication': self._score_vocabulary(linguistic_feats),
            'Sentence Fluency': self._score_sentence_fluency(linguistic_feats),
            'Organization & Structure': self._score_organization(linguistic_feats),
            'Content Development': self._score_content_development(linguistic_feats, neural_feats),
            'Readability': self._score_readability(linguistic_feats),
        }

        # ── Weighted holistic score (0–1) ────────────────────────────────────
        # Weights reflect typical rubric emphasis
        weights = {
            'Grammar & Mechanics': 0.15,
            'Vocabulary Sophistication': 0.20,
            'Sentence Fluency': 0.15,
            'Organization & Structure': 0.20,
            'Content Development': 0.20,
            'Readability': 0.10,
        }
        holistic_raw = sum(traits[t] * weights[t] for t in traits)

        # Neural pathway bonus: only add when coherence is above neutral;
        # TF-IDF fallback coherence is typically < 0.5 and must never penalise.
        coherence = neural_feats.get('scalar_features', {}).get('semantic_coherence', 0.5)
        neural_bonus = max((coherence - 0.5) * 0.10, 0.0)  # 0 → +5% only
        holistic = clamp(holistic_raw + neural_bonus, 0.0, 1.0)

        # ── Minimum floor for substantive essays ─────────────────────────────
        # Prevents well-formed essays from being dragged to F by a single weak
        # feature.  Floor rises with word count.
        wc = linguistic_feats.get('word_count', 0)
        if wc >= 200:
            holistic = max(holistic, 0.45)
        elif wc >= 100:
            holistic = max(holistic, 0.40)

        # ── Map to ASAP score range ───────────────────────────────────────────
        raw_score = self.score_min + holistic * self.score_range
        # Round to nearest valid score
        raw_score = round(raw_score, 1)
        final_score = clamp(raw_score, self.score_min, self.score_max)

        # ── Confidence score (proxy: higher when features agree) ─────────────
        trait_values = list(traits.values())
        trait_std = np.std(trait_values)
        confidence = clamp(0.85 - trait_std * 0.5, 0.50, 0.95)

        # ── Percentage score ──────────────────────────────────────────────────
        pct_score = round(holistic * 100, 1)

        return {
            'final_score': round(final_score, 1),
            'score_min': self.score_min,
            'score_max': self.score_max,
            'percentage_score': pct_score,
            'holistic_normalized': round(holistic, 4),
            'confidence': round(confidence, 3),
            'trait_scores': traits,
            'weights': weights,
            'neural_bonus': round(neural_bonus, 4),
            'prompt_id': self.prompt_id,
        }

    def batch_evaluate(
        self,
        essays: list,
        preprocessor,
        linguistic_extractor,
        neural_extractor,
        prompt_id: int = None,
    ) -> list:
        """
        Score a batch of essays and return a list of prediction dicts
        ready for EvaluationMetrics.evaluate_predictions().

        Each item in `essays` must be a dict with keys:
            ``text``        — raw essay text
            ``human_score`` — expert-assigned ground-truth score

        Returns a list of dicts:
            {
                "human_score":     float,
                "predicted_score": float,
                "percentage_score": float,
                "grade_letter":    str,
                "essay_index":     int,
            }
        """
        from src.evaluation_metrics import EvaluationMetrics  # lazy import

        pid = prompt_id or self.prompt_id
        results = []

        for idx, item in enumerate(essays):
            try:
                text = item.get("text", "")
                human_score = float(item.get("human_score", 0))

                # Run full pipeline
                processed   = preprocessor.process(text)
                ling_feats  = linguistic_extractor.extract(processed)
                neural_feats = neural_extractor.extract(processed)
                score_result = self.fuse_and_score(ling_feats, neural_feats, pid)

                grade_letter, _ = self.get_grade_label(score_result["percentage_score"])

                results.append({
                    "essay_index":      idx,
                    "human_score":      human_score,
                    "predicted_score":  score_result["final_score"],
                    "percentage_score": score_result["percentage_score"],
                    "grade_letter":     grade_letter,
                    "confidence":       score_result["confidence"],
                })
            except Exception as e:
                results.append({
                    "essay_index":     idx,
                    "human_score":     float(item.get("human_score", 0)),
                    "predicted_score": float(self.score_min),
                    "percentage_score": 0.0,
                    "grade_letter":    "F",
                    "confidence":      0.0,
                    "error":           str(e),
                })

        return results

    def get_grade_label(self, pct_score: float) -> Tuple[str, str]:
        """Map percentage score to letter grade and label.

        Thresholds lowered by ~5 pts to reflect the recalibrated scoring
        model, which is more lenient on academic / journal-style writing.
        """
        if pct_score >= 85:
            return "A", "Excellent"
        elif pct_score >= 72:
            return "B", "Good"
        elif pct_score >= 58:
            return "C", "Satisfactory"
        elif pct_score >= 45:
            return "D", "Below Average"
        else:
            return "F", "Needs Significant Improvement"
