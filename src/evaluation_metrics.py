"""
HFC-AES System Evaluation Metrics
Kwara State University, Malete — Faculty of ICT, Dept. of Computer Science

Evaluates the HFC-AES system using its own graded essay outputs.
No human scores or external CSV required — every essay graded by the
system is automatically recorded and measured.

The four metrics from §2.1.4 of the project report are applied using
TWO INTERNAL RATERS derived from the scoring pipeline itself:

    Rater A — Holistic Score (0–100%)
        The weighted neural-hybrid final score produced by fuse_and_score().

    Rater B — Trait-Average Score (0–100%)
        The simple unweighted mean of the six individual trait scores:
        Grammar, Vocabulary, Sentence Fluency, Organization,
        Content Development, and Readability.

This measures INTERNAL CONSISTENCY — how well the holistic scoring
pathway agrees with the independent trait-level evidence. A well-
calibrated system should show high agreement between these two raters.

Metrics (§2.1.4):
    1. QWK   — Quadratic Weighted Kappa (holistic vs trait-average)
    2. MAE   — Mean Absolute Error between the two raters (percentage points)
    3. RMSE  — Root Mean Square Error between the two raters
    4. r     — Pearson Correlation between holistic and trait-average series

Additional system-health metrics (computed from graded essays only):
    - Score distribution: mean, std, min, max, spread
    - Grade distribution: count and % per letter grade
    - Per-trait statistics: mean and std for each of the 6 traits
    - Confidence vs score correlation
    - Consistency index: % of essays where holistic and trait-avg
      agree within ±10 percentage points
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _trait_average(trait_scores: Dict[str, float]) -> float:
    """Unweighted mean of all trait scores, expressed as 0–100."""
    if not trait_scores:
        return 0.0
    return _mean([v * 100.0 for v in trait_scores.values()])


# ─────────────────────────────────────────────────────────────────────────────
# Four metric formulas (§2.1.4) — operating on 0–100 percentage values
# ─────────────────────────────────────────────────────────────────────────────

def _qwk(rater_a: List[float], rater_b: List[float]) -> float:
    """
    Quadratic Weighted Kappa between two internal raters (0–100 scale).

    Formula (§2.1.4):  kappa = 1 − Σ(w·O) / Σ(w·E)
    Scores are rounded to nearest integer (0–100) to build the rating grid.
    """
    n = len(rater_a)
    if n < 2:
        return 1.0

    # Round to nearest integer percentage for grid construction
    ra = [int(round(_clamp(v, 0, 100))) for v in rater_a]
    rb = [int(round(_clamp(v, 0, 100))) for v in rater_b]

    # Use only the levels that actually appear (keeps matrix sparse-safe)
    levels = sorted(set(ra) | set(rb))
    n_lev  = len(levels)
    if n_lev < 2:
        return 1.0 if ra == rb else 0.0

    idx = {lv: i for i, lv in enumerate(levels)}
    denom_w = (n_lev - 1) ** 2

    # Weight matrix
    W = [[(i - j) ** 2 / denom_w for j in range(n_lev)] for i in range(n_lev)]

    # Observed matrix
    O = [[0.0] * n_lev for _ in range(n_lev)]
    for a, b in zip(ra, rb):
        O[idx[a]][idx[b]] += 1.0

    row_sums = [sum(O[i]) for i in range(n_lev)]
    col_sums = [sum(O[i][j] for i in range(n_lev)) for j in range(n_lev)]

    # Expected matrix
    E = [[(row_sums[i] * col_sums[j]) / n for j in range(n_lev)]
         for i in range(n_lev)]

    num = sum(W[i][j] * O[i][j] for i in range(n_lev) for j in range(n_lev))
    den = sum(W[i][j] * E[i][j] for i in range(n_lev) for j in range(n_lev))

    return round(1.0 - num / den, 6) if den else 1.0


def _mae(rater_a: List[float], rater_b: List[float]) -> float:
    """MAE = (1/n) Σ |a_i − b_i|  (§2.1.4)"""
    return round(_mean([abs(a - b) for a, b in zip(rater_a, rater_b)]), 4)


def _rmse(rater_a: List[float], rater_b: List[float]) -> float:
    """RMSE = √[(1/n) Σ (a_i − b_i)²]  (§2.1.4)"""
    mse = _mean([(a - b) ** 2 for a, b in zip(rater_a, rater_b)])
    return round(math.sqrt(mse), 4)


def _pearson(rater_a: List[float], rater_b: List[float]) -> float:
    """Pearson r = Σ(a−ā)(b−b̄) / √[Σ(a−ā)² · Σ(b−b̄)²]  (§2.1.4)"""
    n = len(rater_a)
    if n < 2:
        return 0.0
    ma, mb = _mean(rater_a), _mean(rater_b)
    cov    = sum((a - ma) * (b - mb) for a, b in zip(rater_a, rater_b))
    std_a  = math.sqrt(sum((a - ma) ** 2 for a in rater_a))
    std_b  = math.sqrt(sum((b - mb) ** 2 for b in rater_b))
    if std_a == 0.0 or std_b == 0.0:
        return 0.0
    return round(cov / (std_a * std_b), 6)


# ─────────────────────────────────────────────────────────────────────────────
# GradedEssay — one record stored per essay the system grades
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GradedEssay:
    """
    Snapshot of a single essay graded by the HFC-AES system.
    Populated automatically from fuse_and_score() output in app.py.
    """
    essay_index:       int
    timestamp:         str
    prompt_id:         int
    holistic_pct:      float          # percentage_score from fuse_and_score
    trait_avg_pct:     float          # unweighted mean of trait_scores × 100
    final_score:       float          # ASAP-scale score
    score_min:         float
    score_max:         float
    grade_letter:      str
    confidence:        float          # 0–1 confidence estimate
    trait_scores:      Dict[str, float]  # raw 0–1 per trait
    word_count:        int            = 0

    @classmethod
    def from_score_result(
        cls,
        score_result: dict,
        essay_index: int,
        prompt_id: int,
        word_count: int = 0,
        timestamp: str = "",
    ) -> "GradedEssay":
        """Build a GradedEssay from the dict returned by fuse_and_score()."""
        traits = score_result.get("trait_scores", {})
        return cls(
            essay_index   = essay_index,
            timestamp     = timestamp or time.strftime("%H:%M:%S"),
            prompt_id     = prompt_id,
            holistic_pct  = float(score_result["percentage_score"]),
            trait_avg_pct = round(_trait_average(traits), 2),
            final_score   = float(score_result["final_score"]),
            score_min     = float(score_result.get("score_min", 0)),
            score_max     = float(score_result.get("score_max", 10)),
            grade_letter  = score_result.get("grade_letter", ""),
            confidence    = float(score_result.get("confidence", 0.0)),
            trait_scores  = {k: float(v) for k, v in traits.items()},
            word_count    = word_count,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SystemEvaluationReport — full report built from graded essays
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SystemEvaluationReport:
    """
    Evaluation report produced entirely from the system's own graded essays.

    Primary metrics (§2.1.4) compare Rater A (holistic) vs Rater B (trait-avg):
        qwk       — Quadratic Weighted Kappa
        mae       — Mean Absolute Error         (percentage points)
        rmse      — Root Mean Square Error      (percentage points)
        pearson_r — Pearson Correlation

    System-health metrics:
        consistency_index — % of essays where |holistic − trait_avg| ≤ 10 pp
        mean_holistic     — average holistic score across all essays
        std_holistic      — standard deviation of holistic scores
        mean_confidence   — average system confidence
        grade_distribution — { grade_letter: count }
        trait_means       — { trait_name: mean score (%) }
        trait_stds        — { trait_name: std score (%) }
    """
    # §2.1.4 metrics
    qwk:               float
    mae:               float
    rmse:              float
    pearson_r:         float

    # Session context
    n_essays:          int
    elapsed_s:         float          = 0.0

    # System-health metrics
    consistency_index: float          = 0.0   # % within ±10 pp
    mean_holistic:     float          = 0.0
    std_holistic:      float          = 0.0
    min_holistic:      float          = 0.0
    max_holistic:      float          = 0.0
    mean_trait_avg:    float          = 0.0
    mean_confidence:   float          = 0.0
    grade_distribution: Dict[str, int] = field(default_factory=dict)
    trait_means:       Dict[str, float] = field(default_factory=dict)
    trait_stds:        Dict[str, float] = field(default_factory=dict)

    # ── QWK interpretation (Landis & Koch, adapted for AES) ───────────────────
    _QWK_BANDS = [
        (0.80, "Excellent internal consistency"),
        (0.70, "Good — publication-quality consistency"),
        (0.60, "Moderate — acceptable consistency"),
        (0.40, "Fair — holistic and trait scores diverge"),
        (0.20, "Slight — system calibration needed"),
        (-1.0, "Poor — holistic and trait scores disagree strongly"),
    ]

    def qwk_interpretation(self) -> str:
        for threshold, label in self._QWK_BANDS:
            if self.qwk >= threshold:
                return label
        return "Poor"

    def summary(self) -> str:
        return (
            f"QWK={self.qwk:.4f}  MAE={self.mae:.2f}pp  "
            f"RMSE={self.rmse:.2f}pp  r={self.pearson_r:.4f}  "
            f"n={self.n_essays}  consistency={self.consistency_index:.1f}%"
        )

    def to_dict(self) -> dict:
        return {
            "QWK":                self.qwk,
            "MAE_pp":             self.mae,
            "RMSE_pp":            self.rmse,
            "Pearson_r":          self.pearson_r,
            "qwk_interpretation": self.qwk_interpretation(),
            "n_essays":           self.n_essays,
            "consistency_index":  self.consistency_index,
            "mean_holistic_pct":  self.mean_holistic,
            "std_holistic_pct":   self.std_holistic,
            "min_holistic_pct":   self.min_holistic,
            "max_holistic_pct":   self.max_holistic,
            "mean_trait_avg_pct": self.mean_trait_avg,
            "mean_confidence":    self.mean_confidence,
            "grade_distribution": self.grade_distribution,
            "trait_means":        self.trait_means,
            "trait_stds":         self.trait_stds,
            "elapsed_s":          self.elapsed_s,
        }


# ─────────────────────────────────────────────────────────────────────────────
# EvaluationMetrics — main public interface
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationMetrics:
    """
    Evaluates the HFC-AES system using essays it has already graded.

    Usage in app.py
    ---------------
    # After each essay is graded, record it:
    EvaluationMetrics.record(st.session_state, score_result, prompt_id, word_count)

    # In the Evaluation tab, compute and display the report:
    report = EvaluationMetrics.evaluate_session(st.session_state)
    """

    SESSION_KEY = "graded_essays"   # key used in st.session_state

    # ── Recording ─────────────────────────────────────────────────────────────

    @staticmethod
    def record(
        session_state,
        score_result: dict,
        prompt_id: int,
        word_count: int = 0,
    ) -> None:
        """
        Record a graded essay into session_state.
        Call this immediately after fuse_and_score() returns in app.py.
        """
        key = EvaluationMetrics.SESSION_KEY
        if key not in session_state:
            session_state[key] = []

        idx = len(session_state[key]) + 1
        essay = GradedEssay.from_score_result(
            score_result  = score_result,
            essay_index   = idx,
            prompt_id     = prompt_id,
            word_count    = word_count,
            timestamp     = time.strftime("%H:%M:%S"),
        )
        session_state[key].append(essay)

    @staticmethod
    def clear(session_state) -> None:
        """Wipe all recorded essays from session state."""
        session_state[EvaluationMetrics.SESSION_KEY] = []

    @staticmethod
    def get_essays(session_state) -> List[GradedEssay]:
        """Return the list of recorded GradedEssay objects."""
        return session_state.get(EvaluationMetrics.SESSION_KEY, [])

    # ── Core evaluation ───────────────────────────────────────────────────────

    @staticmethod
    def evaluate_session(session_state) -> "SystemEvaluationReport":
        """
        Compute all metrics from the essays graded in this session.
        Requires at least 2 graded essays.
        """
        essays = EvaluationMetrics.get_essays(session_state)
        return EvaluationMetrics.evaluate(essays)

    @staticmethod
    def evaluate(essays: List[GradedEssay]) -> "SystemEvaluationReport":
        """
        Compute the full system evaluation report from a list of GradedEssay records.

        §2.1.4 metrics use:
            Rater A = holistic_pct  (neural-hybrid weighted score)
            Rater B = trait_avg_pct (unweighted mean of 6 trait scores)
        """
        t0 = time.perf_counter()

        if len(essays) < 2:
            raise ValueError(
                f"At least 2 graded essays are required for evaluation "
                f"(currently have {len(essays)})."
            )

        rater_a = [e.holistic_pct  for e in essays]   # holistic
        rater_b = [e.trait_avg_pct for e in essays]   # trait average

        # ── Four §2.1.4 metrics ───────────────────────────────────────────────
        qwk   = _qwk(rater_a, rater_b)
        mae   = _mae(rater_a, rater_b)
        rmse  = _rmse(rater_a, rater_b)
        pr    = _pearson(rater_a, rater_b)

        # ── Consistency index ─────────────────────────────────────────────────
        within_10 = sum(1 for a, b in zip(rater_a, rater_b) if abs(a - b) <= 10.0)
        consistency = round(within_10 / len(essays) * 100, 1)

        # ── Score distribution ────────────────────────────────────────────────
        mean_h = round(_mean(rater_a), 2)
        std_h  = round(_std(rater_a),  2)
        min_h  = round(min(rater_a),   2)
        max_h  = round(max(rater_a),   2)
        mean_t = round(_mean(rater_b), 2)
        mean_c = round(_mean([e.confidence for e in essays]), 4)

        # ── Grade distribution ────────────────────────────────────────────────
        grade_dist: Dict[str, int] = {}
        for e in essays:
            g = e.grade_letter or "?"
            grade_dist[g] = grade_dist.get(g, 0) + 1

        # ── Per-trait statistics ──────────────────────────────────────────────
        all_traits = list(essays[0].trait_scores.keys()) if essays else []
        trait_means, trait_stds = {}, {}
        for trait in all_traits:
            vals = [e.trait_scores.get(trait, 0.0) * 100 for e in essays]
            trait_means[trait] = round(_mean(vals), 2)
            trait_stds[trait]  = round(_std(vals),  2)

        return SystemEvaluationReport(
            qwk               = qwk,
            mae               = mae,
            rmse              = rmse,
            pearson_r         = pr,
            n_essays          = len(essays),
            elapsed_s         = round(time.perf_counter() - t0, 4),
            consistency_index = consistency,
            mean_holistic     = mean_h,
            std_holistic      = std_h,
            min_holistic      = min_h,
            max_holistic      = max_h,
            mean_trait_avg    = mean_t,
            mean_confidence   = mean_c,
            grade_distribution = grade_dist,
            trait_means       = trait_means,
            trait_stds        = trait_stds,
        )
