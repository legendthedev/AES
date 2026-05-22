"""
HFC-AES: Text-Based Automated Essay Grading System
Kwara State University, Malete — Faculty of ICT, Dept. of Computer Science

Streamlit Application — Main Entry Point
"""

import warnings

# Suppress the specific path access deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

import sys
import os
import time
import io

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.preprocessor import TextPreprocessor
from src.linguistic_features import LinguisticFeatureExtractor
from src.neural_features import NeuralFeatureExtractor
from src.scoring_engine import ScoringEngine, PROMPT_SCORE_RANGES
from src.feedback_engine import FeedbackEngine
from src.evaluation_metrics import EvaluationMetrics, SystemEvaluationReport

import spacy

nlp = spacy.load("en_core_web_sm")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HFC-AES | Essay Grading System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Core palette ── */
:root {
    --green-dark:   #0d5c30;
    --green-mid:    #1a7a4a;
    --green-light:  #2ea868;
    --green-pale:   #d6e8dc;
    --gold:         #c8973a;
    --gold-light:   #f5dea0;
    --text-dark:    #1a2e22;
    --text-mid:     #3a5a45;
    --bg-main:      #f0f4f1;
    --bg-card:      #ffffff;
    --border:       #b8d4c0;
}

/* ── App background ── */
.stApp { background-color: var(--bg-main); }

/* ── Header banner ── */
.kwasu-header {
    background: linear-gradient(135deg, var(--green-dark) 0%, var(--green-mid) 60%, var(--green-light) 100%);
    padding: 28px 36px 22px;
    border-radius: 14px;
    margin-bottom: 28px;
    box-shadow: 0 4px 20px rgba(13,92,48,0.25);
}
.kwasu-header h1 {
    color: #ffffff !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    margin: 0 0 4px 0 !important;
    letter-spacing: -0.5px;
}
.kwasu-header p {
    color: rgba(255,255,255,0.85) !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
}
.kwasu-badge {
    display: inline-block;
    background: var(--gold);
    color: #fff;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 22px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    height: 100%;
}
.metric-card .big-num {
    font-size: 2.8rem;
    font-weight: 900;
    color: var(--green-dark);
    line-height: 1;
    margin-bottom: 6px;
}
.metric-card .label {
    font-size: 0.82rem;
    color: var(--text-mid);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.metric-card .sub {
    font-size: 0.78rem;
    color: #888;
    margin-top: 4px;
}

/* ── Grade badge ── */
.grade-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 80px;
    height: 80px;
    border-radius: 50%;
    font-size: 2.2rem;
    font-weight: 900;
    margin: 8px auto;
}
.grade-A { background: #d4edda; color: #155724; border: 3px solid #28a745; }
.grade-B { background: #d1ecf1; color: #0c5460; border: 3px solid #17a2b8; }
.grade-C { background: #fff3cd; color: #856404; border: 3px solid #ffc107; }
.grade-D { background: #ffe5d0; color: #7d3c00; border: 3px solid #fd7e14; }
.grade-F { background: #f8d7da; color: #721c24; border: 3px solid #dc3545; }

/* ── Trait bars ── */
.trait-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}
.trait-label {
    min-width: 200px;
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text-dark);
}
.trait-bar-bg {
    flex: 1;
    height: 14px;
    background: #e0ece4;
    border-radius: 7px;
    overflow: hidden;
}
.trait-bar-fill {
    height: 100%;
    border-radius: 7px;
    transition: width 0.6s ease;
}

/* ── Feedback cards ── */
.fb-card {
    background: var(--bg-card);
    border-left: 5px solid var(--green-mid);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 1px 5px rgba(0,0,0,0.05);
}
.fb-card.excellent { border-color: #28a745; background: #f8fff9; }
.fb-card.good      { border-color: #17a2b8; background: #f5fcff; }
.fb-card.satisfactory { border-color: #ffc107; background: #fffdf0; }
.fb-card.poor      { border-color: #dc3545; background: #fff5f5; }

.fb-title { font-weight: 700; font-size: 1rem; color: var(--text-dark); margin-bottom: 6px; }
.fb-summary { font-size: 0.9rem; color: var(--text-mid); margin-bottom: 8px; }

/* ── Insight chips ── */
.insight-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.83rem;
    font-weight: 600;
    margin: 4px;
}
.chip-good { background: #d4edda; color: #155724; }
.chip-warn { background: #fff3cd; color: #856404; }
.chip-poor { background: #f8d7da; color: #721c24; }

/* ── Highlighted sentence ── */
.hl-sentence {
    background: linear-gradient(90deg, #d4edda, #f0f9f3);
    border-left: 4px solid var(--green-mid);
    padding: 10px 14px;
    border-radius: 6px;
    margin: 6px 0;
    font-size: 0.9rem;
    color: var(--text-dark);
    font-style: italic;
}

/* ── Section headers ── */
.section-head {
    font-size: 1.15rem;
    font-weight: 800;
    color: var(--green-dark);
    padding-bottom: 6px;
    border-bottom: 2px solid var(--green-pale);
    margin-bottom: 18px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d5c30 0%, #1a7a4a 100%) !important;
}
section[data-testid="stSidebar"] * { color: #e8f5ec !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label { color: #b8e0c8 !important; font-weight: 600; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--green-dark), var(--green-mid)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 12px 32px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 3px 12px rgba(13,92,48,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(13,92,48,0.45) !important;
}

/* ── Text area ── */
.stTextArea textarea {
    border: 2px solid var(--border) !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
    line-height: 1.65 !important;
    background: #fafcfb !important;
}
.stTextArea textarea:focus {
    border-color: var(--green-mid) !important;
    box-shadow: 0 0 0 3px rgba(26,122,74,0.15) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--green-pale);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: var(--text-mid) !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--green-mid) !important;
    color: white !important;
}

/* ── Progress bar ── */
.stProgress > div > div { background: var(--green-mid) !important; }

/* ── Word counter ── */
.word-counter {
    text-align: right;
    font-size: 0.82rem;
    color: var(--text-mid);
    padding: 4px 0;
}
.word-ok   { color: var(--green-mid); font-weight: 700; }
.word-warn { color: #d97706; font-weight: 700; }
.word-low  { color: #dc3545; font-weight: 700; }

/* ── Dividers ── */
hr { border-color: var(--green-pale) !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--green-pale) !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    color: var(--green-dark) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── SESSION STATE ────────────────────────────────────────────────────────────

if 'history' not in st.session_state:
    st.session_state.history = []  # list of (pct_score, timestamp)
if 'prev_score' not in st.session_state:
    st.session_state.prev_score = None


# ─── CACHED COMPONENTS ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline():
    preprocessor = TextPreprocessor()
    ling_extractor = LinguisticFeatureExtractor()
    neural_extractor = NeuralFeatureExtractor(use_deberta=True)
    feedback_engine = FeedbackEngine()
    return preprocessor, ling_extractor, neural_extractor, feedback_engine


# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def get_bar_color(score: float) -> str:
    if score >= 0.80:
        return "#28a745"
    elif score >= 0.65:
        return "#1a7a4a"
    elif score >= 0.45:
        return "#ffc107"
    else:
        return "#dc3545"


def render_trait_bar(name: str, score: float, weight: float):
    pct = score * 100
    color = get_bar_color(score)
    st.markdown(f"""
    <div class="trait-row">
        <div class="trait-label">{name}</div>
        <div class="trait-bar-bg">
            <div class="trait-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
        </div>
        <div style="min-width:52px;text-align:right;font-size:0.88rem;font-weight:700;color:{color}">
            {pct:.1f}%
        </div>
        <div style="min-width:36px;text-align:right;font-size:0.75rem;color:#888;">{int(weight*100)}%w</div>
    </div>
    """, unsafe_allow_html=True)


def render_insight_chip(insight: dict):
    chip_class = f"chip-{insight['status']}"
    st.markdown(
        f'<span class="insight-chip {chip_class}">{insight["icon"]} '
        f'<strong>{insight["metric"]}:</strong> {insight["value"]} — {insight["note"]}</span>',
        unsafe_allow_html=True
    )


def _clean_pdf_text(raw: str) -> str:
    """
    Clean and normalise raw text extracted from a PDF.

    PDFs often contain:
    - Hyphenated line-breaks  (e.g. "organ-\nisation" → "organisation")
    - Mid-word newlines       (e.g. "the\ncat" → "the cat")
    - Multiple blank lines    collapsed to one paragraph break
    - Leading/trailing spaces on every line
    """
    import re
    # 1. Re-join hyphenated line-breaks: "organ-\nisation" → "organisation"
    raw = re.sub(r'-\n(\S)', r'\1', raw)
    # 2. Join lines that are part of the same sentence (no blank line between)
    #    A blank line (two+ newlines) marks a paragraph boundary — keep it.
    raw = re.sub(r'(?<!\n)\n(?!\n)', ' ', raw)
    # 3. Collapse multiple spaces
    raw = re.sub(r'[ \t]+', ' ', raw)
    # 4. Collapse 3+ consecutive newlines to exactly two (paragraph break)
    raw = re.sub(r'\n{3,}', '\n\n', raw)
    # 5. Strip leading/trailing whitespace on each line
    lines = [line.strip() for line in raw.splitlines()]
    return '\n'.join(lines).strip()


def read_uploaded_file(uploaded_file) -> str:
    """
    Extract plain text from an uploaded PDF, DOCX, or TXT file.

    PDF extraction strategy (most → least reliable):
      1. pypdf  — fast, handles most modern PDFs
      2. pdfplumber  — slower but recovers text from complex layouts
      If both return < 50 words, warn the user (likely a scanned/image PDF).
    """
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()

    try:
        if name.endswith('.pdf'):
            text = ""

            # ── Strategy 1: pypdf (successor to PyPDF2) ──────────────────────
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(raw_bytes))
                pages_text = []
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    pages_text.append(page_text)
                text = "\n\n".join(pages_text)
            except Exception:
                text = ""

            # ── Strategy 2: pdfplumber fallback ──────────────────────────────
            if len(text.split()) < 50:
                try:
                    import pdfplumber
                    pages_text = []
                    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text() or ""
                            pages_text.append(page_text)
                    plumber_text = "\n\n".join(pages_text)
                    if len(plumber_text.split()) > len(text.split()):
                        text = plumber_text
                except Exception:
                    pass  # keep whatever pypdf gave us

            text = _clean_pdf_text(text)

            if len(text.split()) < 20:
                st.warning(
                    "⚠️ Very little text could be extracted from this PDF. "
                    "It may be a scanned/image-based PDF. "
                    "Please try copying the text and using the **Type / Paste Essay** tab instead."
                )
            return text

        elif name.endswith('.docx'):
            from docx import Document
            doc = Document(io.BytesIO(raw_bytes))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

        else:
            return raw_bytes.decode('utf-8', errors='ignore')

    except Exception as e:
        st.error(f"Could not read file: {e}")
        return ""


def plot_radar(trait_scores: dict) -> go.Figure:
    names = list(trait_scores.keys())
    values = [v * 100 for v in trait_scores.values()]
    values_closed = values + [values[0]]
    names_closed = names + [names[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=names_closed,
        fill='toself',
        fillcolor='rgba(26,122,74,0.18)',
        line=dict(color='#1a7a4a', width=2.5),
        marker=dict(size=7, color='#1a7a4a'),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(size=10), gridcolor='#c8e0d0'),
            angularaxis=dict(tickfont=dict(size=11, color='#1a2e22')),
            bgcolor='rgba(240,244,241,0.5)',
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=30),
        height=360,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def plot_progress(history):
    if len(history) < 2:
        return None
    df = pd.DataFrame(history, columns=['Score', 'Label'])
    df['Attempt'] = range(1, len(df) + 1)
    fig = px.line(df, x='Attempt', y='Score', markers=True,
                  title='Score Progress Across Submissions',
                  color_discrete_sequence=['#1a7a4a'])
    fig.update_traces(line_width=2.5, marker_size=8)
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(range=[0, 100], gridcolor='#e0ece4'),
        xaxis=dict(gridcolor='#e0ece4'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,244,241,0.5)',
    )
    return fig


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🎓 HFC-AES System")
    st.markdown("---")

    st.markdown("**📝 Essay Configuration**")
    prompt_id = st.selectbox(
        "ASAP Prompt ID",
        options=list(PROMPT_SCORE_RANGES.keys()),
        format_func=lambda x: f"Prompt {x} — {['Persuasive/Arg (Gr.8)', 'Persuasive/Arg (Gr.10)', 'Source-Dep (Gr.10)', 'Source-Dep (Gr.10)', 'Source-Dep (Gr.8)', 'Source-Dep (Gr.10)', 'Narrative (Gr.7)', 'Narrative (Gr.10)'][x-1]}",
        help="Select the ASAP writing prompt your essay responds to."
    )
    score_min, score_max = PROMPT_SCORE_RANGES[prompt_id]
    st.caption(f"Score range for Prompt {prompt_id}: **{score_min} – {score_max}**")

    st.markdown("---")
    st.markdown("**⚙️ Analysis Settings**")
    use_deberta = st.toggle("Use DeBERTa-v3 Neural Model", value=True,
                             help="Enable deep semantic analysis (requires transformers + torch)")
    show_heatmap = st.toggle("Show Semantic Highlights", value=True,
                              help="Highlight semantically strong sentences")
    show_raw_metrics = st.toggle("Show Raw Feature Metrics", value=False,
                                  help="Display all extracted linguistic feature values")

    st.markdown("---")
    st.markdown("**ℹ️ About**")
    st.markdown("""
    <small>
    <b>HFC-AES</b> — Hybrid Feature-based Cross-prompt Automated Essay Scoring System<br><br>
    Combines <b>DeBERTa-v3</b> neural embeddings with handcrafted linguistic features to produce 
    interpretable, rubric-aligned essay scores.<br><br>
    <i>Kwara State University, Malete<br>Dept. of Computer Science, 2024/2025</i>
    </small>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("---")
        st.markdown("**📊 Session History**")
        scores = [s for s, _ in st.session_state.history]
        st.caption(f"Submissions: **{len(scores)}** | Best: **{max(scores):.1f}%** | Avg: **{sum(scores)/len(scores):.1f}%**")
        if len(scores) >= 2:
            fig_prog = plot_progress(st.session_state.history)
            if fig_prog:
                st.plotly_chart(fig_prog, use_container_width=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.session_state.prev_score = None
            st.rerun()


# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="kwasu-header">
    <div class="kwasu-badge">HFC-AES v1.0 · KWASU ICT</div>
    <h1>🎓 Automated Essay Grading System</h1>
    <p>Hybrid Feature-based Cross-prompt Model · DeBERTa-v3 + Linguistic Analysis · Explainable AI Feedback</p>
</div>
""", unsafe_allow_html=True)


# ─── INPUT SECTION ────────────────────────────────────────────────────────────

st.markdown('<div class="section-head">📄 Essay Submission</div>', unsafe_allow_html=True)

tab_text, tab_file, tab_eval = st.tabs(["✍️ Type / Paste Essay", "📎 Upload File (PDF / DOCX / TXT)", "📊 System Evaluation Metrics"])

essay_text = ""

with tab_text:
    col_input, col_hint = st.columns([3, 1])
    with col_input:
        raw_input = st.text_area(
            "Paste or type your essay below:",
            height=300,
            placeholder="Begin typing or paste your essay here...\n\nThe system will analyse grammar, vocabulary, sentence structure, organisation, content depth, and overall coherence.",
            key="essay_input",
            label_visibility="collapsed"
        )
    with col_hint:
        st.markdown("**💡 Tips**")
        s_min, s_max = PROMPT_SCORE_RANGES[prompt_id]
        st.info(f"""
**Prompt {prompt_id}**
Score: {s_min}–{s_max}

**Graded Traits:**
- Grammar & Mechanics
- Vocabulary
- Sentence Fluency
- Organisation
- Content Depth
- Readability
        """)
    if raw_input:
        wc = len(raw_input.split())
        ideal = {1:350, 2:350, 3:150, 4:150, 5:150, 6:150, 7:250, 8:650}.get(prompt_id, 300)
        color_cls = "word-ok" if wc >= ideal*0.8 else "word-warn" if wc >= 50 else "word-low"
        st.markdown(f'<div class="word-counter"><span class="{color_cls}">{wc} words</span> · Target: ~{ideal} words</div>', unsafe_allow_html=True)
    essay_text = raw_input

with tab_file:
    uploaded = st.file_uploader(
        "Upload your essay (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        help="The system will extract the text and analyse it automatically."
    )
    if uploaded:
        with st.spinner("Extracting text from file..."):
            essay_text = read_uploaded_file(uploaded)
        if essay_text:
            st.success(f"✅ Extracted {len(essay_text.split())} words from **{uploaded.name}**")
            with st.expander("Preview extracted text"):
                st.text(essay_text[:800] + ("..." if len(essay_text) > 800 else ""))

st.markdown("<br>", unsafe_allow_html=True)
col_btn, col_clear = st.columns([2, 1])
with col_btn:
    evaluate = st.button("🔍 Evaluate Essay", use_container_width=True, type="primary")
with col_clear:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.essay_input = ""
        st.rerun()


# ─── EVALUATION PIPELINE ─────────────────────────────────────────────────────

if evaluate:
    if not essay_text or len(essay_text.split()) < 20:
        st.warning("⚠️ Please provide an essay with at least 20 words before evaluating.")
        st.stop()

    preprocessor, ling_extractor, neural_extractor, feedback_engine = load_pipeline()

    # Override neural setting from sidebar
    if not use_deberta:
        neural_extractor.use_deberta = False
        neural_extractor.deberta_available = False

    results_placeholder = st.empty()

    with st.status("🔬 Analysing your essay...", expanded=True) as status:
        # Step 1: Preprocessing
        st.write("🧹 Cleaning and normalizing text...")
        time.sleep(0.3)
        processed = preprocessor.process(essay_text)

        # Step 2: Linguistic features
        st.write("📐 Extracting linguistic features (grammar, vocabulary, structure)...")
        time.sleep(0.4)
        linguistic_feats = ling_extractor.extract(processed)

        # Step 3: Neural features
        model_label = "DeBERTa-v3" if (neural_extractor.deberta_available and use_deberta) else "TF-IDF + LSA"
        st.write(f"🧠 Computing semantic embeddings ({model_label})...")
        time.sleep(0.3)
        neural_feats = neural_extractor.compute_semantic_features(processed['normalized'])

        # Step 4: Semantic highlights
        if show_heatmap:
            st.write("🔦 Identifying semantically strong sentences...")
            highlights = neural_extractor.get_attention_highlights(processed['normalized'], top_n=3)
        else:
            highlights = []

        # Step 5: Scoring
        st.write("🏆 Computing holistic score via feature fusion...")
        time.sleep(0.3)
        scorer = ScoringEngine(prompt_id=prompt_id)
        score_result = scorer.fuse_and_score(linguistic_feats, neural_feats, prompt_id)

        # Step 6: Feedback
        st.write("💬 Generating XAI diagnostic feedback...")
        time.sleep(0.2)
        feedback = feedback_engine.generate(score_result, linguistic_feats, neural_feats, highlights)

        status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

    # ─── Store history & evaluation record ──────────────────────────────────
    pct = score_result['percentage_score']
    st.session_state.prev_score = st.session_state.history[-1][0] if st.session_state.history else None
    st.session_state.history.append((pct, time.strftime("%H:%M")))

    grade_letter, grade_label = scorer.get_grade_label(pct)
    score_result['grade_letter'] = grade_letter   # stamp for EvaluationMetrics

    # Record this essay for the Evaluation Metrics tab
    EvaluationMetrics.record(
        st.session_state,
        score_result,
        prompt_id,
        word_count=linguistic_feats.get('word_count', 0),
    )
    delta = pct - st.session_state.prev_score if st.session_state.prev_score is not None else None

    # ─────────────────────────────────────────────────────────────────────────
    # RESULTS SECTION
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown('<div class="section-head">📊 Results Dashboard</div>', unsafe_allow_html=True)

    # ── Top Score Cards ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    final_score = score_result['final_score']
    confidence = score_result['confidence']
    holistic = score_result['holistic_normalized']

    with c1:
        delta_html = f'<div class="sub">{"▲" if delta and delta > 0 else "▼" if delta and delta < 0 else ""} {abs(delta):.1f}% vs last</div>' if delta is not None else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-num">{pct:.0f}<span style="font-size:1.2rem">%</span></div>
            <div class="label">Overall Score</div>
            {delta_html}
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex;justify-content:center;margin:4px 0 8px">
                <div class="grade-badge grade-{grade_letter}">{grade_letter}</div>
            </div>
            <div class="label">{grade_label}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-num">{final_score}</div>
            <div class="label">ASAP Score</div>
            <div class="sub">Range: {score_result['score_min']} – {score_result['score_max']}</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-num">{int(confidence*100)}<span style="font-size:1.2rem">%</span></div>
            <div class="label">AI Confidence</div>
            <div class="sub">{model_label}</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        wc_display = linguistic_feats.get('word_count', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-num">{wc_display}</div>
            <div class="label">Word Count</div>
            <div class="sub">{linguistic_feats.get('sentence_count', 0)} sentences</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Holistic Feedback Banner ──────────────────────────────────────────────
    tier_colors = {'excellent': '#28a745', 'good': '#17a2b8',
                   'satisfactory': '#ffc107', 'poor': '#dc3545'}
    tier_icons = {'excellent': '🏆', 'good': '✅', 'satisfactory': '📋', 'poor': '⚠️'}
    h_tier = feedback['holistic_tier']
    hmsgs = feedback['holistic_messages']
    if hmsgs:
        st.markdown(f"""
        <div style="background:{tier_colors[h_tier]}18;border:2px solid {tier_colors[h_tier]};
                    border-radius:12px;padding:18px 22px;margin-bottom:20px;">
            <b style="font-size:1.05rem;color:{tier_colors[h_tier]};">{tier_icons[h_tier]} Overall Assessment</b><br>
            {'<br>'.join(f'<span style="color:#1a2e22;font-size:0.92rem">• {m}</span>' for m in hmsgs)}
        </div>
        """, unsafe_allow_html=True)

    # ── Strengths / Weaknesses ────────────────────────────────────────────────
    if feedback['strengths'] or feedback['weaknesses']:
        ca, cb = st.columns(2)
        with ca:
            if feedback['strengths']:
                st.markdown("**💪 Key Strengths**")
                for s in feedback['strengths']:
                    st.markdown(f'<span class="insight-chip chip-good">✓ {s}</span>', unsafe_allow_html=True)
        with cb:
            if feedback['weaknesses']:
                st.markdown("**🎯 Priority Areas for Improvement**")
                for w in feedback['weaknesses']:
                    st.markdown(f'<span class="insight-chip chip-warn">→ {w}</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column: Radar + Trait Bars ────────────────────────────────────────
    col_radar, col_bars = st.columns([1, 1])

    with col_radar:
        st.markdown('<div class="section-head">🕸️ Writing Profile</div>', unsafe_allow_html=True)
        fig_radar = plot_radar(score_result['trait_scores'])
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_bars:
        st.markdown('<div class="section-head">📏 Trait Breakdown</div>', unsafe_allow_html=True)
        for trait_name, trait_score in score_result['trait_scores'].items():
            render_trait_bar(trait_name, trait_score, score_result['weights'][trait_name])

    # ── Metric Insights ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-head">🔬 Linguistic Metric Insights</div>', unsafe_allow_html=True)
    insight_cols = st.columns(2)
    insights = feedback['metric_insights']
    for i, insight in enumerate(insights):
        with insight_cols[i % 2]:
            render_insight_chip(insight)

    # ─────────────────────────────────────────────────────────────────────────
    # XAI DIAGNOSTIC FEEDBACK (Expanders)
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown('<div class="section-head">💬 Diagnostic Feedback (XAI Layer)</div>', unsafe_allow_html=True)
    st.caption("Click each category to see detailed AI-generated feedback grounded in your essay's linguistic features.")

    tier_emoji = {'excellent': '🟢', 'good': '🔵', 'satisfactory': '🟡', 'poor': '🔴'}

    for trait_name, tf in feedback['trait_feedback'].items():
        emoji = tier_emoji.get(tf['tier'], '⚪')
        score_display = f"{tf['score_pct']:.0f}%"
        with st.expander(f"{emoji} {trait_name} — {score_display} ({tf['tier'].capitalize()})"):
            st.markdown(f"""
            <div class="fb-card {tf['tier']}">
                <div class="fb-title">{trait_name}</div>
                <div class="fb-summary">{tf['summary']}</div>
            </div>""", unsafe_allow_html=True)

            if tf['details']:
                st.markdown("**What the AI detected:**")
                for d in tf['details']:
                    st.markdown(f"- {d}")

            if tf['suggestions']:
                st.markdown("**📌 Suggestions for improvement:**")
                for s in tf['suggestions']:
                    st.markdown(f"- {s}")

    # ── Semantic Highlights ───────────────────────────────────────────────────
    if show_heatmap and feedback['highlighted_sentences']:
        st.markdown("---")
        with st.expander("🧠 Semantic Highlights — Strongest Sentences (AI-Identified)"):
            st.caption(
                f"The {model_label} model identified these sentences as semantically rich. "
                "Understanding what makes them strong can help you replicate this quality throughout."
            )
            for i, sent in enumerate(feedback['highlighted_sentences'], 1):
                st.markdown(f'<div class="hl-sentence">#{i}: "{sent.strip()}"</div>', unsafe_allow_html=True)

    # ── Raw Feature Metrics ───────────────────────────────────────────────────
    if show_raw_metrics:
        st.markdown("---")
        with st.expander("🔩 Raw Feature Vector (Advanced)"):
            st.caption("All handcrafted linguistic features extracted during the analysis.")
            feat_df = pd.DataFrame([
                {"Feature": k, "Value": round(v, 4) if isinstance(v, float) else v}
                for k, v in linguistic_feats.items()
                if not isinstance(v, list)
            ])
            st.dataframe(feat_df, use_container_width=True, height=400)

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("---")
    col_dl1, col_dl2, _ = st.columns([1, 1, 2])
    with col_dl1:
        report_text = f"""HFC-AES ESSAY GRADING REPORT
{'='*50}
Date: {time.strftime('%Y-%m-%d %H:%M')}
Prompt ID: {prompt_id}
Grade: {grade_letter} ({grade_label})
Overall Score: {pct:.1f}%
ASAP Score: {final_score} / {score_result['score_max']}
AI Confidence: {int(confidence*100)}%
Model Used: {model_label}
Word Count: {linguistic_feats.get('word_count', 0)}

TRAIT SCORES
{'-'*30}
""" + "\n".join(f"{t}: {s*100:.1f}%" for t, s in score_result['trait_scores'].items()) + f"""

HOLISTIC FEEDBACK
{'-'*30}
""" + "\n".join(hmsgs) + f"""

KEY SUGGESTIONS
{'-'*30}
""" + "\n".join(
    f"[{t}] {fb['suggestions'][0]}"
    for t, fb in feedback['trait_feedback'].items()
    if fb['suggestions'] and fb['tier'] in ('satisfactory', 'poor')
)
        st.download_button(
            "📄 Download Report (.txt)",
            data=report_text,
            file_name=f"AES_Report_{time.strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    with col_dl2:
        feat_csv = pd.DataFrame([{
            "metric": k,
            "value": round(v, 4) if isinstance(v, float) else v
        } for k, v in linguistic_feats.items() if not isinstance(v, list)]).to_csv(index=False)
        st.download_button(
            "📊 Download Features (.csv)",
            data=feat_csv,
            file_name=f"AES_Features_{time.strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )


# ─── EMPTY STATE ─────────────────────────────────────────────────────────────

else:
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:48px 20px;color:#3a5a45;">
        <div style="font-size:4rem;margin-bottom:16px;">🎓</div>
        <h3 style="color:#0d5c30;font-weight:800;">Ready to Grade Your Essay</h3>
        <p style="font-size:1rem;max-width:580px;margin:0 auto;line-height:1.7;">
            Paste your essay in the text area above and click <strong>Evaluate Essay</strong>.<br>
            The system will analyse grammar, vocabulary, organisation, semantic coherence, 
            and provide detailed AI-generated feedback in seconds.
        </p>
        <br>
        <div style="display:flex;justify-content:center;gap:30px;flex-wrap:wrap;margin-top:12px;">
            <span style="background:#d4edda;padding:8px 20px;border-radius:20px;font-weight:600;color:#155724;">✅ 6 Trait Analysis</span>
            <span style="background:#d1ecf1;padding:8px 20px;border-radius:20px;font-weight:600;color:#0c5460;">🧠 DeBERTa-v3 Semantic AI</span>
            <span style="background:#fff3cd;padding:8px 20px;border-radius:20px;font-weight:600;color:#856404;">💬 XAI Feedback</span>
            <span style="background:#f3e5f5;padding:8px 20px;border-radius:20px;font-weight:600;color:#6a1b9a;">📊 ASAP Scoring</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:0.8rem;color:#5a8a6a;padding:10px 0 4px;">
    <strong>HFC-AES</strong> · Hybrid Feature-based Cross-prompt Automated Essay Scoring System<br>
    Kwara State University, Malete · Faculty of ICT, Dept. of Computer Science · 2024/2025
</div>
""", unsafe_allow_html=True)


# ─── EVALUATION TAB ──────────────────────────────────────────────────────────

with tab_eval:
    st.markdown("### 📊 System Evaluation Metrics")
    st.markdown(
        "These metrics are computed from every essay graded in this session. "
        "Grade at least **2 essays** first, then return here to see results. "
        "The four metrics from **§2.1.4** of the project report are applied using "
        "two internal raters derived from the engine itself:"
    )
    col_ra, col_rb = st.columns(2)
    with col_ra:
        st.info("**Rater A — Holistic Score**\nThe weighted neural-hybrid final score produced by the full pipeline.")
    with col_rb:
        st.info("**Rater B — Trait-Average Score**\nUnweighted mean of all 6 trait scores: Grammar, Vocabulary, Fluency, Organization, Content, Readability.")

    st.markdown("---")

    # ── live essay count ──────────────────────────────────────────────────────
    essays_so_far = EvaluationMetrics.get_essays(st.session_state)
    n_so_far      = len(essays_so_far)

    count_col, clear_col = st.columns([3, 1])
    with count_col:
        if n_so_far == 0:
            st.warning("⚠️ No essays graded yet this session. Submit essays in the tabs above, then return here.")
        elif n_so_far == 1:
            st.warning("⚠️ Only 1 essay graded. Grade at least one more to run evaluation.")
        else:
            st.success(f"✅ **{n_so_far} essay{'s' if n_so_far > 1 else ''} graded** this session — ready to evaluate.")
    with clear_col:
        if st.button("🗑️ Clear Session", use_container_width=True,
                     help="Remove all recorded essays and start fresh"):
            EvaluationMetrics.clear(st.session_state)
            st.rerun()

    # ── metric reference ──────────────────────────────────────────────────────
    with st.expander("📖 Metric Definitions (§2.1.4)"):
        st.markdown("""
| Metric | Formula | What it measures here |
|--------|---------|----------------------|
| **QWK** | `1 − Σ w·O / Σ w·E` | Agreement between holistic and trait-average scores. Penalises large gaps quadratically. |
| **MAE** | `(1/n) Σ |A - B|` | Average difference (in percentage points) between holistic and trait-average raters. |
| **RMSE** | `√[(1/n) Σ (A−B)²]` | Same as MAE but large discrepancies are penalised more heavily. |
| **Pearson r** | `Σ(A−Ā)(B−B̄) / √[Σ(A−Ā)²·Σ(B−B̄)²]` | Whether the two raters rank essays in the same order. High r = consistent ranking. |
""")

    # ── run evaluation ────────────────────────────────────────────────────────
    if n_so_far >= 2:
        if st.button("▶ Compute Evaluation Metrics", type="primary",
                     use_container_width=True, key="run_eval_btn"):
            with st.spinner("Computing metrics…"):
                try:
                    report = EvaluationMetrics.evaluate_session(st.session_state)
                    st.session_state["eval_report_cache"] = report.to_dict()
                except Exception as exc:
                    st.error(f"❌ Evaluation failed: {exc}")
                    import traceback
                    st.code(traceback.format_exc())
                    report = None

        # ── render cached report if available ────────────────────────────────
        if "eval_report_cache" in st.session_state and st.session_state["eval_report_cache"]:
            rep = st.session_state["eval_report_cache"]

            qwk = rep["QWK"]
            mae = rep["MAE_pp"]
            rmse = rep["RMSE_pp"]
            pr   = rep["Pearson_r"]
            interp = rep["qwk_interpretation"]
            n    = rep["n_essays"]

            # ── colour helper ─────────────────────────────────────────────────
            def _qwk_color(q):
                if q >= 0.80: return "#0d5c30"
                if q >= 0.70: return "#1a7a4a"
                if q >= 0.60: return "#c8973a"
                if q >= 0.40: return "#d97706"
                return "#b91c1c"

            def _card(label, value, sub="", color="#0d5c30", fmt=".4f"):
                return f"""<div style="background:#fff;border:1px solid #b8d4c0;border-radius:12px;
                    padding:18px 12px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.05);">
                    <div style="font-size:2rem;font-weight:900;color:{color};line-height:1;">{value:{fmt}}</div>
                    <div style="font-size:.78rem;font-weight:700;color:#3a5a45;text-transform:uppercase;
                        letter-spacing:.8px;margin-top:6px;">{label}</div>
                    {"<div style='font-size:.7rem;color:#777;margin-top:4px;'>" + sub + "</div>" if sub else ""}
                </div>"""

            st.markdown(" ")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(_card("QWK", qwk, interp, _qwk_color(qwk)), unsafe_allow_html=True)
            with c2:
                st.markdown(_card("MAE", mae, "percentage points", fmt=".2f"), unsafe_allow_html=True)
            with c3:
                st.markdown(_card("RMSE", rmse, "percentage points", fmt=".2f"), unsafe_allow_html=True)
            with c4:
                st.markdown(_card("Pearson r", pr, "ranking agreement"), unsafe_allow_html=True)

            # ── interpretation banner ─────────────────────────────────────────
            st.markdown(" ")
            if qwk >= 0.70:
                box_bg, icon = "#d4edda", "✅"
            elif qwk >= 0.60:
                box_bg, icon = "#fff3cd", "⚠️"
            else:
                box_bg, icon = "#f8d7da", "❌"

            ci = rep.get("consistency_index", 0)
            mc = rep.get("mean_confidence", 0)

            st.markdown(
                f"""<div style="background:{box_bg};border-radius:10px;padding:14px 18px;margin-bottom:16px;">
                <strong>{icon} QWK Interpretation:</strong> {interp}
                &nbsp;|&nbsp;<strong>Essays:</strong> {n}
                &nbsp;|&nbsp;<strong>Consistency Index:</strong> {ci:.1f}%
                &nbsp;|&nbsp;<strong>Mean Confidence:</strong> {int(mc*100)}%
                &nbsp;|&nbsp;<strong>Mean Score:</strong> {rep.get('mean_holistic_pct',0):.1f}%
                </div>""",
                unsafe_allow_html=True
            )

            # ── two-column detail section ─────────────────────────────────────
            left, right = st.columns(2)

            # Grade distribution donut
            with left:
                st.markdown("##### Grade Distribution")
                grade_dist = rep.get("grade_distribution", {})
                if grade_dist:
                    order  = ["A", "B", "C", "D", "F"]
                    grades = [g for g in order if g in grade_dist]
                    counts = [grade_dist[g] for g in grades]
                    colors = {
                        "A": "#1a7a4a", "B": "#2ea868",
                        "C": "#c8973a", "D": "#d97706", "F": "#b91c1c"
                    }
                    fig_donut = go.Figure(go.Pie(
                        labels=grades, values=counts,
                        hole=0.55,
                        marker_colors=[colors.get(g, "#888") for g in grades],
                        textinfo="label+percent",
                        textfont_size=13,
                    ))
                    fig_donut.update_layout(
                        height=260, margin=dict(t=10, b=10, l=10, r=10),
                        showlegend=False, paper_bgcolor="white"
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)

            # Per-trait bar chart
            with right:
                st.markdown("##### Mean Score per Trait (%)")
                trait_means = rep.get("trait_means", {})
                trait_stds  = rep.get("trait_stds",  {})
                if trait_means:
                    tnames = list(trait_means.keys())
                    tmeans = [trait_means[t] for t in tnames]
                    tstds  = [trait_stds.get(t, 0) for t in tnames]
                    short  = [t.split()[0] for t in tnames]   # first word only
                    fig_bar = go.Figure(go.Bar(
                        x=short, y=tmeans,
                        error_y=dict(type="data", array=tstds, visible=True),
                        marker_color="#1a7a4a",
                        text=[f"{v:.1f}" for v in tmeans],
                        textposition="outside",
                    ))
                    fig_bar.update_layout(
                        height=260, yaxis_range=[0, 110],
                        yaxis_title="Mean %",
                        plot_bgcolor="white", paper_bgcolor="white",
                        margin=dict(t=10, b=10, l=10, r=10),
                        font=dict(size=11),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

            # ── holistic vs trait-avg scatter ─────────────────────────────────
            with st.expander("📈 Holistic vs Trait-Average Scatter"):
                essays_data = EvaluationMetrics.get_essays(st.session_state)
                hol  = [e.holistic_pct  for e in essays_data]
                tavg = [e.trait_avg_pct for e in essays_data]
                gl   = [e.grade_letter  for e in essays_data]
                idxs = [f"Essay {e.essay_index}" for e in essays_data]
                grade_colors_map = {
                    "A": "#1a7a4a", "B": "#2ea868",
                    "C": "#c8973a", "D": "#d97706", "F": "#b91c1c"
                }
                pt_colors = [grade_colors_map.get(g, "#888") for g in gl]

                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=tavg, y=hol,
                    mode="markers+text",
                    text=idxs, textposition="top center",
                    textfont=dict(size=9),
                    marker=dict(color=pt_colors, size=10, opacity=0.85,
                                line=dict(color="#fff", width=1)),
                    name="Essays",
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Trait avg: %{x:.1f}%<br>"
                        "Holistic: %{y:.1f}%<extra></extra>"
                    )
                ))
                # Perfect-agreement line
                lo_lim = max(0,   min(tavg + hol) - 5)
                hi_lim = min(100, max(tavg + hol) + 5)
                fig_sc.add_trace(go.Scatter(
                    x=[lo_lim, hi_lim], y=[lo_lim, hi_lim],
                    mode="lines",
                    line=dict(color="#c8973a", dash="dash", width=1.5),
                    name="Perfect Agreement",
                ))
                fig_sc.update_layout(
                    height=380,
                    xaxis_title="Trait-Average Score (%)",
                    yaxis_title="Holistic Score (%)",
                    plot_bgcolor="white", paper_bgcolor="white",
                    margin=dict(t=20, b=40, l=50, r=20),
                    legend=dict(orientation="h", y=-0.15),
                )
                st.plotly_chart(fig_sc, use_container_width=True)

            # ── per-essay table ───────────────────────────────────────────────
            with st.expander("📋 Per-Essay Detail Table"):
                essays_data = EvaluationMetrics.get_essays(st.session_state)
                rows = []
                for e in essays_data:
                    rows.append({
                        "Essay #":        e.essay_index,
                        "Time":           e.timestamp,
                        "Prompt":         e.prompt_id,
                        "Holistic (%)":   round(e.holistic_pct,  1),
                        "Trait Avg (%)":  round(e.trait_avg_pct, 1),
                        "Diff (pp)":      round(abs(e.holistic_pct - e.trait_avg_pct), 1),
                        "Grade":          e.grade_letter,
                        "ASAP Score":     f"{e.final_score}/{int(e.score_max)}",
                        "Confidence":     f"{int(e.confidence*100)}%",
                        "Words":          e.word_count,
                    })
                df_essays = pd.DataFrame(rows)
                st.dataframe(df_essays, use_container_width=True, hide_index=True)

            # ── download buttons ──────────────────────────────────────────────
            dl1, dl2 = st.columns(2)
            import json as _json
            with dl1:
                st.download_button(
                    "⬇️ Download Report (JSON)",
                    data=_json.dumps(rep, indent=2),
                    file_name=f"hfc_aes_eval_{time.strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with dl2:
                txt = "\n".join([
                    "HFC-AES SYSTEM EVALUATION REPORT",
                    f"Essays graded this session : {n}",
                    f"Evaluation time            : {rep.get('elapsed_s',0):.4f}s",
                    "─" * 44,
                    f"QWK  (primary)   : {qwk:.6f}  ← {interp}",
                    f"MAE  (pp)        : {mae:.4f}",
                    f"RMSE (pp)        : {rmse:.4f}",
                    f"Pearson r        : {pr:.6f}",
                    "─" * 44,
                    f"Consistency Index: {ci:.1f}%  (essays within ±10 pp)",
                    f"Mean Holistic    : {rep.get('mean_holistic_pct',0):.2f}%",
                    f"Std  Holistic    : {rep.get('std_holistic_pct',0):.2f}%",
                    f"Mean Confidence  : {int(mc*100)}%",
                    "─" * 44,
                    "Trait Means:",
                ] + [f"  {t}: {v:.2f}%" for t, v in rep.get("trait_means", {}).items()] + [
                    "─" * 44,
                    "Grade Distribution:",
                ] + [f"  {g}: {c}" for g, c in sorted(rep.get("grade_distribution", {}).items())]
                )
                st.download_button(
                    "⬇️ Download Report (TXT)",
                    data=txt,
                    file_name=f"hfc_aes_eval_{time.strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

