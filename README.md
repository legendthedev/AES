# HFC-AES: Automated Essay Grading System
**Kwara State University, Malete — Faculty of ICT, Dept. of Computer Science**

A Text-Based Automated Essay Grading System using the **Hybrid Feature-based Cross-prompt (HFC-AES)** methodology. Combines DeBERTa-v3 neural embeddings with handcrafted linguistic features for interpretable, rubric-aligned essay scoring.

---

## 🚀 Quick Setup

### 1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download spaCy language model
```bash
python -m spacy download en_core_web_sm
```

### 4. Run the application
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📦 Minimal Install (no GPU / no DeBERTa)

If you don't have PyTorch or want a lighter install:
```bash
pip install streamlit spacy nltk textstat pyspellchecker scikit-learn pandas numpy plotly python-docx PyPDF2
python -m spacy download en_core_web_sm
streamlit run app.py
```
The system will automatically fall back to **TF-IDF + LSA** for semantic features.

---

## 🏗️ System Architecture

```
essay_text
    │
    ▼
┌─────────────────────────────────┐
│     Data Preprocessing Module   │
│  (normalize → tokenize → spell) │
└──────────────┬──────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌─────────────┐  ┌──────────────────┐
│  Linguistic │  │  Neural Pathway  │
│   Pathway   │  │  (DeBERTa-v3 /  │
│ (Handcrafted│  │   TF-IDF+LSA)   │
│  Features)  │  └────────┬─────────┘
└──────┬──────┘           │
       └────────┬──────────┘
                ▼
    ┌───────────────────────┐
    │  Feature Fusion Layer  │
    │  (Z-score normalized)  │
    └───────────┬────────────┘
                ▼
    ┌───────────────────────┐
    │    Scoring Engine      │
    │  (6 trait rubric)      │
    └───────────┬────────────┘
                ▼
    ┌───────────────────────┐
    │  XAI Feedback Engine   │
    │  (HIF generation)      │
    └───────────────────────┘
                ▼
         Streamlit GUI
```

---

## 📊 Graded Traits

| Trait | Weight | Description |
|-------|--------|-------------|
| Grammar & Mechanics | 15% | Spelling, punctuation, capitalization |
| Vocabulary Sophistication | 20% | Lexical diversity, academic word usage |
| Sentence Fluency | 15% | Variety, complexity, avg length |
| Organization & Structure | 20% | Paragraphing, transitions, intro/conclusion |
| Content Development | 20% | Depth, coherence, length adequacy |
| Readability | 10% | Flesch-Kincaid, Gunning Fog |

---

## 📁 Project Structure

```
aes_system/
├── app.py                      # Main Streamlit application
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml             # Custom green theme
└── src/
    ├── __init__.py
    ├── preprocessor.py         # Text normalization & tokenization
    ├── linguistic_features.py  # Handcrafted feature extraction
    ├── neural_features.py      # DeBERTa-v3 / TF-IDF embeddings
    ├── scoring_engine.py       # Feature fusion & grade prediction
    └── feedback_engine.py      # XAI natural-language feedback
```

---

## 📚 Dataset

The system is calibrated for the **ASAP (Automated Student Assessment Prize)** dataset:
- **8 essay prompts** (Persuasive, Source-Dependent, Narrative)
- **~12,978 essays** across Grade 7–10
- Download: https://www.kaggle.com/datasets/lburleigh/asap-2-0

To train a custom scoring model on ASAP, add a `train.py` script that:
1. Loads the dataset
2. Runs all essays through the feature extraction pipeline
3. Fits a `sklearn` regressor (e.g., `SVR`, `GradientBoostingRegressor`) on the features
4. Saves the model with `joblib.dump`
5. Loads it in `scoring_engine.py`

---

## 📝 Evaluation Metrics

- **Quadratic Weighted Kappa (QWK)** — Primary metric
- **Mean Absolute Error (MAE)**
- **Root Mean Square Error (RMSE)**
- **Pearson Correlation Coefficient (r)**

---

*Faculty of Information and Communication Technology, KWASU, Malete.*
