# Resume Intelligence Pipeline & Semantic Ranking App

An end-to-end NLP pipeline and interactive **Streamlit** application that ingests multi-format resumes (`PDF`, `DOCX`, `TXT`), extracts structured candidate entities and technical skills using **spaCy** (`en_core_web_sm`) with deterministic **Regex fallbacks**, and ranks candidates against a Job Description (JD) using a hybrid **Rule-Based + Sentence-Transformer Semantic Matching** scoring formula.

---

## Project Architecture

```text
├── requirements.txt   # Python dependencies (Streamlit, spaCy, Sentence-Transformers, pdfplumber, etc.)
├── utils.py           # Document text extraction for PDF (pdfplumber + PyPDF2 fallback) & DOCX
├── parser.py          # spaCy NER + Regex fallback parser for Name, Email, Phone, Education, and Skills
├── matcher.py         # Rule-based keyword scorer, Sentence-Transformer cosine similarity, and aggregator
├── app.py             # Streamlit web application with interactive weights, ranked DataFrame, & CSV/JSON export
└── README.md          # Setup, execution, and evaluation documentation
```

---

## 1. Environment Setup & Installation

### Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell
```

### Install Required Packages & spaCy Language Model

```bash
pip install --upgrade pip
pip install -r requirements.txt

# Download the spaCy English NER model
python -m spacy download en_core_web_sm
```

---

## 2. Running the Application

Launch the interactive Streamlit workbench:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 3. Scoring Methodology

Each candidate receives a **Final Score (0–100)** calculated as a weighted combination of two complementary signals:

$$\text{Final Score} = w_{\text{rule}} \cdot S_{\text{rule}} + w_{\text{semantic}} \cdot S_{\text{semantic}}$$

1. **Rule-Based Keyword Match ($S_{\text{rule}}$, default weight $60\%$):**
   - Extracts canonical technical skills from the Job Description ($S_{\text{JD}}$) and the Resume ($S_{\text{Resume}}$).
   - Computes exact and lemmatized overlap:
     $$S_{\text{rule}} = 100 \times \frac{\sum_{s \in S_{\text{JD}} \cap S_{\text{Resume}}} w(s)}{\sum_{s \in S_{\text{JD}}} w(s)}$$
2. **Semantic Embedding Similarity ($S_{\text{semantic}}$, default weight $40\%$):**
   - Encodes the full JD text and Resume text into L2-normalized 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2`.
   - Computes cosine similarity $\cos(\theta) = \mathbf{u} \cdot \mathbf{v}$ and scales to $[0, 100]$.

---

## 4. Evaluating Pipeline Metrics

To benchmark extraction accuracy and ranking quality on a labeled validation set:

1. **Entity & Skill Extraction Quality (Precision, Recall, $F_1$):**
   - Compare extracted `(Name, Email, Phone, Skills)` against ground-truth annotations:
     $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
2. **Ranking Quality (Precision@K, MRR, and NDCG@K):**
   - **Precision@K:** Fraction of top-$K$ ranked candidates rated as qualified by human recruiters.
   - **Normalized Discounted Cumulative Gain ($\text{NDCG@K}$):** Evaluates graded relevance scores ($r_i \in \{0, 1, 2, 3\}$) across the ranked leaderboard:
     $$\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{r_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$
