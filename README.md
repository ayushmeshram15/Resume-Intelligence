# Resume Intelligence & Surgical LaTeX Auto-Editor Pipeline

An end-to-end **NLP Applicant Tracking System (ATS)**, **Hybrid Semantic Candidate Ranking**, **D3 Skill Overlap Heatmap**, and **Style-Locked LaTeX Resume Optimizer** pipeline built with `spaCy`, `sentence-transformers`, `Gemini API`, and `Streamlit` / `React`.

**Developer:** Ayush Harshwardhan Meshram

---

## Key Features

1. **Three-Page Interactive Architecture:**
   - **Page 1 — Project Intro:** Architecture overview, hybrid scoring formula, LaTeX style-lock rules, and GitHub documentation.
   - **Page 2 — ATS Dashboard:** Dynamic in-memory file uploads (`io.BytesIO`), real-time **Parsing Error Diagnostics & 1-Click Auto-Fix Solutions**, ranked candidates leaderboard (`CSV` export), and an interactive **D3 Skill Overlap & Rarity Heatmap** across the applicant pool.
   - **Page 3 — Surgical LaTeX Resume Optimizer:** Actionable **Gap Analysis Tips** (3–5 bullet points) and a **Style-Locked LaTeX (`.tex`) Auto-Editor** that injects missing Job Description keywords into existing `\item` and Skills lines while preserving **100% of the original `\documentclass`, font packages, margins, and custom macros**.

2. **Robust In-Memory Parsing & Error Removal Solutions:**
   - Processes uploaded `PDF`, `DOCX`, and `LaTeX (.tex)` files strictly in memory without hardcoded file paths.
   - Strips non-standard PDF unicode ligatures (`ﬁ`, `ﬂ`, private-use icon fonts `\uE000–\uF8FF`), repairs hyphenated line breaks (`\n`), and normalizes all skill matching to lowercase.
   - Extracts **Name**, **Email**, **Phone**, **Education**, and **Technical Skills** via `spaCy` (`en_core_web_sm`) with deterministic **Regex fallbacks** and 1-click **Auto-Fix Parsing Error** repair.

3. **D3 Skill Overlap & Applicant Pool Rarity Heatmap:**
   - Visualizes candidate-by-skill coverage across all uploaded resumes.
   - Categorizes skills into **Common Pool Skills ($\ge 60\%$)**, **Moderate Coverage ($30\text{--}59\%$)**, **Unique / Rare Differentiators ($< 30\%$)**, and **Uncovered JD Gaps ($0\%$)** with interactive sorting and filtering.

---

## Repository Structure

```text
├── requirements.txt                  # Python dependencies (Streamlit, spaCy, Sentence-Transformers, PyPDF2, python-docx, google-generativeai)
├── utils.py                          # In-memory io.BytesIO text extraction for PDF, DOCX, TXT, and .tex uploads
├── nlp_engine.py                     # Unicode/newline sanitization, spaCy NER + Regex fallbacks, and 60/40 hybrid scoring
├── optimizer.py                      # Gap Analysis Tips generator and Surgical LaTeX (.tex) Gemini Auto-Editor
├── app.py                            # 3-Page Streamlit application (Project Intro, ATS Dashboard, Resume Optimizer)
├── server.ts                         # Express + Gemini API backend for surgical LaTeX optimization
├── src/
│   ├── App.tsx                       # 3-Page interactive web workbench with 1-Click Parsing Error Auto-Fix
│   ├── components/
│   │   └── SkillOverlapHeatmap.tsx   # Interactive D3.js Skill Overlap & Rarity Matrix visualization
│   └── nlp/
│       └── engine.ts                 # Browser NLP parser, kerning-aware PDF extractor, and LaTeX surgical editor
└── README.md                         # Project documentation and GitHub setup guide
```

---

## Quick Start (Python & Streamlit)

### 1. Clone the Repository & Create a Virtual Environment

```bash
git clone <your-github-repo-url>
cd resume-intelligence-pipeline

python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell
```

### 2. Install Dependencies & spaCy Model

```bash
pip install --upgrade pip
pip install -r requirements.txt

# Download the spaCy English NER model
python -m spacy download en_core_web_sm
```

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```

---

## Quick Start (Web Workbench — Vite + Express + D3.js)

```bash
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

---

## Hybrid Candidate Scoring Formula ($0\text{--}100$ Scale)

Each candidate is evaluated using a weighted combination of **Rule-Based Keyword Matching ($60\%$)** and **Dense Semantic Embedding Similarity ($40\%$):**

$$\text{Final Score} = 0.60 \times S_{\text{rule}} + 0.40 \times S_{\text{semantic}}$$

- **Rule-Based Score ($S_{\text{rule}}$):** Exact lowercase-normalized overlap between required Job Description skills ($S_{\text{JD}}$) and extracted candidate skills ($S_{\text{Resume}}$):
  $$S_{\text{rule}} = 100 \times \frac{|S_{\text{JD}} \cap S_{\text{Resume}}|}{|S_{\text{JD}}|}$$
- **Semantic Similarity Score ($S_{\text{semantic}}$):** Dense vector cosine similarity between sanitized JD text and Resume text using `sentence-transformers/all-MiniLM-L6-v2` scaled to $[0, 100]$.

---

## Solutions to Remove Resume Parsing Errors

If an uploaded resume triggers a parsing warning or error on **Page 2 (ATS Dashboard)**, you can click **"Auto-Fix & Remove All Parsing Errors"** in the UI or apply the permanent fixes below:

| Parsing Error / Diagnostic | Root Cause | Permanent Solution |
| :--- | :--- | :--- |
| **PDF Unicode / Font Ligature Warning** | `pdflatex` ligatures (`fi`, `fl`) or icon fonts (`fontawesome`) lack a Unicode mapping table. | Add `\input{glyphtounicode}` and `\pdfgentounicode=1` immediately after `\documentclass` in your `.tex` preamble. |
| **Missing Email Address (`Not Provided`)** | Email is embedded inside an image or hidden in a custom macro with non-text display labels. | Write email as plain text (`name@domain.com`) or `\href{mailto:name@domain.com}{name@domain.com}`. |
| **Missing Phone Number (`Not Provided`)** | Phone number has fewer than 10 digits or uses non-standard separators. | Format phone numbers in standard international or 10-digit format (`+91 98765 43210` or `555-234-5678`). |
| **Hyphenated Line-Wrap Word Splits** | Multi-column PDF layout splits technical terms across lines (`Trans-\nformers`). | Add `\hyphenpenalty=10000` and `\exhyphenpenalty=10000` in your LaTeX preamble to disable word hyphenation. |

---

## Pushing to GitHub

```bash
git init
git add README.md requirements.txt utils.py nlp_engine.py optimizer.py app.py package.json src/
git commit -m "feat: add Resume Intelligence pipeline, D3 skill heatmap, parser auto-fix, and LaTeX optimizer"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```
