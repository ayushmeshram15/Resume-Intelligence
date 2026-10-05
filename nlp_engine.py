"""
nlp_engine.py
-------------
Core NLP & Semantic Matching Engine with Parser Robustness Fixes:
1. Text Sanitization (`sanitize_resume_text`): Removes stray newline characters (`\\n`),
   strips weird PDF unicode symbols/ligatures, and collapses broken token lines.
2. Strict Lowercase Normalization: Ensures all JD and Resume skill comparisons
   use normalized lowercase strings (`str.lower().strip()`).
3. `spaCy` (en_core_web_sm) NER + Deterministic Regex Backups for Email & Phone.
4. Hybrid Scoring (0-100 scale): 60% Rule-Based Keyword Overlap + 40% Semantic Cosine Similarity.
"""

import re
import unicodedata
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Canonical technical and domain skills taxonomy (keys strictly lowercase)
SKILL_TAXONOMY: Dict[str, str] = {
    "python": "Python",
    "java": "Java",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "c++": "C++",
    "c#": "C#",
    "golang": "Go",
    "rust": "Rust",
    "sql": "SQL",
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
    "mongodb": "MongoDB",
    "redis": "Redis",
    "snowflake": "Snowflake",
    "bigquery": "BigQuery",
    "spark": "Apache Spark",
    "pyspark": "PySpark",
    "kafka": "Apache Kafka",
    "airflow": "Apache Airflow",
    "dbt": "dbt",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "scikit-learn": "Scikit-Learn",
    "sklearn": "Scikit-Learn",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "keras": "Keras",
    "spacy": "spaCy",
    "nltk": "NLTK",
    "huggingface": "Hugging Face",
    "hugging face": "Hugging Face",
    "transformers": "Transformers",
    "sentence-transformers": "Sentence-Transformers",
    "llm": "LLMs",
    "llms": "LLMs",
    "rag": "RAG",
    "langchain": "LangChain",
    "nlp": "NLP",
    "computer vision": "Computer Vision",
    "deep learning": "Deep Learning",
    "machine learning": "Machine Learning",
    "mlops": "MLOps",
    "mlflow": "MLflow",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "aws": "AWS",
    "gcp": "GCP",
    "azure": "Azure",
    "terraform": "Terraform",
    "ci/cd": "CI/CD",
    "github actions": "GitHub Actions",
    "jenkins": "Jenkins",
    "linux": "Linux",
    "git": "Git",
    "fastapi": "FastAPI",
    "flask": "Flask",
    "django": "Django",
    "react": "React",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "graphql": "GraphQL",
    "rest api": "REST APIs",
    "rest apis": "REST APIs",
    "microservices": "Microservices",
    "streamlit": "Streamlit",
    "tableau": "Tableau",
    "power bi": "Power BI",
    "statistics": "Statistics",
    "a/b testing": "A/B Testing",
}

EMAIL_REGEX = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
PHONE_REGEX = re.compile(
    r"(?:(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4})"
)
DEGREE_REGEX = re.compile(
    r"\b(?:"
    r"B\.?S\.?|M\.?S\.?|Ph\.?D\.?|B\.?A\.?|M\.?A\.?|B\.?Tech\.?|M\.?Tech\.?|M\.?B\.?A\.?|B\.?E\.?|M\.?E\.?|"
    r"Bachelor(?:'s)?(?: of [A-Za-z ]+)?|"
    r"Master(?:'s)?(?: of [A-Za-z ]+)?|"
    r"Doctorate(?: in [A-Za-z ]+)?"
    r")\b",
    re.IGNORECASE,
)
EDUCATION_ORG_KEYWORDS = (
    "university",
    "institute",
    "college",
    "school",
    "academy",
    "polytechnic",
)

# PDF Unicode Ligature & Symbol Replacement Map
PDF_UNICODE_LIGATURES: Dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2022": " ",
    "\u25cf": " ",
    "\u25aa": " ",
    "\uf0b7": " ",
    "\uf0a7": " ",
    "\u200b": "",
    "\ufeff": "",
    "\xa0": " ",
    "\x00": " ",
}

_SPACY_NLP = None
_SENTENCE_MODEL: Optional[SentenceTransformer] = None


def get_spacy_model(model_name: str = "en_core_web_sm"):
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            _SPACY_NLP = spacy.load(model_name)
        except OSError:
            _SPACY_NLP = spacy.blank("en")
    return _SPACY_NLP


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        _SENTENCE_MODEL = SentenceTransformer(model_name)
    return _SENTENCE_MODEL


def sanitize_resume_text(raw_text: str) -> Tuple[str, List[str]]:
    """
    Parser Robustness Fix:
    1. Replaces PDF unicode ligatures (e.g., ﬁ -> fi, ﬂ -> fl) and strips private-use
       PDF bullet symbols (`\\uf0b7`, `\\u2022`, zero-width spaces).
    2. Normalizes unicode via NFKD and strips unprintable control characters.
    3. Repairs broken PDF newline wraps (`\\n`) so multi-word skills and sentences
       are not split across lines, while keeping clean single-space separation.
    Returns `(cleaned_text, parsing_notices)`.
    """
    if not raw_text:
        return "", ["Empty document text received."]

    notices: List[str] = []
    text = raw_text

    # 1. Replace known PDF ligatures and bullet artifacts
    ligature_hits = sum(1 for k in PDF_UNICODE_LIGATURES if k in text)
    for bad_char, replacement in PDF_UNICODE_LIGATURES.items():
        text = text.replace(bad_char, replacement)

    # 2. Strip private-use area unicode symbols (U+E000..U+F8FF) common in PDF fonts
    pua_cleaned = re.sub(r"[\ue000-\uf8ff]", " ", text)
    if pua_cleaned != text or ligature_hits > 0:
        notices.append("Stripped non-standard PDF unicode symbols/ligatures.")
    text = pua_cleaned

    # 3. Normalize Unicode compatibility characters
    text = unicodedata.normalize("NFKC", text)

    # 4. Remove non-printable control characters (except newline and tab)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\u024F]", " ", text)

    # 5. Fix hyphenated line-breaks from PDF column wrapping (e.g., "Trans-\nformers" -> "Transformers")
    text = re.sub(r"([a-zA-Z])-\s*\n\s*([a-zA-Z])", r"\1\2", text)

    # 6. Normalize carriage returns and collapse excessive newline characters (`\n`)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    non_empty_lines = [line for line in lines if line]

    cleaned_multiline = "\n".join(non_empty_lines)
    return cleaned_multiline.strip(), notices


def normalize_flat_text(text: str) -> str:
    """
    Flattens all newline characters (`\\n`) into single spaces and normalizes
    to lowercase for reliable keyword and skill matching across line breaks.
    """
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip().lower()


def extract_name(doc, sanitized_text: str, filename: str = "") -> str:
    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent.start_char < 400:
            candidate = re.sub(r"\s+", " ", ent.text).strip()
            if 2 <= len(candidate.split()) <= 4 and not any(ch.isdigit() for ch in candidate):
                return candidate

    blacklist = {
        "resume", "curriculum", "vitae", "cv", "summary", "profile",
        "engineer", "developer", "scientist", "analyst", "phone", "email", "contact",
    }
    lines = [line.strip() for line in sanitized_text.splitlines() if line.strip()]
    name_regex = re.compile(r"^[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z.'-]+){1,3}$")

    for line in lines[:6]:
        cleaned = re.sub(r"^(?:Name|Candidate)\s*[:\-]\s*", "", line, flags=re.IGNORECASE)
        cleaned = re.sub(r"[|,•].*$", "", cleaned).strip()
        words_lower = {w.lower() for w in cleaned.split()}
        if name_regex.match(cleaned) and not (words_lower & blacklist):
            return cleaned

    if filename:
        stem = re.sub(r"\.(pdf|docx|txt)$", "", filename, flags=re.IGNORECASE)
        stem = re.sub(r"[_-]+", " ", stem).strip()
        if stem:
            return stem.title()

    return "Unknown Candidate"


def extract_email(doc, sanitized_text: str) -> Tuple[str, Optional[str]]:
    """
    Uses spaCy `like_email` first; falls back to explicit Regex on flattened text.
    """
    for token in doc:
        if token.like_email:
            return token.text.strip().strip(".,;<>()[]"), None

    flat_text = sanitized_text.replace("\n", " ")
    match = EMAIL_REGEX.search(flat_text)
    if match:
        return match.group(0).strip().lower(), "Email extracted via Regex fallback (missed by spaCy NER)."

    return "Not Provided", "Parsing Error: No valid Email address found in document."


def extract_phone(doc, sanitized_text: str) -> Tuple[str, Optional[str]]:
    """
    Uses spaCy `CARDINAL` entity check first; falls back to explicit Regex.
    """
    for ent in doc.ents:
        if ent.label_ in ("CARDINAL", "QUANTITY") and ent.start_char < 500:
            digits = re.sub(r"\D", "", ent.text)
            if 10 <= len(digits) <= 15 and PHONE_REGEX.search(ent.text):
                return ent.text.strip(), None

    flat_text = sanitized_text.replace("\n", " ")
    for match in PHONE_REGEX.finditer(flat_text):
        candidate = match.group(0).strip()
        digits = re.sub(r"\D", "", candidate)
        if 10 <= len(digits) <= 15:
            return candidate, "Phone extracted via Regex fallback (missed by spaCy NER)."

    return "Not Provided", "Parsing Error: No valid Phone number found in document."


def extract_education(doc, sanitized_text: str) -> List[str]:
    edu_items: List[str] = []

    for ent in doc.ents:
        if ent.label_ == "ORG":
            if any(kw in ent.text.lower() for kw in EDUCATION_ORG_KEYWORDS):
                cleaned = re.sub(r"\s+", " ", ent.text).strip()
                if cleaned not in edu_items:
                    edu_items.append(cleaned)

    for line in sanitized_text.splitlines():
        clean_line = line.strip()
        if not clean_line or len(clean_line) > 150:
            continue
        if DEGREE_REGEX.search(clean_line) or any(
            kw in clean_line.lower() for kw in EDUCATION_ORG_KEYWORDS
        ):
            if clean_line not in edu_items:
                edu_items.append(clean_line)

    return edu_items[:4]


def extract_skills(doc, sanitized_text: str, extra_skills: Optional[List[str]] = None) -> List[str]:
    """
    Parser Robustness Fix:
    Flattens newline characters (`\\n`) and strictly normalizes text and skill keys
    to lowercase before matching so line-wrapped skills are never missed.
    """
    found_canonical: Dict[str, str] = {}
    flat_lower_text = normalize_flat_text(sanitized_text)

    # Strictly lowercase-normalized lookup table
    lookup: Dict[str, str] = {k.lower().strip(): v for k, v in SKILL_TAXONOMY.items()}
    if extra_skills:
        for s in extra_skills:
            norm_key = s.lower().strip()
            if norm_key and norm_key not in lookup:
                lookup[norm_key] = s.strip()

    # 1. spaCy token & lemma matching (normalized to lowercase)
    for token in doc:
        lemma_norm = (token.lemma_ or token.text).lower().strip()
        if lemma_norm in lookup:
            canonical = lookup[lemma_norm]
            found_canonical[canonical.lower()] = canonical

    # 2. Boundary-safe regex matching on newline-flattened lowercase text
    for pattern_lower, canonical in lookup.items():
        escaped = re.escape(pattern_lower)
        if re.search(rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])", flat_lower_text):
            found_canonical[canonical.lower()] = canonical

    return sorted(found_canonical.values())


def extract_jd_skills(jd_text: str) -> List[str]:
    """
    Sanitizes JD text and extracts required skills using strict lowercase normalization.
    """
    sanitized_jd, _ = sanitize_resume_text(jd_text)
    nlp = get_spacy_model()
    doc = nlp(sanitized_jd)
    skills_map: Dict[str, str] = {
        s.lower().strip(): s for s in extract_skills(doc, sanitized_jd)
    }

    header_matches = re.findall(
        r"(?:required skills|technical skills|skills|technologies)\s*[:\-]\s*([^\n]+)",
        sanitized_jd,
        flags=re.IGNORECASE,
    )
    for line in header_matches:
        for item in re.split(r"[,;/•|]", line):
            norm_item = item.lower().strip()
            if norm_item in SKILL_TAXONOMY:
                canonical = SKILL_TAXONOMY[norm_item]
                skills_map[canonical.lower()] = canonical

    return sorted(skills_map.values())


def parse_resume(raw_text: str, filename: str = "", jd_skills: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Sanitizes raw resume text, extracts entities via spaCy + Regex backups,
    and collects any parsing warnings/errors for display in the ATS Dashboard.
    """
    sanitized_text, parsing_warnings = sanitize_resume_text(raw_text)
    nlp = get_spacy_model()
    doc = nlp(sanitized_text)

    name = extract_name(doc, sanitized_text, filename=filename)
    email, email_warn = extract_email(doc, sanitized_text)
    phone, phone_warn = extract_phone(doc, sanitized_text)
    education = extract_education(doc, sanitized_text)
    skills = extract_skills(doc, sanitized_text, extra_skills=jd_skills)

    if email_warn:
        parsing_warnings.append(email_warn)
    if phone_warn:
        parsing_warnings.append(phone_warn)
    if not skills:
        parsing_warnings.append("Parsing Warning: 0 technical skills matched from taxonomy.")

    return {
        "filename": filename,
        "name": name,
        "email": email,
        "phone": phone,
        "education": education,
        "skills": skills,
        "raw_text": sanitized_text,
        "parsing_warnings": parsing_warnings,
    }


def score_candidate(
    parsed_resume: Dict[str, Any],
    jd_text: str,
    jd_skills: List[str],
    rule_weight: float = 0.60,
    semantic_weight: float = 0.40,
) -> Tuple[float, List[str], List[str], str]:
    """
    Computes the hybrid 0-100 score using strict lowercase skill normalization:
      - Rule-Based Score (60%): Keyword overlap between JD skills and Resume skills.
      - Semantic Score (40%): Cosine similarity between sanitized single-line JD and Resume text.
    """
    # Strict lowercase normalization for skill comparison
    resume_skills_lower: Set[str] = {
        s.lower().strip() for s in parsed_resume.get("skills", [])
    }

    found_skills: List[str] = []
    missing_skills: List[str] = []

    for req_skill in jd_skills:
        if req_skill.lower().strip() in resume_skills_lower:
            found_skills.append(req_skill)
        else:
            missing_skills.append(req_skill)

    # 1. Rule-Based Score (0-100)
    if jd_skills:
        rule_score = (len(found_skills) / len(jd_skills)) * 100.0
    else:
        rule_score = 100.0 if resume_skills_lower else 0.0

    # 2. Semantic Similarity Score (0-100) on newline-cleaned text
    clean_resume_flat = re.sub(r"\s+", " ", parsed_resume.get("raw_text", "").replace("\n", " ")).strip()
    clean_jd_flat = re.sub(r"\s+", " ", jd_text.replace("\n", " ")).strip()

    if clean_resume_flat and clean_jd_flat:
        model = get_embedding_model("all-MiniLM-L6-v2")
        embeddings = model.encode(
            [clean_jd_flat, clean_resume_flat],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        jd_vec = np.asarray(embeddings[0]).reshape(1, -1)
        res_vec = np.asarray(embeddings[1]).reshape(1, -1)
        cos_sim = float(cosine_similarity(jd_vec, res_vec)[0][0])
        semantic_score = max(0.0, min(1.0, cos_sim)) * 100.0
    else:
        semantic_score = 0.0

    # 3. Weighted Final Score (60% Rule-Based + 40% Semantic)
    final_score = round((rule_weight * rule_score) + (semantic_weight * semantic_score), 2)

    found_str = ", ".join(found_skills[:5]) if found_skills else "None"
    if len(found_skills) > 5:
        found_str += f" (+{len(found_skills) - 5} more)"

    missing_str = ", ".join(missing_skills[:5]) if missing_skills else "None"
    if len(missing_skills) > 5:
        missing_str += f" (+{len(missing_skills) - 5} more)"

    reason = (
        f"Rule: {rule_score:.1f}% | Semantic: {semantic_score:.1f}% — "
        f"Found: {found_str}; Missing: {missing_str}"
    )

    return final_score, found_skills, missing_skills, reason
