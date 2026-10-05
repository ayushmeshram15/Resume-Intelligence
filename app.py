"""
app.py
------
Three-Page Streamlit Application for the Resume Intelligence Pipeline
developed by Ayush Harshwardhan Meshram.

Pages (`st.sidebar.radio`):
1. Page 1: Project Intro
2. Page 2: ATS Dashboard (In-memory processing, parsing error diagnostics, & ranking)
3. Page 3: Resume Optimizer (Gap Analysis Tips & Gemini AI Auto-Editor)
"""

import pandas as pd
import streamlit as st

from utils import extract_text_from_upload
from nlp_engine import extract_jd_skills, parse_resume, score_candidate
from optimizer import generate_gap_analysis_tips, rewrite_resume_with_gemini

st.set_page_config(
    page_title="Resume Intelligence & AI Optimizer",
    page_icon="📄",
    layout="wide",
)

# Initialize session state so processed candidates persist across Page 2 and Page 3
if "processed_candidates" not in st.session_state:
    st.session_state["processed_candidates"] = []
if "active_jd_text" not in st.session_state:
    st.session_state["active_jd_text"] = ""
if "parsing_errors_log" not in st.session_state:
    st.session_state["parsing_errors_log"] = []


def render_project_intro_page() -> None:
    """
    Page 1: Project Intro
    """
    st.title("Resume Intelligence & AI Auto-Editor Pipeline")
    st.subheader("End-to-End NLP Parsing, Semantic Ranking & Generative Resume Optimization")

    st.markdown(
        """
        Welcome to the **Resume Intelligence Pipeline**, an end-to-end Applicant Tracking,
        Semantic Ranking, and Generative AI Resume Optimization application built with
        `spaCy`, `sentence-transformers`, and the **Gemini API**.
        """
    )

    st.divider()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### Core Pipeline Capabilities")
        st.markdown(
            """
            1. **In-Memory Document Ingestion (`io.BytesIO`):**
               - Processes uploaded **PDF** (`PyPDF2`) and **DOCX** (`python-docx`) files directly in memory without hardcoded file paths.
            2. **Robust NLP Parsing (`spaCy` + Regex Backups):**
               - Sanitizes broken PDF newline characters (`\\n`), ligatures, and non-ASCII unicode symbols.
               - Extracts **Name**, **Education**, and **Skills** (with strict lowercase normalization), falling back to deterministic **Regex** rules for **Email** and **Phone**.
            3. **Hybrid Candidate Scoring (0–100 Scale):**
               - **Rule-Based Matching (60%):** Keyword overlap between JD skills and candidate skills.
               - **Semantic Matching (40%):** Dense vector cosine similarity via `all-MiniLM-L6-v2`.
            4. **Resume Optimizer & Auto-Editor (Gemini API):**
               - Generates actionable **Gap Analysis Tips** from missing skills and rewrites the candidate's summary and experience bullet points without fabricating experience.
            """
        )

    with col2:
        st.markdown("### Developer Attribution")
        st.info(
            """
            - **Project Title:** End-to-End Resume Intelligence & AI Optimizer Pipeline
            - **Developer:** Ayush Harshwardhan Meshram
            - **Tech Stack:** Python, Streamlit, spaCy (`en_core_web_sm`), Sentence-Transformers (`all-MiniLM-L6-v2`), Google Generative AI (`gemini-1.5-flash`), Pandas, PyPDF2, python-docx
            """
        )

        st.markdown("### Hybrid Scoring Formula")
        st.latex(
            r"\text{Final Score} = 0.60 \times S_{\text{rule}} + 0.40 \times S_{\text{semantic}}"
        )


def render_ats_dashboard_page() -> None:
    """
    Page 2: ATS Dashboard (Processing, Parsing Error Diagnostics & Ranking)
    """
    st.title("ATS Candidate Screening & Ranking Dashboard")
    st.markdown(
        "Paste a **Job Description** (or upload a JD file) and upload multiple **Candidate Resumes** (`PDF` or `DOCX`). "
        "Click **Process Candidates** to run entity extraction, inspect parsing errors, and view ranked candidates."
    )

    col_jd, col_resumes = st.columns(2, gap="large")

    with col_jd:
        st.subheader("1. Job Description Input")
        jd_text_input = st.text_area(
            "Paste the Job Description Text",
            value=st.session_state.get("active_jd_text", ""),
            height=200,
            placeholder=(
                "Paste Job Description requirements here (e.g., Python, PyTorch, spaCy, NLP, "
                "Transformers, Docker, Kubernetes, AWS, FastAPI, SQL)..."
            ),
        )
        jd_file_upload = st.file_uploader(
            "Or Upload a Job Description File (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            key="jd_uploader",
        )

    with col_resumes:
        st.subheader("2. Upload Candidate Resumes")
        uploaded_resumes = st.file_uploader(
            "Upload Multiple Resumes (PDF, DOCX)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="resume_uploader",
        )
        st.caption(
            "Files are processed in-memory via `io.BytesIO`. Any file or entity parsing errors will be displayed below."
        )

    process_btn = st.button("Process Candidates", type="primary", use_container_width=True)

    if process_btn:
        st.session_state["parsing_errors_log"] = []
        jd_text = jd_text_input.strip()

        if jd_file_upload is not None:
            try:
                jd_text = extract_text_from_upload(jd_file_upload)
                st.success(f"Loaded Job Description from `{jd_file_upload.name}`.")
            except Exception as exc:
                err_msg = f"JD Parsing Error (`{jd_file_upload.name}`): {exc}"
                st.session_state["parsing_errors_log"].append(err_msg)
                st.error(err_msg)
                return

        if not jd_text:
            st.error("Please paste a Job Description or upload a valid JD file before processing.")
            return

        if not uploaded_resumes:
            st.error("Please upload at least one candidate resume (PDF or DOCX).")
            return

        st.session_state["active_jd_text"] = jd_text
        jd_skills = extract_jd_skills(jd_text)

        if jd_skills:
            st.info(f"**Extracted JD Skills ({len(jd_skills)}):** {', '.join(jd_skills)}")
        else:
            st.warning(
                "No specific taxonomy skills matched in the JD; ranking will rely on semantic similarity."
            )

        processed_list = []

        with st.spinner("Sanitizing text, extracting entities, and computing embeddings..."):
            for resume_file in uploaded_resumes:
                try:
                    raw_resume_text = extract_text_from_upload(resume_file)
                    parsed = parse_resume(
                        raw_text=raw_resume_text,
                        filename=resume_file.name,
                        jd_skills=jd_skills,
                    )
                    final_score, found_skills, missing_skills, reason = score_candidate(
                        parsed_resume=parsed,
                        jd_text=jd_text,
                        jd_skills=jd_skills,
                        rule_weight=0.60,
                        semantic_weight=0.40,
                    )

                    # Log any entity/unicode parsing warnings for transparency
                    for warn in parsed.get("parsing_warnings", []):
                        st.session_state["parsing_errors_log"].append(
                            f"[{resume_file.name}] {warn}"
                        )

                    processed_list.append(
                        {
                            "Filename": resume_file.name,
                            "Name": parsed["name"],
                            "Email": parsed["email"],
                            "Phone": parsed["phone"],
                            "Final Score": final_score,
                            "Found Skills": ", ".join(found_skills) if found_skills else "None",
                            "Missing Skills": ", ".join(missing_skills) if missing_skills else "None",
                            "Reason": reason,
                            "found_skills_list": found_skills,
                            "missing_skills_list": missing_skills,
                            "raw_text": parsed["raw_text"],
                            "parsing_warnings": parsed.get("parsing_warnings", []),
                        }
                    )
                except Exception as exc:
                    err_entry = f"File Parsing Error — Skipping `{resume_file.name}`: {exc}"
                    st.session_state["parsing_errors_log"].append(err_entry)
                    st.error(err_entry)

        # Sort highest-to-lowest by Final Score and assign Rank
        processed_list.sort(key=lambda x: x["Final Score"], reverse=True)
        for idx, item in enumerate(processed_list, start=1):
            item["Rank"] = idx

        st.session_state["processed_candidates"] = processed_list

    # Display Parsing Errors & Diagnostics Log if any exist
    if st.session_state["parsing_errors_log"]:
        st.subheader("Parsing Errors & Sanitization Diagnostics")
        for log_msg in st.session_state["parsing_errors_log"]:
            if "Parsing Error" in log_msg or "Skipping" in log_msg:
                st.error(log_msg)
            else:
                st.warning(log_msg)

    # Display Ranked DataFrame if candidates have been processed
    candidates = st.session_state.get("processed_candidates", [])
    if candidates:
        ordered_columns = [
            "Rank",
            "Name",
            "Email",
            "Phone",
            "Final Score",
            "Found Skills",
            "Missing Skills",
            "Reason",
        ]
        df = pd.DataFrame(candidates)[ordered_columns]

        st.subheader("3. Ranked Candidates Leaderboard")
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Ranked Report (CSV)",
            data=csv_bytes,
            file_name="ranked_candidates_ats_report.csv",
            mime="text/csv",
        )


def render_resume_optimizer_page(api_key: str) -> None:
    """
    Page 3: Resume Optimizer & Auto-Editor
    """
    st.title("Resume Optimizer & AI Auto-Editor")
    st.markdown(
        "Select a processed candidate to view **Gap Analysis Tips** based on their missing skills "
        "and generate an **AI-Rewritten Optimized Resume** tailored to the Job Description."
    )

    # Clear info/warning banner instructing the user to enter their API key
    if not api_key.strip():
        st.warning(
            "⚠️ **API Key Required for AI Auto-Editor:** Please enter your **Gemini API Key** "
            "in the left sidebar before clicking **Generate AI-Optimized Resume**. "
            "(Rule-based Gap Analysis Tips are available immediately below.)"
        )
    else:
        st.info(
            "✅ **Gemini API Key Detected:** Select a candidate below and click "
            "**Generate AI-Optimized Resume** to rewrite their summary and experience bullet points."
        )

    candidates = st.session_state.get("processed_candidates", [])
    jd_text = st.session_state.get("active_jd_text", "")

    if not candidates:
        st.info(
            "ℹ️ No processed candidates found yet. Please go to **Page 2: ATS Dashboard**, "
            "upload a Job Description and candidate resumes, and click **Process Candidates** first."
        )
        return

    candidate_labels = [
        f"Rank #{c['Rank']} — {c['Name']} ({c['Final Score']:.2f} pts)"
        for c in candidates
    ]
    selected_idx = st.selectbox(
        "Select Uploaded Candidate to Optimize",
        options=range(len(candidates)),
        format_func=lambda i: candidate_labels[i],
    )

    selected_candidate = candidates[selected_idx]
    missing_skills = selected_candidate.get("missing_skills_list", [])
    found_skills = selected_candidate.get("found_skills_list", [])
    original_text = selected_candidate.get("raw_text", "")

    # Side-by-Side Comparison (`st.columns(2)`)
    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.subheader("Original Resume & Missing Skills")
        st.markdown(f"**Candidate:** {selected_candidate['Name']} (`{selected_candidate['Filename']}`)")
        st.markdown(f"**Current ATS Score:** `{selected_candidate['Final Score']:.2f} / 100`")

        st.markdown("#### Missing JD Skills to Address")
        if missing_skills:
            st.error(", ".join(missing_skills))
        else:
            st.success("None — Candidate matches all detected JD skills!")

        st.markdown("#### Original Extracted Resume Text")
        st.text_area(
            "Original Extracted Text",
            value=original_text,
            height=420,
            disabled=True,
            label_visibility="collapsed",
        )

    with right_col:
        st.subheader("Actionable Gap Analysis Tips & AI Auto-Editor")

        st.markdown("#### Actionable Gap Analysis Tips (3–5 Bullet Points)")
        tips = generate_gap_analysis_tips(
            candidate_name=selected_candidate["Name"],
            found_skills=found_skills,
            missing_skills=missing_skills,
        )
        for tip in tips:
            st.markdown(f"- {tip}")

        st.divider()
        st.markdown("#### AI-Rewritten Optimized Resume")

        run_rewrite = st.button(
            "Generate AI-Optimized Resume",
            type="primary",
            use_container_width=True,
        )

        rewrite_state_key = f"rewritten_{selected_candidate['Filename']}"

        if run_rewrite:
            if not api_key.strip():
                st.warning(
                    "Please enter your Gemini API Key in the sidebar before trying to edit a resume."
                )
            else:
                with st.spinner("Calling Gemini API to rewrite summary and experience bullet points..."):
                    try:
                        rewritten_md = rewrite_resume_with_gemini(
                            resume_text=original_text,
                            jd_text=jd_text,
                            missing_skills=missing_skills,
                            api_key=api_key,
                        )
                        st.session_state[rewrite_state_key] = rewritten_md
                    except Exception as exc:
                        st.error(f"Failed to generate AI-rewritten resume: {exc}")

        if rewrite_state_key in st.session_state:
            st.markdown(st.session_state[rewrite_state_key])
            st.download_button(
                label="Download Optimized Resume (Markdown)",
                data=st.session_state[rewrite_state_key].encode("utf-8"),
                file_name=f"Optimized_{selected_candidate['Name'].replace(' ', '_')}.md",
                mime="text/markdown",
            )
        else:
            st.caption(
                "Click **Generate AI-Optimized Resume** above to rewrite the candidate's "
                "professional summary and experience bullet points using the missing JD keywords."
            )


def main() -> None:
    st.sidebar.title("Resume Intelligence")

    selected_page = st.sidebar.radio(
        "Navigation",
        options=["Project Intro", "ATS Dashboard", "Resume Optimizer"],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.subheader("Gemini API Configuration")
    gemini_api_key = st.sidebar.text_input(
        "Enter Gemini API Key",
        type="password",
        help="Required for Page 3: Resume Optimizer AI Auto-Editor.",
    )
    if not gemini_api_key:
        st.sidebar.info("Enter your Gemini API key above to enable AI resume rewriting on Page 3.")

    st.sidebar.divider()
    st.sidebar.markdown("**Developer:**\nAyush Harshwardhan Meshram")

    if selected_page == "Project Intro":
        render_project_intro_page()
    elif selected_page == "ATS Dashboard":
        render_ats_dashboard_page()
    else:
        render_resume_optimizer_page(api_key=gemini_api_key)


if __name__ == "__main__":
    main()
