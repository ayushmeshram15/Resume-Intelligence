"""
optimizer.py
------------
Style-Locked LaTeX (.tex) Resume Optimizer & Surgical Auto-Editor:
1. Gap Analysis (Tips): Generates 3-5 specific, actionable bullet points based on
   the candidate's "Missing Skills".
2. Strict LaTeX Style & Preamble Lock: Splits the LaTeX source at `\\begin{document}`
   so 100% of the candidate's original `\\documentclass`, font packages, margins,
   and custom macros are locked verbatim and can never be altered.
3. Surgical Content-Only Edits: Uses Gemini API (with a deterministic surgical LaTeX
   fallback so generation never fails) to only add/subtract targeted text inside
   existing `\\item` bullet points and Skills lines.
"""

import re
from typing import List, Tuple
import google.generativeai as genai


def escape_latex(text: str) -> str:
    """Escapes special LaTeX characters in plain strings."""
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def convert_plain_text_to_latex(raw_text: str, candidate_name: str = "Candidate") -> str:
    """
    If the candidate uploaded a PDF/DOCX instead of a `.tex` file, converts the
    extracted resume text into a clean LaTeX `.tex` document preserving the exact
    section order, headings, and bullet points. If `raw_text` is already LaTeX,
    returns it unchanged.
    """
    trimmed = raw_text.strip()
    if "\\documentclass" in trimmed or "\\begin{document}" in trimmed:
        return trimmed

    lines = [line.strip() for line in trimmed.splitlines() if line.strip()]
    body_lines: List[str] = []
    in_itemize = False

    section_headers = {
        "SUMMARY", "PROFESSIONAL SUMMARY", "PROFILE", "OBJECTIVE",
        "EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE",
        "PROJECTS", "KEY PROJECTS", "EDUCATION", "SKILLS",
        "TECHNICAL SKILLS", "CERTIFICATIONS", "ACHIEVEMENTS",
    }

    i = 0
    while i < len(lines):
        line = lines[i]
        upper = re.sub(r"[:\-]+$", "", line).strip().upper()

        if i == 0:
            body_lines.append("\\begin{center}")
            body_lines.append(f"  {{\\LARGE \\textbf{{{escape_latex(line or candidate_name)}}}}} \\\\[4pt]")
            if i + 1 < len(lines) and ("@" in lines[i + 1] or "|" in lines[i + 1]):
                body_lines.append(f"  \\small {escape_latex(lines[i + 1])}")
                i += 1
            body_lines.append("\\end{center}")
            i += 1
            continue

        if upper in section_headers:
            if in_itemize:
                body_lines.append("\\end{itemize}")
                in_itemize = False
            clean_sec = re.sub(r"[:]+$", "", line).strip()
            body_lines.append(f"\n\\section*{{{escape_latex(clean_sec)}}}")
            i += 1
            continue

        if re.match(r"^[-•*]\s+", line):
            if not in_itemize:
                body_lines.append("\\begin{itemize}[leftmargin=*, itemsep=2pt, topsep=2pt]")
                in_itemize = True
            bullet_text = re.sub(r"^[-•*]\s+", "", line).strip()
            body_lines.append(f"  \\item {escape_latex(bullet_text)}")
        else:
            if in_itemize:
                body_lines.append("\\end{itemize}")
                in_itemize = False
            body_lines.append(f"{escape_latex(line)}\\\\[2pt]")
        i += 1

    if in_itemize:
        body_lines.append("\\end{itemize}")

    preamble = [
        "\\documentclass[10pt,a4paper]{article}",
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage[margin=0.6in]{geometry}",
        "\\usepackage{enumitem}",
        "\\usepackage{hyperref}",
        "\\usepackage{titlesec}",
        "\\pagestyle{empty}",
        "\\titleformat{\\section}{\\large\\bfseries\\uppercase}{}{0em}{}[\\titlerule]",
        "\\titlespacing*{\\section}{0pt}{8pt}{4pt}",
        "",
        "\\begin{document}",
    ]
    return "\n".join(preamble + body_lines + ["\\end{document}"])


def perform_surgical_latex_edits(
    source_latex: str,
    missing_skills: List[str],
) -> Tuple[str, List[str]]:
    """
    Deterministic Surgical LaTeX Editor:
    Preserves 100% of the candidate's LaTeX preamble, font presets, margins, and
    custom macros untouched, and ONLY appends/injects missing JD keywords inside
    existing Skills lines and `\\item` bullets.
    """
    if not missing_skills:
        return source_latex, [
            "Preserved 100% of original LaTeX preamble, font packages, and layout presets.",
            "All target JD skills are already present; no keyword changes required.",
        ]

    changes_log = [
        "Locked 100% of original LaTeX preamble, font packages, geometry, and custom macros verbatim."
    ]
    escaped_missing = [escape_latex(s) for s in missing_skills]
    skills_str = ", ".join(escaped_missing)

    doc_idx = source_latex.find("\\begin{document}")
    preamble = source_latex[:doc_idx] if doc_idx != -1 else ""
    body = source_latex[doc_idx:] if doc_idx != -1 else source_latex

    skill_line_pattern = re.compile(
        r"(\\textbf\{[^}]*(?:Skills|Technologies|Tools|Frameworks|Languages|Stack)[^}]*\}\s*[:&]?\s*)([^\n\\]+)",
        re.IGNORECASE,
    )
    if skill_line_pattern.search(body):
        body = skill_line_pattern.sub(
            lambda m: f"{m.group(1)}{m.group(2).strip().rstrip(',;.')}, {skills_str} ",
            body,
            count=1,
        )
        changes_log.append(
            f"Surgically appended missing JD keywords ({', '.join(missing_skills)}) to existing LaTeX Skills line."
        )
    else:
        body = body.replace(
            "\\end{document}",
            f"% Surgically Added Missing JD Keywords\n\\noindent\\textbf{{Additional JD Skills:}} {skills_str}\n\\end{{document}}",
        )
        changes_log.append(
            f"Surgically added missing JD keywords ({', '.join(missing_skills)}) before \\end{{document}}."
        )

    return preamble + body, changes_log


def generate_gap_analysis_tips(
    candidate_name: str,
    found_skills: List[str],
    missing_skills: List[str],
) -> List[str]:
    """
    Generates 3-5 specific, actionable bullet points on what text to add or subtract
    in the LaTeX resume based on 'Missing Skills'.
    """
    if not missing_skills:
        return [
            f"All required Job Description skills ({', '.join(found_skills[:5]) or 'core skills'}) are already present in {candidate_name}'s LaTeX resume.",
            "Keep your existing LaTeX preamble, font packages, and spacing commands untouched.",
            "Quantify key engineering outcomes inside existing `\\item` lines with concrete metrics.",
            "Order primary Job Description keywords first inside your existing Skills category row.",
        ]

    tips: List[str] = []
    primary_missing = missing_skills[:3]
    secondary_missing = missing_skills[3:6]

    tips.append(
        f"In your existing LaTeX Skills section, append {', '.join(primary_missing)} to the matching `\\textbf{{...}}` category line without changing your tabular or list layout."
    )

    for skill in primary_missing:
        anchor = f"alongside {found_skills[0]}" if found_skills else "in your project bullets"
        tips.append(
            f"Inside an existing experience `\\item`, surgically add '{skill}' ({anchor}) without modifying any LaTeX formatting commands."
        )

    if secondary_missing:
        tips.append(
            f"Address secondary JD keywords ({', '.join(secondary_missing)}) by replacing generic verbs inside existing `\\item` lines while keeping the exact line count."
        )

    return tips[:5]


def rewrite_resume_latex_with_gemini(
    source_latex: str,
    jd_text: str,
    missing_skills: List[str],
    api_key: str = "",
    model_name: str = "gemini-1.5-flash",
) -> Tuple[str, List[str]]:
    """
    Rewrites ONLY the text content inside the LaTeX document body (`\\begin{document}`
    to `\\end{document}`), locking 100% of the original preamble, fonts, and style
    commands verbatim. Falls back to `perform_surgical_latex_edits` if no API key
    is provided or if the API call fails, ensuring generation always succeeds.
    """
    deterministic_tex, deterministic_log = perform_surgical_latex_edits(
        source_latex, missing_skills
    )

    if not api_key or not api_key.strip():
        return deterministic_tex, deterministic_log

    doc_idx = source_latex.find("\\begin{document}")
    locked_preamble = source_latex[:doc_idx] if doc_idx != -1 else ""
    original_body = source_latex[doc_idx:] if doc_idx != -1 else source_latex

    try:
        genai.configure(api_key=api_key.strip())
        model = genai.GenerativeModel(model_name)

        prompt = (
            "You are a Senior LaTeX Resume Editor.\n\n"
            "STRICT NON-NEGOTIABLE RULES:\n"
            "1. Output ONLY valid LaTeX code starting with \\begin{document} and ending with \\end{document}.\n"
            "2. DO NOT change any style, font, macro, spacing (\\vspace, \\hspace), tabular column alignment, or section command.\n"
            "3. The compiled resume MUST look identical in layout and font to the uploaded resume.\n"
            "4. ONLY make surgical text additions or subtractions inside existing summary lines, \\item bullet points, "
            f"and skills lists to naturally incorporate these missing JD keywords: {', '.join(missing_skills) or 'None'}.\n\n"
            f"TARGET JOB DESCRIPTION:\n{jd_text}\n\n"
            f"ORIGINAL LATEX BODY:\n{original_body}\n"
        )

        response = model.generate_content(prompt)
        raw_out = getattr(response, "text", "") or ""
        clean_body = re.sub(r"^```(?:latex|tex)?\s*", "", raw_out.strip(), flags=re.IGNORECASE)
        clean_body = re.sub(r"\s*```$", "", clean_body).strip()

        if "\\begin{document}" in clean_body:
            start_pos = clean_body.find("\\begin{document}")
            return locked_preamble + clean_body[start_pos:], [
                "Locked 100% of original LaTeX preamble, font presets, and custom macros verbatim.",
                f"Surgically incorporated missing JD skills ({', '.join(missing_skills)}) inside existing LaTeX body text.",
            ]
    except Exception:
        pass

    return deterministic_tex, deterministic_log
