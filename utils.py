"""
utils.py
--------
In-memory document text extraction utilities for Streamlit UploadedFile objects.
Strictly uses `io.BytesIO` to parse PDF (`PyPDF2`) and DOCX (`python-docx`) files
directly in memory without any hardcoded file paths or temporary disk writes.
"""

import io
import re
from PyPDF2 import PdfReader
import docx


def clean_text(text: str) -> str:
    """
    Normalizes whitespace and strips control characters while preserving
    line breaks needed for NER and regex parsing.
    """
    if not text:
        return ""
    text = text.replace("\x00", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """
    Extracts raw text from PDF bytes in-memory using `io.BytesIO` and `PyPDF2.PdfReader`.
    Raises a ValueError if the PDF is encrypted, corrupted, or contains no extractable text.
    """
    if not file_bytes:
        raise ValueError("Uploaded PDF file is empty.")

    pdf_stream = io.BytesIO(file_bytes)
    reader = PdfReader(pdf_stream)

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError("PDF is password-protected and cannot be read.") from exc

    pages_text = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            pages_text.append(extracted)

    combined = clean_text("\n".join(pages_text))
    if not combined:
        raise ValueError("No readable text found in PDF (file may be a scanned image).")

    return combined


def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    """
    Extracts raw text from DOCX bytes in-memory using `io.BytesIO` and `python-docx`.
    Reads both body paragraphs and table cells.
    """
    if not file_bytes:
        raise ValueError("Uploaded DOCX file is empty.")

    docx_stream = io.BytesIO(file_bytes)
    doc = docx.Document(docx_stream)
    blocks = []

    for para in doc.paragraphs:
        if para.text.strip():
            blocks.append(para.text.strip())

    for table in doc.tables:
        for row in table.rows:
            row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_cells:
                # Preserve unique cell values across merged table columns
                blocks.append(" | ".join(dict.fromkeys(row_cells)))

    combined = clean_text("\n".join(blocks))
    if not combined:
        raise ValueError("No readable text found inside the DOCX file.")

    return combined


def extract_text_from_upload(uploaded_file) -> str:
    """
    Reads a Streamlit `UploadedFile` object strictly in-memory via `io.BytesIO`.
    Supports `.pdf`, `.docx`, and `.txt` formats.
    """
    if uploaded_file is None:
        raise ValueError("No file object provided.")

    filename = getattr(uploaded_file, "name", "").lower()
    raw_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    if filename.endswith(".pdf"):
        return extract_text_from_pdf_bytes(raw_bytes)
    elif filename.endswith(".docx"):
        return extract_text_from_docx_bytes(raw_bytes)
    elif filename.endswith(".txt"):
        text = clean_text(raw_bytes.decode("utf-8", errors="ignore"))
        if not text:
            raise ValueError("Uploaded TXT file is empty.")
        return text
    else:
        raise ValueError(f"Unsupported file format for '{uploaded_file.name}'. Please upload a PDF or DOCX file.")
