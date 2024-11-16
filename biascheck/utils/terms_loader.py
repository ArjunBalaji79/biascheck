import re
import os
import fitz  # PyMuPDF for PDF parsing
import pandas as pd


def preprocess_text(text):
    """
    Preprocess text to remove references, bibliographies, and unnecessary tokens.
    Parameters:
        text (str): Raw text input.
    Returns:
        str: Cleaned text.
    """
    # Remove brackets with numbers (e.g., [3], [29])
    text = re.sub(r"\[\d+\]", "", text)
    # Remove multiple spaces or line breaks
    text = re.sub(r"\s+", " ", text)
    # Optionally remove a References section if it exists
    text = re.sub(r"(?i)references.*", "", text, flags=re.DOTALL)
    return text

def load_terms(terms):
    """
    Load terms from a file or list.
    Parameters:
        terms (str or list): Path to terms file or a list of terms.
    Returns:
        list: Loaded terms.
    """
    if isinstance(terms, str) and os.path.exists(terms):
        with open(terms, "r", encoding="utf-8") as file:
            return file.read().splitlines()
    elif isinstance(terms, list):
        return terms
    return []

def load_document(file_path):
    """
    Load text from a document (PDF or TXT).
    Parameters:
        file_path (str): Path to the document.
    Returns:
        str: Extracted text.
    """
    if file_path.endswith(".pdf"):
        text = parse_pdf(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    else:
        raise ValueError("Unsupported file format. Please provide a PDF or TXT file.")
    
    return preprocess_text(text)  # Preprocess the text before returning it

def parse_pdf(pdf_path):
    """
    Parse text content from a PDF file.
    Parameters:
        pdf_path (str): Path to the PDF file.
    Returns:
        str: Extracted text.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text