import spacy
import os
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document

nlp = spacy.load("en_core_web_sm")

def extract_text_from_job_file(filepath):
    if filepath.endswith('.pdf'):
        return extract_pdf_text(filepath)
    elif filepath.endswith('.docx'):
        doc = Document(filepath)
        return '\n'.join([p.text for p in doc.paragraphs])
    elif filepath.endswith('.txt'):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return ''

def extract_job_description_data(filepath):
    text = extract_text_from_job_file(filepath)
    doc = nlp(text)

    keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop]

    return {
        "filename": os.path.basename(filepath),
        "total_words": len(text.split()),
        "top_keywords": list(set(keywords))[:15],  # top 15 potential skill-related terms
        "raw_text": text[:300] + "..."  # preview snippet
    }
