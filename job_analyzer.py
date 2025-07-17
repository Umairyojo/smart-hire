import os
import re
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document
from bert_skill_extractor import extract_skills_from_text

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

def extract_possible_skill_lines(text):
    lines = text.splitlines()
    candidates = []
    # Common JD sections
    trigger_keywords = [
        'skills', 'requirements', 'qualifications', 'responsibilities',
        'technical skills', 'preferred', 'must have', 'experience'
    ]

    capture = False
    for line in lines:
        line_lower = line.lower().strip()
        if any(k in line_lower for k in trigger_keywords):
            capture = True
            continue
        if capture:
            if line.strip() == "" or re.match(r'^[A-Z][a-z]+\s*:', line):
                break
            candidates.append(line.strip())
    return candidates

def extract_inline_from_lines(lines):
    extracted_phrases = []
    for line in lines:
        part = line
        for sep in [':', '–', '-', '•', '*']:
            if sep in line:
                part = line.split(sep, 1)[1]
                break
        cleaned = re.sub(r'[•◆●→•■*•\-]', '', part)
        extracted_phrases.append(cleaned.strip())
    return ' '.join(extracted_phrases)

def extract_job_description_data(filepath):
    text = extract_text_from_job_file(filepath)

    lines = extract_possible_skill_lines(text)
    inline_text = extract_inline_from_lines(lines)

    skill_input = inline_text if len(inline_text.split()) >= 5 else text
    skills = extract_skills_from_text(skill_input)

    # Prepare keywords (for legacy use or display)
    keywords = [word for word in text.split() if len(word) > 2 and word.isalpha()]

    return {
        "filename": os.path.basename(filepath),
        "total_words": len(text.split()),
        "top_keywords": list(set(keywords))[:15],
        "skills_detected": skills if skills else ["(no skills detected)"],
        "raw_text": text[:300] + "...",
        "full_text": text  # optional, if you want preview in matching
    }
