import os
import re
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document
from bert_skill_extractor import extract_skills_from_text

def extract_text_from_file(filepath):
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

    # Section headers to detect
    trigger_keywords = [
        'skills', 'technical skills', 'technologies', 'tools', 'frameworks',
        'languages', 'software', 'platforms', 'competencies', 'proficiencies'
    ]

    capture = False
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()

        # Start capturing if header found
        if any(k in line_lower for k in trigger_keywords):
            capture = True
            continue

        if capture:
            if line.strip() == '' or re.match(r'^[A-Z][a-z]+\s*:', line):  # likely a new section
                break
            candidates.append(line.strip())

    return candidates

def extract_inline_skills(lines):
    extracted_phrases = []

    for line in lines:
        # Grab things after ":" or bullet symbols
        if ':' in line:
            part = line.split(':', 1)[1]
        elif '–' in line:
            part = line.split('–', 1)[1]
        elif '•' in line:
            part = line.split('•', 1)[-1]
        else:
            part = line

        # Clean common punctuation
        cleaned = re.sub(r'[•◆●→•■*•\-]', '', part)
        extracted_phrases.append(cleaned.strip())

    return ' '.join(extracted_phrases)

def extract_resume_data(filepath):
    text = extract_text_from_file(filepath)

    # Try skill-focused extraction
    lines = extract_possible_skill_lines(text)
    inline_skill_text = extract_inline_skills(lines)

    # If not enough content, fallback to entire text
    skill_input_text = inline_skill_text if len(inline_skill_text.split()) >= 5 else text
    skills = extract_skills_from_text(skill_input_text)

    keywords = [word for word in text.split() if len(word) > 2 and word.isalpha()]

    return {
        "filename": os.path.basename(filepath),
        "total_words": len(text.split()),
        "top_keywords": list(set(keywords))[:10],
        "skills_detected": skills if skills else ["(no skills detected)"]
    }
