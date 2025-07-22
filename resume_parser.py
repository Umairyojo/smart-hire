from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document
from bert_skill_extractor import SkillExtractor
import os

from bert_skill_extractor import SkillExtractor

def get_extractor():
    if not hasattr(get_extractor, "_instance"):
        print("[INFO] Loading SkillExtractor...")
        get_extractor._instance = SkillExtractor()
    return get_extractor._instance

def extract_text_from_file(filepath):
    if filepath.endswith('.pdf'):
        return extract_pdf_text(filepath)
    elif filepath.endswith('.docx'):
        doc = Document(filepath)
        return '\n'.join([p.text for p in doc.paragraphs])
    elif filepath.endswith('.txt'):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

def extract_resume_data(filepath):
    text = extract_text_from_file(filepath)
    extractor = get_extractor()
    skills = extractor.extract_skills(text)
    keywords = [word for word in text.split() if len(word) > 2 and word.isalpha()]
    return {
        "filename": os.path.basename(filepath),
        "total_words": len(text.split()),
        "top_keywords": list(set(keywords))[:10],
        "skills_detected": skills if skills else ["(no skills detected)"]
    }
