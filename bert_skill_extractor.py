from keybert import KeyBERT
import re

model = KeyBERT(model='paraphrase-MiniLM-L6-v2')

NON_SKILL_PHRASES = ['gmail', 'email', 'linkedin', 'objective', 'contact', 'career', 'bangalore', 'address', 'phone']

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_likely_skill(phrase):
    phrase = phrase.lower()
    if any(bad in phrase for bad in NON_SKILL_PHRASES):
        return False
    if len(phrase) < 3 or len(phrase.split()) > 5:
        return False
    if phrase.isdigit():
        return False
    return True

def extract_skills_from_text(text, top_n=20, diversity=0.3):
    text = clean_text(text)
    if not text or len(text.split()) < 10:
        return []

    keywords = model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        use_mmr=True,
        diversity=diversity,
        top_n=top_n
    )

    skills = sorted(set([
        phrase.lower()
        for phrase, _ in keywords
        if is_likely_skill(phrase)
    ]))

    return skills
