from nltk.corpus import stopwords
import string

stop_words = set(stopwords.words('english'))

# Extend with common non-skill words that often show up
additional_stopwords = {
    "job", "skills", "responsibilities", "qualifications", "description", "team", "use", "user", "end", "gap"
}
stop_words.update(additional_stopwords)

def is_valid_skill(word):
    return (
        len(word) > 2 and
        word.lower() not in stop_words and
        not all(char in string.punctuation for char in word)
    )

def detect_skill_gap(resume_keywords, job_keywords):
    resume_set = set([kw.lower() for kw in resume_keywords])
    job_set = set([kw.lower() for kw in job_keywords])

    raw_missing = job_set - resume_set
    filtered = [skill for skill in raw_missing if is_valid_skill(skill)]
    return sorted(filtered)
