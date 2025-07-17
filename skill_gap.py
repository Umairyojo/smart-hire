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

from rapidfuzz import fuzz

def detect_skill_gap(resume_keywords, job_keywords, similarity_threshold=80):
    resume_set = set(kw.lower() for kw in resume_keywords)
    job_set = set(kw.lower() for kw in job_keywords)

    missing_skills = []
    for job_skill in job_set:
        if not is_valid_skill(job_skill):
            continue

        match_found = False
        for resume_skill in resume_set:
            if fuzz.token_sort_ratio(job_skill, resume_skill) >= similarity_threshold:
                match_found = True
                break

        if not match_found:
            missing_skills.append(job_skill)

    return sorted(set(missing_skills))

