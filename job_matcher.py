import os
import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from job_analyzer import extract_text_from_job_file
from bert_skill_extractor import extract_skills_from_text
JOBS_FOLDER = 'data/jobs'

def get_all_job_descriptions():
    job_files = glob.glob(os.path.join(JOBS_FOLDER, '*'))
    jobs = []

    for path in job_files:
        text = extract_text_from_job_file(path)
        if text.strip():  # skip empty jobs
            jobs.append({
                'filename': os.path.basename(path),
                'text': text
            })

    return pd.DataFrame(jobs)

def match_jobs(resume_keywords, top_n=5):
    resume_text = " ".join(resume_keywords)
    jobs_df = get_all_job_descriptions()

    jobs_df['job_keywords'] = jobs_df['text'].apply(lambda text: extract_skills_from_text(text) or [])
    jobs_df['text_clean'] = jobs_df['job_keywords'].apply(lambda kws: " ".join(kws))

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform([resume_text] + jobs_df['text_clean'].tolist())

    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    jobs_df['score'] = cosine_sim

    top_matches = jobs_df.sort_values(by='score', ascending=False).head(top_n)

    
    return top_matches[['filename', 'score', 'text']].to_dict(orient='records')
