import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

COURSE_DATA_PATH = 'data/courses.csv'

def load_courses():
    return pd.read_csv(COURSE_DATA_PATH)


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

COURSE_DATA_PATH = 'data/courses.csv'

def load_courses():
    return pd.read_csv(COURSE_DATA_PATH)

def recommend_courses(missing_skills, top_n=3):
    courses_df = load_courses()
    if courses_df.empty:
        return {}

    courses_df['text'] = courses_df['title'].fillna('') + " " + courses_df['description'].fillna('')

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(courses_df['text'])

    skill_course_scores = {}
    for skill in missing_skills:
        skill_vec = tfidf.transform([skill])
        scores = cosine_similarity(skill_vec, tfidf_matrix).flatten()
        
        # Only keep relevant courses (cosine similarity > 0.1)
        relevant_indices = [i for i, score in enumerate(scores) if score > 0.1]
        if not relevant_indices:
            continue  # No strong matches for this skill

        top_indices = sorted(relevant_indices, key=lambda i: scores[i], reverse=True)[:top_n]
        top_courses = courses_df.iloc[top_indices][['title', 'url']].to_dict(orient='records')
        skill_course_scores[skill] = top_courses

    return skill_course_scores