import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

COURSE_DATA_PATH = 'data/courses.csv'

def load_courses():
    return pd.read_csv(COURSE_DATA_PATH)

def recommend_courses(missing_skills, top_n=7):
    courses_df = load_courses()
    if courses_df.empty:
        return []

    courses_df['text'] = courses_df['title'].fillna('') + " " + courses_df['description'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(courses_df['text'])

    # Combine all missing skills into one search query
    query = " ".join(missing_skills)
    query_vec = tfidf.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = scores.argsort()[-top_n:][::-1]
    top_courses = courses_df.iloc[top_indices][['title', 'url']].to_dict(orient='records')
    return top_courses
