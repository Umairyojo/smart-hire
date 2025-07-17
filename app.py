from flask import Flask, request, render_template, redirect, url_for
import os
from resume_parser import extract_resume_data
from job_analyzer import extract_job_description_data
from skill_gap import detect_skill_gap
from recommender import recommend_courses
from job_matcher import match_jobs
from bert_skill_extractor import extract_skills_from_text




app = Flask(__name__)
UPLOAD_FOLDER = 'data/resumes'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

#@app.route('/upload_resume', methods=['GET', 'POST'])
def upload_resume():
    if request.method == 'POST':
        file = request.files['resume']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            extracted_data = extract_resume_data(filepath)
            return render_template('result.html', data=extracted_data)
    return render_template('upload.html')

#@app.route('/upload_job', methods=['GET', 'POST'])
def upload_job():
    if request.method == 'POST':
        file = request.files['job']
        if file:
            filepath = os.path.join('data/jobs', file.filename)
            file.save(filepath)
            job_data = extract_job_description_data(filepath)
            return render_template('job_result.html', data=job_data)
    return render_template('upload_job.html')

@app.route('/analyze_gap', methods=['GET', 'POST'])
def analyze_gap():
    if request.method == 'POST':
        resume_file = request.files['resume']
        job_file = request.files.get('job')
        job_text = request.form.get('job_text', '').strip()

        from bert_skill_extractor import extract_skills_from_text  # Add this here if needed

        # Parse resume
        if resume_file:
            resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(resume_path)
            resume_data = extract_resume_data(resume_path)
        else:
            return "Resume file is required.", 400

        # Job description: file or text
        if job_file and job_file.filename:
            job_path = os.path.join('data/jobs', job_file.filename)
            job_file.save(job_path)
            job_data = extract_job_description_data(job_path)
            job_keywords = job_data["skills_detected"]
        elif job_text:
            job_keywords = extract_skills_from_text(job_text)
        else:
            return "Provide job description via upload or paste.", 400

        missing_skills = detect_skill_gap(
            resume_data["skills_detected"],
            job_keywords
        )

        recommended_courses = recommend_courses(missing_skills)

        return render_template(
            'gap_result.html',
            resume_keywords=resume_data["skills_detected"],
            job_keywords=job_keywords,
            missing_skills=missing_skills,
            recommended_courses=recommended_courses
        )

    return render_template('upload_gap.html')


@app.route('/match_jobs', methods=['GET', 'POST'])
def match_jobs_view():
    if request.method == 'POST':
        resume_file = request.files['resume']
        if not resume_file:
            return "Please upload a resume.", 400

        resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
        resume_file.save(resume_path)
        resume_data = extract_resume_data(resume_path)

        matched_jobs = match_jobs(resume_data['skills_detected'])  # ✅

        return render_template('job_matches.html',
                               resume_keywords=resume_data['top_keywords'],
                               matched_jobs=matched_jobs)

    return render_template('upload_resume_match.html')


if __name__ == '__main__':
    app.run(debug=True)
