# SmartHire – AI-Powered Resume Screening System

SmartHire is an AI-powered recruitment automation platform designed to simplify resume screening, job matching, skill gap analysis, and candidate shortlisting. The system helps HR teams reduce manual resume review effort while helping job seekers understand their profile strength, missing skills, recommended courses, and suitable job opportunities.

---

## 1. Project Overview

SmartHire is a Flask-based AI recruitment platform with two main roles:

- **HR / Recruiter**
- **User / Job Seeker**

HR users can post jobs, view applicants, screen candidates, analyze applicant counts, and notify candidates. Job seekers can create a profile, upload resumes, view available jobs, apply for jobs, check match scores, view skill gaps, receive course recommendations, and receive HR notifications.

---

## 2. Problem Statement

Traditional resume screening has several challenges:

- Recruiters spend a lot of time manually reading resumes.
- Candidate shortlisting may be inconsistent across recruiters.
- Skill gaps are not clearly visible to job seekers.
- Candidates do not always know which jobs fit their profile.
- HR teams need a faster way to analyze applicant data.
- Manual screening can introduce human bias.

SmartHire solves this by using AI and NLP to automate resume parsing, job matching, skill gap analysis, and candidate shortlisting.

---

## 3. Proposed Solution

SmartHire provides a web-based platform where:

1. HR users post job openings with title, description, skills, deadline, and optional application URL.
2. Job seekers create and manage their profile.
3. Users upload resumes.
4. The system parses resumes and extracts structured information.
5. The system compares resume data with job descriptions.
6. Job match scores and AI feedback are displayed.
7. Skill gaps are identified.
8. Recommended courses are suggested based on missing skills.
9. HR users notify candidates with status messages.
10. Users view received notifications.

---

## 4. Key Features

### HR Features

- HR login and role-based dashboard.
- Company profile section with editable company name and description.
- Separate Post Job page.
- Job posting with title, description, required skills, deadline, and optional application URL.
- Posted jobs listed on HR Home.
- Candidate screening for each job.
- Applied candidates table with candidate name, job title, score, status message box, and notify button.
- Resume-JD Match Analytics showing number of applicants per job.

### User Features

- User registration and login.
- User profile form with full name, career objective, gender, education, experience, skills, hobbies, and resume upload.
- User Home page showing profile summary and progress.
- User Dashboard with available jobs, resume upload, job matches, skill gap report, and recommended courses.
- Apply for jobs.
- View job details page.
- Receive HR notifications.
- Notifications page to view messages from HR.

### AI / NLP Features

- Resume parsing from uploaded files.
- Skill extraction from resumes.
- Job description and resume matching.
- Fit score calculation.
- AI-generated feedback.
- Skill gap detection.
- Course recommendation based on missing skills.

---

## 5. Technology Stack

### Backend

- Python
- Flask
- Flask-SQLAlchemy
- SQLite
- Pandas
- Matplotlib
- CSV file handling

### Frontend

- HTML
- CSS
- Jinja2 templates
- Responsive dark-theme UI

### AI / NLP

- Google Gemini API
- Resume parsing logic
- Skill matching logic
- Job matching logic
- Skill gap analysis

---

## 6. Project Directory Structure

```text
Smart-hire-main/
│
├── data/
│   ├── jobs/
│   ├── resumes/
│   ├── applications.csv
│   ├── courses.csv
│   ├── jobs.csv
│   ├── posted.csv
│   └── smarthire.db
│
├── static/
│   └── styles.css
│
├── templates/
│   ├── hr_dashboard.html
│   ├── hr_home.html
│   ├── index.html
│   ├── job_view.html
│   ├── layout.html
│   ├── login.html
│   ├── post_job.html
│   ├── register.html
│   ├── screen_candidates.html
│   ├── upload_resume.html
│   ├── user_dashboard.html
│   ├── user_home.html
│   ├── user_notifications.html
│   └── user_profile.html
│
├── app.py
├── auth_bp.py
├── course_recommender.py
├── dashboard.py
├── extensions.py
├── job_matcher.py
├── resume_parser.py
├── skill_gap.py
├── requirements.txt
└── readme.md
```

---

## 7. Main Modules

### `app.py`

Main Flask application file. It manages app configuration, database setup, user routes, HR routes, job posting, resume upload, notifications, dashboards, and CSV helper functions.

### `auth_bp.py`

Handles registration, login, password hashing, role-based redirection, and session setup.

### `resume_parser.py`

Reads uploaded resumes and extracts structured resume details such as skills, education, experience, and keywords.

### `job_matcher.py`

Compares resume information with job requirements and returns a match score with feedback.

### `skill_gap.py`

Compares candidate skills with job required skills and identifies missing skills.

### `course_recommender.py`

Reads `courses.csv` and recommends courses based on missing skills.

### `dashboard.py`

Generates charts and analytics using Matplotlib.

### `extensions.py`

Contains the shared SQLAlchemy database instance.

---

## 8. Data Storage

SmartHire currently uses a hybrid storage approach.

### SQLite Database

Used for:

- Users
- Company profiles
- User profiles
- Notifications
- Optional match cache tables

### CSV Files

| File | Purpose |
|---|---|
| `jobs.csv` | Stores full job details |
| `posted.csv` | Stores HR-posted job records for HR Home |
| `applications.csv` | Stores job applications |
| `courses.csv` | Stores course recommendation data |

---

## 9. AI Request Optimization

Since Gemini free tier has request limits, SmartHire should minimize AI calls.

Recommended optimization strategy:

1. **Parse resume only once per upload**
   - Use a resume file hash.
   - Reuse parsed resume data if the same file is uploaded again.

2. **Use local skill matching first**
   - Compare resume skills and job skills locally using keyword overlap.
   - Use Gemini only for deeper feedback if needed.

3. **Cache job match results**
   - Cache using `resume_hash + job_signature`.
   - Reuse old scores instead of calling Gemini repeatedly.

4. **Limit Gemini matching to top jobs**
   - Use local matching for all jobs.
   - Use Gemini only for the top 3 jobs if required.

5. **Avoid Gemini calls on page refresh**
   - Never call Gemini repeatedly just because the user refreshed the dashboard.

This keeps the app within strict free-tier API limits.

---

## 10. User Flow

### Job Seeker Flow

1. Register as User.
2. Login.
3. Fill profile details.
4. Upload resume.
5. View available jobs.
6. Apply to suitable jobs.
7. View job match scores.
8. Check skill gap report.
9. View recommended courses.
10. Receive HR notifications.

### HR Flow

1. Register/Login as HR.
2. Add company profile details.
3. Post job.
4. View posted jobs on HR Home.
5. Open HR Dashboard.
6. Screen applicants.
7. View applicant match scores.
8. Send notification/status message to candidates.
9. View applicant analytics.

---

## 11. Resume Matching Workflow

```text
Resume Upload
      ↓
Resume Text Extraction
      ↓
Skill / Education / Experience Extraction
      ↓
Compare Resume Skills with Job Skills
      ↓
Generate Match Score
      ↓
Show Feedback, Skill Gap, and Course Recommendations
```

---

## 12. Notification Workflow

```text
User applies for job
      ↓
HR views applied candidate in HR Dashboard
      ↓
HR enters status message
      ↓
HR clicks Notify
      ↓
Notification is stored in database
      ↓
User views message in Notifications page
```

---

## 13. Fairness and Bias Reduction

SmartHire should reduce bias by:

- Ranking candidates based on job-relevant skills, experience, and education.
- Avoiding protected attributes such as gender, age, religion, and personal background in scoring.
- Keeping HR users in control of final decisions.
- Showing transparent feedback for match scores.
- Maintaining audit-friendly records of notifications and decisions.

---

## 14. Installation and Setup

### Step 1: Open project folder

```bash
cd Smart-hire-main
```

### Step 2: Create virtual environment

```bash
python -m venv venv
```

### Step 3: Activate virtual environment

Windows PowerShell:

```bash
.\venv\Scripts\Activate.ps1
```

Windows CMD:

```bash
venv\Scripts\activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Create `.env` file

```env
FLASK_SECRET_KEY=your_secret_key
GEMINI_API_KEY=your_gemini_api_key
```

### Step 6: Run application

```bash
python app.py
```

### Step 7: Open browser

```text
http://127.0.0.1:5000
```

---

## 15. Default Test Accounts

If enabled in `setup_database(app)`, the app may create:

| Role | Username | Password |
|---|---|---|
| User | `test_user` | `test_password` |
| HR | `test_hr` | `hr_password` |

---

## 16. Testing Checklist

### HR Testing

- Register as HR.
- Login as HR.
- Save company profile.
- Post a job.
- Verify job appears on HR Home.
- Verify job appears in HR Dashboard.
- Apply as user to the job.
- Verify candidate appears in Applied Candidates table.
- Send notification to candidate.
- Verify analytics chart updates.

### User Testing

- Register as User.
- Login as User.
- Fill profile details.
- Upload resume.
- Verify resume parsing success message.
- Verify available jobs are displayed.
- Apply to a job.
- Verify applied status.
- Verify job matches are shown.
- Verify skill gap report.
- Verify course recommendations.
- Verify HR notification appears on Notifications page.

---

## 17. Future Enhancements

- Move jobs and applications from CSV to database tables.
- Add stable UUID job IDs instead of CSV row index.
- Add read/unread notification actions.
- Add HR candidate status tracking.
- Add email notifications.
- Add pagination and search filters.
- Add resume ranking leaderboard.
- Add course completion tracking.
- Add candidate profile export.
- Add fairness audit report.
- Add stronger Gemini caching and background processing.

---

## 18. Known Limitations

- Gemini free-tier API may hit rate limits if too many jobs are matched at once.
- CSV-based job IDs can shift if rows are manually deleted or reordered.
- Matching quality depends on resume parsing quality and job description quality.
- Course recommendations depend on the quality of `courses.csv`.
- For production, database tables are recommended instead of CSV files.

---

## 19. Conclusion

SmartHire provides a practical AI-assisted recruitment workflow for both HR users and job seekers. It automates resume parsing, job matching, skill gap analysis, and candidate notifications while keeping HR users in control of final hiring decisions. With caching, database migration, and fairness improvements, SmartHire can become a scalable and production-ready recruitment automation platform.
