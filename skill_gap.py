def compute_skill_gap(resume_skills: list[str], jd_skills: list[str]) -> list[str]:
    return sorted(set(jd_skills) - set(resume_skills))