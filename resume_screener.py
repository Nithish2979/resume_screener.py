import streamlit as st
import os
import fitz  # PyMuPDF
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

# Create resume folder if not exists
UPLOAD_FOLDER = "resumes"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.set_page_config(page_title="AI Resume Screener", layout="centered")
st.title("🤖 AI-Powered Resume Screening System")

# --- Upload resumes ---
uploaded_files = st.file_uploader("Upload Resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
job_description = st.text_area("Paste Job Description", height=200)

# --- Function to extract text ---
def extract_text(file_path):
    if file_path.endswith(".pdf"):
        text = ""
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        return text
    elif file_path.endswith(".docx"):
        return docx2txt.process(file_path)
    return ""

# --- Suggestions based on missing content ---
def generate_detailed_suggestions(job_desc, resume_text):
    job_words = set(job_desc.lower().split())
    resume_words = set(resume_text.lower().split())
    missing = job_words - resume_words

    present = job_words & resume_words

    tech_keywords = ["python", "java", "c++", "sql", "machine learning", "deep", "flask", "django",
                     "nlp", "tensorflow", "pandas", "numpy", "matplotlib", "api", "deployment", "mongodb",
                     "selenium", "testng", "junit", "automation", "postman", "jmeter", "jenkins", "maven"]
    certs_keywords = ["aws", "azure", "gcp", "oracle", "ibm", "meta", "google", "coursera", "udemy"]
    soft_skills = ["communication", "problem", "teamwork", "leadership", "detail", "time"]
    project_keywords = ["chatbot", "recommendation", "testing", "qa", "automation", "ai & ml", "dashboard"]

    suggestions = []
    suggestions.append("1. 🎯 **Job Title & Objective Match**: ✅" if any(word in resume_words for word in ["developer", "engineer", "ai", "ml"]) else
                   "1. 🎯 **Job Title & Objective Match**: ❌\nMake sure your resume clearly mentions your intent to work as an AI/ML or relevant role.")

    # Technical Skills
    present_tech = [word for word in tech_keywords if word in resume_words]
    missing_tech = [word for word in tech_keywords if word in missing]
    skill_suggestion = "2. 🛠 **Skills & Tools Match**: "
    if present_tech:
        skill_suggestion += "⚠️ Partial\n✅ Present: " + ", ".join(present_tech)
        if missing_tech:
            skill_suggestion += f"\n❌ Missing Key Skills: {', '.join(missing_tech)}"
    else:
        skill_suggestion += f"❌ No relevant technical skills found. Add: {', '.join(missing_tech)}"
    suggestions.append(skill_suggestion)

    # Certifications
    present_certs = [word for word in certs_keywords if word in resume_words]
    if present_certs:
        suggestions.append("3. 📄 **Certifications & Tools**: ✅\nMentioned: " + ", ".join(present_certs))
    else:
        suggestions.append("3. 📄 **Certifications & Tools**: ❌ Missing\nConsider adding certifications from Udemy, Coursera, or Google/AWS related to testing or ML.")

    # Soft Skills
    present_soft = [word for word in soft_skills if word in resume_words]
    if present_soft:
        suggestions.append("4. 📐 **Soft Skills**: ✅\nMentioned: " + ", ".join(present_soft))
    else:
        suggestions.append("4. 📐 **Soft Skills**: ⚠️ Not clearly mentioned\nHighlight relevant soft skills such as communication, teamwork, or attention to detail.")

    # Project Suggestions
    present_projects = [p for p in project_keywords if p in resume_words]
    if present_projects:
        suggestions.append("5. 📁 **Project Experience**: ✅\nMentioned: " + ", ".join(present_projects))
    else:
        suggestions.append("5. 📁 **Project Experience**: ⚠️ Generic\nProjects are decent but not related to AI/ML or automation.\n\nExample: Add chatbot, testing automation, or AI-based dashboard to demonstrate alignment with job description.")

    return suggestions

# --- Circular Score Chart ---
def draw_score_chart(score):
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    wedges, _ = ax.pie([score, 100 - score], startangle=90, colors=['#4CAF50', '#eeeeee'], wedgeprops=dict(width=0.3))
    ax.text(0, 0, f"{score}%", ha='center', va='center', fontsize=10, weight='bold')
    ax.set(aspect="equal")
    plt.tight_layout()
    return fig

# --- Session state init ---
if 'results' not in st.session_state:
    st.session_state.results = []

# --- Process and Score ---
if st.button("Score Resumes"):
    if uploaded_files and job_description:
        results = []
        for file in uploaded_files:
            file_path = os.path.join(UPLOAD_FOLDER, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())

            resume_text = extract_text(file_path)
            documents = [job_description.lower(), resume_text.lower()]
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(documents)
            raw_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            score = round(min(max((raw_score + 0.33) * 100, 45), 99), 2)

            suggestions = generate_detailed_suggestions(job_description, resume_text)
            results.append((file.name, score, suggestions))

        st.session_state.results = sorted(results, key=lambda x: x[1], reverse=True)

# --- Display Results ---
if st.session_state.results:
    st.subheader("📊 Resume Match Results")
    for i, (name, score, suggestions) in enumerate(st.session_state.results, 1):
        st.markdown(f"### {i}. {name}")

        #layout in colums 
        col1 , col2 = st.columns([1,2])

        with col1:
            fig = draw_score_chart(score)
            st.pyplot(fig)
        
        with col2:
            show_suggestions = st.checkbox(f"💡 Show Suggestions for {name}", key=f"suggestions_{i}")
        if show_suggestions:
            st.markdown("#### 📌 Breakdown of ATS Evaluation")
            for s in suggestions:
                st.markdown(f"- {s}")

elif uploaded_files or job_description:
    st.info("⬆️ Click 'Score Resumes' to analyze the uploaded resumes.")
