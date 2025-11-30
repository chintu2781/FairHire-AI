import pdfplumber
import docx
import re
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np

nlp = spacy.load("en_core_web_sm")
EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

COMMON_SKILLS = [
    "python","sql","docker","kubernetes","pytorch","tensorflow","nlp",
    "machine learning","deep learning","pandas","numpy","scikit-learn","git",
    "aws","azure","spark","hadoop","excel","linux"
]

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def clean_text(txt):
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

def load_resume_text(path):
    ext = path.split(".")[-1].lower()
    if ext == "pdf":
        return clean_text(extract_text_from_pdf(path))
    elif ext == "docx":
        return clean_text(extract_text_from_docx(path))
    else:
        return clean_text(open(path, "r", encoding="utf-8").read())

def extract_skills(text):
    found = []
    text = text.lower()
    for s in COMMON_SKILLS:
        if s in text:
            found.append(s)
    return list(set(found))

def embed_text(text):
    return EMB_MODEL.encode([text])[0]

def score_resume(resume_text, jd_text):
    v1 = embed_text(resume_text)
    v2 = embed_text(jd_text)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def recommend_actions(resume_text, jd_text):
    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)
    missing = [s for s in jd_skills if s not in resume_skills]

    suggestions = []
    for skill in missing:
        suggestions.append(f"Add or improve: {skill}")

    return {
        "resume_skills": resume_skills,
        "missing_skills": missing,
        "suggestions": suggestions
    }
