# FairHire AI — Resume Ranking & Bias-Free Recommendations

FairHire AI is an intelligent HR screening tool that:
- Parses and analyzes resumes
- Matches resumes with job descriptions using deep learning embeddings
- Detects and reduces hiring bias (gender, location, education)
- Generates personalized recommendations for candidates
- Provides explainability using LIME

Tech Stack:
- Sentence-BERT (SBERT)
- spaCy NLP
- LIME & SHAP explainability
- Streamlit Web App

Run the app:
    streamlit run src/streamlit_app.py

Dataset needed: resumes (PDF/DOCX/TXT) + job descriptions.
