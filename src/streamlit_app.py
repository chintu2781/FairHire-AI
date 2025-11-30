import streamlit as st
from ranker import rank_resumes, bias_check
from explanation import make_predict_fn, explainer
from utils import load_resume_text
import tempfile
import os

st.title("FairHire AI — Resume Ranking + Recommendations")

jd = st.text_area("Paste Job Description Here", height=150)

uploads = st.file_uploader("Upload Resumes", type=["pdf","docx","txt"], accept_multiple_files=True)

if st.button("Run"):
    if jd and uploads:
        paths = []
        temp_dir = tempfile.mkdtemp()

        for file in uploads:
            path = os.path.join(temp_dir, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            paths.append(path)

        results = rank_resumes(paths, jd)

        st.subheader("Results")
        for r in results:
            st.write(f"### {os.path.basename(r['resume'])}")
            st.write(f"**Score:** {r['final_score']}")

            st.write("**Extracted Skills:**", ", ".join(r["recommendations"]["resume_skills"]))
            st.write("**Missing Skills:**", ", ".join(r["recommendations"]["missing_skills"]))
            st.write("**Suggestions:**")
            for s in r["recommendations"]["suggestions"]:
                st.write("-", s)

            bc = bias_check(r["text"], jd)
            st.write(f"**Bias Check:** Score change = {bc['change']:.4f}")

            predict_fn = make_predict_fn(jd)
            explanation = explainer.explain_instance(r["text"], predict_fn, num_features=5)
            st.write("**Top Influential Keywords:**")
            st.json(explanation.as_list())

            st.write("---")
