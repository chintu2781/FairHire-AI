from utils import load_resume_text, score_resume, recommend_actions
import numpy as np
import re

def rank_resumes(resume_files, jd_text):
    results = []

    for path in resume_files:
        text = load_resume_text(path)
        score = score_resume(text, jd_text)
        recs = recommend_actions(text, jd_text)

        results.append({
            "resume": path,
            "text": text,
            "score": score,
            "recommendations": recs
        })

    scores = np.array([r["score"] for r in results])
    normalized = 100 * (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

    for i, r in enumerate(results):
        r["final_score"] = round(float(normalized[i]), 2)

    return sorted(results, key=lambda x: x["final_score"], reverse=True)

def bias_check(text, jd_text):
    sensitive_words = [
        "male","female","mumbai","delhi","bangalore",
        "iit","nit","iiit"
    ]

    base = score_resume(text, jd_text)
    masked = text

    for w in sensitive_words:
        masked = re.sub(w, "[MASKED]", masked, flags=re.I)

    new = score_resume(masked, jd_text)

    return {
        "base_score": base,
        "masked_score": new,
        "change": new - base
    }
