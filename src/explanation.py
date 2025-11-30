from lime.lime_text import LimeTextExplainer
from utils import EMB_MODEL, embed_text
import numpy as np

explainer = LimeTextExplainer(class_names=["match"])

def make_predict_fn(jd_text):
    jd_vec = embed_text(jd_text)

    def predict(texts):
        embs = EMB_MODEL.encode(texts, convert_to_numpy=True)
        sims = np.dot(embs, jd_vec) / (np.linalg.norm(jd_vec) * np.linalg.norm(embs, axis=1))
        return np.vstack([sims, 1 - sims]).T

    return predict
