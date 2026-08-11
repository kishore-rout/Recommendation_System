"""
Flask Web UI for the Medicine Recommendation Engine
Supports 4 modes: Content-Based, Collaborative, Association Rules, Hybrid
"""
import os, json, pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── Load artefacts ──
clf = pickle.load(open(os.path.join(MODEL_DIR, "symptom_clf.pkl"), "rb"))
le = pickle.load(open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb"))
feature_cols = pickle.load(open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb"))
collab_data = pickle.load(open(os.path.join(MODEL_DIR, "collab_model.pkl"), "rb"))
assoc_data = pickle.load(open(os.path.join(MODEL_DIR, "assoc_rules.pkl"), "rb"))
meta_df = pd.read_csv(os.path.join(DATA_DIR, "conditions_meta.csv"))
metrics = json.load(open(os.path.join(MODEL_DIR, "metrics.json")))
    

# Get all unique medicines for the association rules UI selector
all_medicines = sorted(set(
    m for _, row in meta_df.iterrows() for m in row["medicines"].split("|")
))



# Import recommendation functions
import importlib.util
spec = importlib.util.spec_from_file_location("engine", os.path.join(BASE_DIR, "engine.py"))
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


@app.route("/")
def index():
    symptoms = [f.replace("_", " ").title() for f in feature_cols]
    user_ids = collab_data["user_ids"][:20]
    return render_template("index.html",
                           symptoms=list(zip(feature_cols, symptoms)),
                           user_ids=user_ids,
                           medicines=all_medicines,
                           metrics=metrics)


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    mode = data.get("mode", "content")
    selected = data.get("symptoms", [])
    user_id = data.get("user_id")
    selected_medicines = data.get("medicines", [])

    symptom_vec = {f: (1 if f in selected else 0) for f in feature_cols}

    if mode == "content":
        results = engine.recommend_content_based(symptom_vec, clf, le, feature_cols, meta_df)
        return jsonify({"mode": "content", "results": results})

    elif mode == "collaborative":
        uid = int(user_id) if user_id else 1
        results = engine.recommend_collaborative(uid, collab_data, top_n=5)
        return jsonify({"mode": "collaborative", "results": results})

    elif mode == "association":
        results = engine.recommend_association(selected_medicines, assoc_data, top_n=8)
        return jsonify({"mode": "association", "results": results})

    else:  # hybrid
        uid = int(user_id) if user_id else 1
        results = engine.recommend_hybrid(symptom_vec, uid, clf, le, feature_cols, meta_df, collab_data)
        return jsonify({"mode": "hybrid", "results": results})


@app.route("/analytics")
def analytics():
    return render_template("analytics.html", metrics=metrics)


if __name__ == "__main__":
    print("\n  MedRec Engine running at  http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)