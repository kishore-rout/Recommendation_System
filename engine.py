"""
Medicine Recommendation Engine (Detailed Version)
================================================
This script implements a multi-strategy recommendation engine for pharmaceuticals.
It utilizes four distinct methodologies to suggest medicines based on symptoms,
user history, and transactional patterns.

STRATEGIES:
1. Content-Based: Predicts health conditions from symptoms using ensemble classifiers.
2. Collaborative Filtering: Uses Matrix Factorization (SVD) to predict user-item ratings.
3. Association Rules: Discovers co-prescription patterns using Pandas-based frequency counting.
4. Hybrid: A weighted rank-fusion of content and collaborative scores.


"""

import os
import json
import pickle
from itertools import combinations
import numpy as np
import pandas as pd

# Machine Learning & Evaluation
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder

# Visualization - Using 'Agg' backend to ensure compatibility in headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Path Configurations
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")
STATIC = os.path.join(BASE, "static")

# Ensure directory structure exists
os.makedirs(MODELS, exist_ok=True)
os.makedirs(STATIC, exist_ok=True)


# ────────────────────────────────────────────────────────────────
# DATA LOADING
# ────────────────────────────────────────────────────────────────

def load_data():
    """
    Loads the required CSV datasets.
    Expected Files:
    - symptom_condition.csv: Binary/scalar flags of symptoms mapped to conditions.
    - user_ratings.csv: Historical medicine ratings by users (User, Item, Rating).
    - conditions_meta.csv: Mapping of conditions to standard treatments/severity.
    - prescription_transactions.csv: Historical co-prescriptions (baskets).
    """
    symptom_df = pd.read_csv(os.path.join(DATA, "symptom_condition.csv"))
    ratings_df = pd.read_csv(os.path.join(DATA, "user_ratings.csv"))
    meta_df    = pd.read_csv(os.path.join(DATA, "conditions_meta.csv"))
    trans_df   = pd.read_csv(os.path.join(DATA, "prescription_transactions.csv"))
    return symptom_df, ratings_df, meta_df, trans_df


# ────────────────────────────────────────────────────────────────
# 1. CONTENT-BASED CLASSIFIER (Supervised Learning)
# ────────────────────────────────────────────────────────────────

def train_symptom_classifier(symptom_df):
    """
    Trains a classifier to map symptoms (features) to medical conditions (labels).
    
    Logic:
    1. Preprocessing: Encodes text conditions into numeric integers using LabelEncoder.
    2. Model Selection: Compares Random Forest vs. Gradient Boosting via 5-Fold Cross Validation.
    3. Evaluation: Generates a confusion matrix and feature importance plot.
    """

    # Feature engineering: All columns except 'condition' are treated as symptom inputs
    feature_cols = [c for c in symptom_df.columns if c != "condition"]
    X = symptom_df[feature_cols].values
    
    le = LabelEncoder()
    y = le.fit_transform(symptom_df["condition"].values)

    # Stratified split ensures class proportions are maintained in train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialization of ensemble models
    rf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=120, max_depth=5, random_state=42)
    
    # Fitting models
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    # Performance Validation
    rf_cv = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
    gb_cv = cross_val_score(gb, X, y, cv=5, scoring="accuracy")

    # Winner selection based on mean Cross-Validation accuracy
    best = rf if rf_cv.mean() >= gb_cv.mean() else gb
    best_name = "RandomForest" if rf_cv.mean() >= gb_cv.mean() else "GradientBoosting"

    y_pred = best.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

    # --- Visualizations ---
    
    # Confusion Matrix: Helps identify which conditions are being confused with others
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Condition Classification: Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9); plt.yticks(fontsize=9)
    plt.tight_layout(); fig.savefig(os.path.join(STATIC, "confusion_matrix.png"), dpi=150); plt.close()

    # Feature Importance: Identifies which symptoms are the strongest predictors
    if hasattr(best, "feature_importances_"):
        imp = best.feature_importances_
        top_idx = np.argsort(imp)[-15:]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(top_idx)), imp[top_idx], color="#e85d04")
        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels([feature_cols[i].replace("_", " ").title() for i in top_idx])
        ax.set_xlabel("Importance Weight"); ax.set_title("Top Predictor Symptoms", fontweight="bold")
        plt.tight_layout(); fig.savefig(os.path.join(STATIC, "feature_importance.png"), dpi=150); plt.close()

    # Serialization: Saving model artifacts for production use
    pickle.dump(best, open(os.path.join(MODELS, "symptom_clf.pkl"), "wb"))
    pickle.dump(le, open(os.path.join(MODELS, "label_encoder.pkl"), "wb"))
    pickle.dump(feature_cols, open(os.path.join(MODELS, "feature_cols.pkl"), "wb"))

    metrics = {
        "best_model": best_name,
        "rf_cv_mean": round(rf_cv.mean(), 4),
        "gb_cv_mean": round(gb_cv.mean(), 4),
        "test_accuracy": round(report["accuracy"], 4),
        "per_class": {k: {"precision": round(v["precision"], 3),
                          "recall": round(v["recall"], 3),
                          "f1": round(v["f1-score"], 3)}
                      for k, v in report.items() if k in le.classes_},
    }
    return best, le, feature_cols, metrics


# ────────────────────────────────────────────────────────────────
# 2. COLLABORATIVE FILTERING (Matrix Factorization)
# ────────────────────────────────────────────────────────────────

def train_collaborative_model(ratings_df):
    """
    Implements a Latent Factor model using Truncated Singular Value Decomposition (SVD).
    
    Code implements Market Basket Analysis
    
    Mathematical Process:
    1. Pivot: Create a User-Item matrix (R).
    2. De-mean: Subtract the average rating of each user to handle 'optimistic' vs 'pessimistic' raters.
    3. Decompose: R ≈ U * Sigma * Vt. We compress the matrix into 'n_components' latent factors.
    4. Reconstruct: Rebuild the matrix to fill in the zeros (predicted ratings).
    
    This code scans thousands of prescriptions to find "best friends" among medicines
    
    """

    # Create User-Medicine sparse matrix
    pivot = ratings_df.pivot_table(index="user_id", columns="medicine", values="rating", fill_value=0)
    user_ids = pivot.index.tolist()
    med_names = pivot.columns.tolist()
    matrix = pivot.values

    # Mean-centering: Vital for SVD to focus on relative preferences rather than absolute rating scales
    user_means = matrix.mean(axis=1, keepdims=True)
    
    # Components define the number of 'latent traits' to learn (e.g., taste in brands or side-effect tolerance)
    n_comp = min(15, min(matrix.shape) - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    
    # Decompose the centered matrix
    user_factors = svd.fit_transform(matrix - user_means)
    
    # Reconstruct the matrix: The dot product of factors + adding back the user means
    reconstructed = user_factors @ svd.components_ + user_means

    # Explained Variance Plot: Shows how much information is retained as we increase complexity
    cumvar = np.cumsum(svd.explained_variance_ratio_) * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    
    ax.plot(range(1, len(cumvar)+1), cumvar, "o-", color="#0077b6", linewidth=2)
    ax.fill_between(range(1, len(cumvar)+1), cumvar, alpha=0.15, color="#0077b6")
    ax.set_xlabel("Latent Components"); ax.set_ylabel("Cumulative Variance (%)")
    ax.set_title("Collaborative Filtering: Info Retention", fontweight="bold"); ax.grid(alpha=0.3)
    
    plt.tight_layout(); fig.savefig(os.path.join(STATIC, "svd_variance.png"), dpi=150); plt.close()

    collab_data = {
        "user_ids": user_ids, "med_names": med_names,
        "reconstructed": reconstructed, "user_means": user_means,
        "n_components": n_comp, "explained_var": round(cumvar[-1], 2),
    }
    pickle.dump(collab_data, open(os.path.join(MODELS, "collab_model.pkl"), "wb"))
    return collab_data


# ────────────────────────────────────────────────────────────────
# 3. ASSOCIATION RULES (Market Basket Analysis)
# ────────────────────────────────────────────────────────────────

def mine_association_rules(trans_df, min_support=0.05, min_confidence=0.4):
    
    '''
    Step 1: Tokenize         → Turn CSV strings into Python sets 
    Step 2: Count singles    → How often does each medicine appear?
    Step 3: Count pairs      → How often do pairs appear together?
    Step 4: Filter pairs     → Keep only pairs above 5% support
    Step 5: Compute metrics  → Calculate confidence and lift for each pair
    '''
    # STEP 1: PREPARATION
    # We turn the string "MedA|MedB" into a Python Set {'MedA', 'MedB'}
    # This makes checking for the presence of a medicine 10x faster.
    baskets = [set(row.split("|")) for row in trans_df["medicines"]]
    n = len(baskets) # Total number of historical prescriptions

    # STEP 2: CALCULATE POPULARITY (Support for single items)
    all_items = []
    for b in baskets:
        all_items.extend(b)
    
    # We count how many times each medicine appears across ALL records
    item_counts = pd.Series(all_items).value_counts()
    # item_support = (How many times A appeared) / (Total prescriptions)
    item_support = item_counts / n 

    # STEP 3: FIND CO-OCCURRENCES (Pair counting)
    pair_list = []
    for b in baskets:
        # combinations() generates every possible pair in one prescription
        # e.g., if a patient has [A, B, C], it creates (A,B), (A,C), and (B,C)
        for pair in combinations(sorted(b), 2):
            pair_list.append(pair)
    
    # Count how many times these specific pairs appeared together
    pair_counts = pd.Series(pair_list).value_counts()
    pair_support = pair_counts / n 

    # STEP 4: FILTERING
    # We discard any pair that doesn't happen at least 5% of the time (min_support)
    # This removes noise and rare, non-significant coincidences.
    freq_pairs = pair_support[pair_support >= min_support]

    # STEP 5: GENERATE RULES & METRICS
    rules = []
    for (item_a, item_b), sup in freq_pairs.items():
        # Get baseline support for the individual medicines
        sup_a = item_support.get(item_a, 0)
        sup_b = item_support.get(item_b, 0)

        # We calculate the rule in both directions:
        # Direction 1: If A then B
        if sup_a > 0:
            # CONFIDENCE: Out of all people taking A, how many also took B?
            conf_ab = sup / sup_a
            # LIFT: Is the pairing better than random chance?
            lift_ab = conf_ab / sup_b if sup_b > 0 else 0
            
            if conf_ab >= min_confidence:
                rules.append({
                    "antecedent": [item_a], "consequent": [item_b],
                    "support": round(sup, 4), "confidence": round(conf_ab, 4),
                    "lift": round(lift_ab, 4),
                })

        # Direction 2: If B then A (Same logic, flipped)
        if sup_b > 0:
            conf_ba = sup / sup_b
            lift_ba = conf_ba / sup_a if sup_a > 0 else 0
            if conf_ba >= min_confidence:
                rules.append({
                    "antecedent": [item_b], "consequent": [item_a],
                    "support": round(sup, 4), "confidence": round(conf_ba, 4),
                    "lift": round(lift_ba, 4),
                })

    # FINAL SORT: We prioritize LIFT (significance) then CONFIDENCE (reliability)
    rules.sort(key=lambda r: (-r["lift"], -r["confidence"]))
    return rules, item_counts, freq_pairs



def train_association_rules(trans_df, min_support=0.05, min_confidence=0.4):
    """Execution wrapper for Association Rule Mining including visualization."""
    rules, item_counts, freq_pairs = mine_association_rules(trans_df, min_support, min_confidence)
    n = len(trans_df)

    # Chart: Top Rules by Lift
    if rules:
        top = rules[:15]
        labels = [f"{', '.join(r['antecedent'])} -> {', '.join(r['consequent'])}" for r in top]
        lifts = [r["lift"] for r in top]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(range(len(labels)), lifts, color="#0077b6")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Lift (Significance)"); ax.set_title("Medication Co-Prescription Strengths", fontweight="bold")
        ax.invert_yaxis()
        plt.tight_layout(); fig.savefig(os.path.join(STATIC, "association_rules_lift.png"), dpi=150); plt.close()

    assoc_data = {
        "rules": rules, "n_transactions": n,
        "n_freq_pairs": len(freq_pairs), "n_rules": len(rules),
        "min_support": min_support, "min_confidence": min_confidence,
    }
    pickle.dump(assoc_data, open(os.path.join(MODELS, "assoc_rules.pkl"), "wb"))
    return assoc_data


# ────────────────────────────────────────────────────────────────
# 4. RECOMMENDATION LOGIC (Inference)
# ────────────────────────────────────────────────────────────────

def recommend_content_based(symptoms_dict, clf, le, feature_cols, meta_df, top_n=5):
    """Predicts a condition, then looks up best-fit medicines from the metadata."""
    # Convert input dict to vector aligned with training feature columns
    vec = np.array([[symptoms_dict.get(f, 0) for f in feature_cols]])
    
    # Get probability distribution across all known conditions
    proba = clf.predict_proba(vec)[0]
    top_idx = np.argsort(proba)[::-1][:3] # Get top 3 likely conditions
    
    results = []
    for idx in top_idx:
        cond = le.classes_[idx]
        row = meta_df[meta_df["condition"] == cond].iloc[0]
        results.append({
            "condition": cond,
            "confidence": round(float(proba[idx]), 3),
            "severity": row["severity"],
            "medicines": row["medicines"].split("|")[:top_n],
        })
    return results


def recommend_collaborative(user_id, collab_data, top_n=5):
    """Retrieves pre-calculated predicted ratings for a specific user."""
    if user_id not in collab_data["user_ids"]:
        return []
    
    '''
        import numpy as np
    
        # Example collaborative filtering output
        collab_data = {
          "user_ids": ["u1", "u2", "u3"],
          "reconstructed": np.array([
              [0.2, 0.9, 0.1, 0.7],   # Scores for user u1
              [0.8, 0.3, 0.4, 0.6],   # Scores for user u2
              [0.5, 0.1, 0.9, 0.2]    # Scores for user u3
              ])
          }
            
        user_id = "u1"
        top_n = 2
        item_names = ["Item_A", "Item_B", "Item_C", "Item_D"]
            
        # Get the row index for the given user
        uid_idx = collab_data["user_ids"].index(user_id)
            
        # Get predicted scores for all items for this user
        scores = collab_data["reconstructed"][uid_idx]
            
        # Get indices of top-N highest scoring items
        top_idx = np.argsort(scores)[::-1][:top_n]
            
        # Map indices to actual item names
        top_items = [item_names[i] for i in top_idx]
            
        print("User:", user_id)
        print("Scores:", scores)
        print("Top indices:", top_idx)
        print("Recommended items:", top_items)
        
        top_idx = [1, 3, 4]

        top_idx = np.argsort(scores)[::-1][:5]
        recommended_items = item_names[top_idx]

    '''
    
    uid_idx = collab_data["user_ids"].index(user_id)
    scores = collab_data["reconstructed"][uid_idx]
    
    top_idx = np.argsort(scores)[::-1][:top_n]
    return [{"medicine": collab_data["med_names"][i],
             "predicted_rating": round(float(scores[i]), 2)} for i in top_idx]


def recommend_association(selected_medicines, assoc_data, top_n=5):
    """Current medicines -> matching rules -> co-prescribed suggestions."""
    selected = set(selected_medicines)
    if not selected or not assoc_data["rules"]:
        return []
    scored = {}
    for rule in assoc_data["rules"]:
        ante = set(rule["antecedent"])
        if ante.issubset(selected):
            for med in rule["consequent"]:
                if med not in selected:
                    score = rule["lift"] * rule["confidence"]
                    if med not in scored or score > scored[med]["score"]:
                        scored[med] = {
                            "medicine": med, "score": round(score, 3),
                            "support": rule["support"], "confidence": rule["confidence"],
                            "lift": rule["lift"], "triggered_by": rule["antecedent"],
                        }
    return sorted(scored.values(), key=lambda x: -x["score"])[:top_n]




def recommend_hybrid(symptoms_dict, user_id, clf, le, feature_cols, meta_df, collab_data,
                     top_n=5, content_weight=0.6):
    """
    Combines Content (Expert knowledge) with Collaborative (Crowd wisdom).
    
    Weights:
    - Content Weight: How much we trust the symptom classifier (default 0.6).
    - Collaborative Weight: How much we trust user ratings (1 - content_weight).
    """
    cb_results = recommend_content_based(symptoms_dict, clf, le, feature_cols, meta_df, top_n=20)
    cf_results = recommend_collaborative(user_id, collab_data, top_n=20)

    # Scoring accumulator
    scores = {}
    
    # Content scoring (Confidence * Rank Discount)
    for r in cb_results:
        for i, med in enumerate(r["medicines"]):
            # Position in list decreases score slightly to prioritize 'primary' medicines
            scores[med] = scores.get(med, 0) + r["confidence"] * (1 - i*0.1) * content_weight
            
    # Collaborative scoring (Normalized Predicted Rating)
    if cf_results:
        max_rating = max(r["predicted_rating"] for r in cf_results) or 1
        for r in cf_results:
            scores[r["medicine"]] = scores.get(r["medicine"], 0) + (r["predicted_rating"]/max_rating) * (1-content_weight)

    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    return [{"medicine": m, "hybrid_score": round(s, 3)} for m, s in ranked]


# ────────────────────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Training orchestration logic...
    symptom_df, ratings_df, meta_df, trans_df = load_data()
    
    print("Training Recommendation Models...")
    clf, le, feature_cols, metrics = train_symptom_classifier(symptom_df)
    collab = train_collaborative_model(ratings_df)
    assoc = train_association_rules(trans_df)
    
    print("Training Complete. Metrics saved to models/metrics.json")