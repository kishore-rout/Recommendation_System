"""
enginev2.py — Core ML Training & Recommendation Logic
=====================================================

PURPOSE:
    This is the SECOND file to run. It reads the 4 CSVs created by generate_data.py,
    trains 3 machine learning models, generates 8 visualisation charts, and saves
    everything to disk. It also contains the 4 recommendation functions that app.py
    calls at runtime when a user requests suggestions.

WHAT IT DOES (when run directly):
    [1/3] Trains a symptom → condition classifier (Random Forest + Gradient Boosting)
          Saves: symptom_clf.pkl, label_encoder.pkl, feature_cols.pkl
          Charts: confusion_matrix.png, feature_importance.png, model_comparison.png

    [2/3] Trains a collaborative filtering model (TruncatedSVD on user-medicine ratings)
          Saves: collab_model.pkl
          Charts: svd_variance.png, rating_distribution.png

    [3/3] Mines association rules from prescription transactions (pandas pair counting)
          Saves: assoc_rules.pkl
          Charts: association_rules_lift.png, association_scatter.png, item_frequency.png

    Final: Saves metrics.json with all evaluation numbers

HOW TO RUN:
    python engine.py

LIBRARIES USED:
    - sklearn.ensemble: RandomForestClassifier, GradientBoostingClassifier
    - sklearn.model_selection: train_test_split, cross_val_score
    - sklearn.metrics: classification_report, confusion_matrix
    - sklearn.decomposition: TruncatedSVD
    - sklearn.preprocessing: LabelEncoder
    - pandas: data manipulation, value_counts for association rules
    - numpy: array operations, sorting, random
    - matplotlib + seaborn: all 8 charts
    - itertools.combinations: generating medicine pairs from baskets
    - pickle + json: saving model artifacts
"""

import os, json, pickle
from itertools import combinations
import numpy as np
import pandas as pd

# ── sklearn imports: all the ML tools we need ──
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder

# ── Visualisation: matplotlib backend must be set BEFORE importing pyplot ──
# "Agg" = non-interactive backend that renders to PNG files (no GUI window needed)
# This is required for running on servers without a display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Directory paths ──
# BASE = the folder containing this engine.py file
# All other paths are relative to this base
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")       # where the CSVs live
MODELS = os.path.join(BASE, "models")   # where trained models are saved
STATIC = os.path.join(BASE, "static")   # where chart images are saved

# Create output directories if they don't exist yet
os.makedirs(MODELS, exist_ok=True)
os.makedirs(STATIC, exist_ok=True)


# ────────────────────────────────────────────────────────────────
# DATA LOADING
# ────────────────────────────────────────────────────────────────

def load_data():
    """
    Read all 4 CSV files into pandas DataFrames.

    Returns 4 DataFrames:
      - symptom_df:  2000 rows × 36 cols  (35 symptoms + 1 condition label)
      - ratings_df:  1600 rows × 3 cols   (user_id, medicine, rating)
      - meta_df:     10 rows × 4 cols     (condition, medicines, severity, symptoms)
      - trans_df:    500 rows × 4 cols    (transaction_id, medicines, conditions, basket_size)
    """
    symptom_df = pd.read_csv(os.path.join(DATA, "symptom_condition.csv"))
    ratings_df = pd.read_csv(os.path.join(DATA, "user_ratings.csv"))
    meta_df    = pd.read_csv(os.path.join(DATA, "conditions_meta.csv"))
    trans_df   = pd.read_csv(os.path.join(DATA, "prescription_transactions.csv"))
    return symptom_df, ratings_df, meta_df, trans_df


# ────────────────────────────────────────────────────────────────
# STRATEGY 1: CONTENT-BASED CLASSIFIER
# ────────────────────────────────────────────────────────────────
#
# CONCEPT:
#   Given a patient's symptoms (binary vector of 35 features),
#   predict which of 10 conditions they most likely have.
#   Then look up the medicines for that condition.
#
# ALGORITHM:
#   Random Forest — an ensemble of 150 decision trees that each learn
#   different symptom patterns. They "vote" to produce a final prediction
#   with probability estimates for each condition.
#
# WHY RANDOM FOREST:
#   - Handles binary features naturally (no scaling needed)
#   - Provides feature importances (which symptoms matter most)
#   - Gives probability estimates (not just a single prediction)
#   - Robust to noise and overfitting with proper max_depth
#

def train_symptom_classifier(symptom_df):
    """
    Train two classifiers (Random Forest and Gradient Boosting), compare them
    with 5-fold cross-validation, pick the winner, and save everything.

    INPUTS:
      symptom_df: DataFrame with 35 symptom columns + 1 "condition" column

    OUTPUTS (saved to models/):
      symptom_clf.pkl   — the winning trained classifier
      label_encoder.pkl — maps condition names to integers and back
      feature_cols.pkl  — ordered list of 35 symptom column names

    OUTPUTS (saved to static/):
      confusion_matrix.png   — shows prediction accuracy per condition
      feature_importance.png — shows which symptoms the model relies on most
      model_comparison.png   — compares RF vs GB accuracy

    RETURNS:
      (classifier, label_encoder, feature_columns, metrics_dict)
    """

    # ── STEP 1: Separate features (X) from labels (y) ──
    # feature_cols = all column names except "condition"
    # X = the 35-column binary matrix (each row is a patient's symptoms)
    # y = integer-encoded condition labels (0-9)
    feature_cols = [c for c in symptom_df.columns if c != "condition"]
    X = symptom_df[feature_cols].values  # shape: (2000, 35)

    # LabelEncoder converts string labels to integers:
    #   "Acid Reflux" → 0, "Allergies" → 1, ..., "Type 2 Diabetes" → 9
    # This is required because sklearn classifiers work with numbers, not strings.
    le = LabelEncoder()
    y = le.fit_transform(symptom_df["condition"].values)  # shape: (2000,)

    # ── STEP 2: Split into train (80%) and test (20%) sets ──
    # stratify=y ensures each condition has the same proportion in train and test
    # random_state=42 makes the split reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    # X_train: (1600, 35), X_test: (400, 35)

    # ── STEP 3: Train two classifiers ──

    # Random Forest: 150 decision trees, each up to 12 levels deep
    # n_jobs=-1 uses all CPU cores for parallel training
    rf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)

    # Gradient Boosting: 120 sequential trees, each up to 5 levels deep
    # Builds trees one at a time, each correcting the previous one's mistakes
    gb = GradientBoostingClassifier(n_estimators=120, max_depth=5, random_state=42)

    # .fit() is where the actual learning happens — the models examine the training data
    # and build internal decision rules
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    # ── STEP 4: Compare with 5-fold cross-validation ──
    # Cross-validation splits the FULL dataset into 5 parts ("folds").
    # For each fold: train on 4 parts, test on 1 part. Repeat 5 times.
    # This gives a more reliable accuracy estimate than a single train/test split.
    rf_cv = cross_val_score(rf, X, y, cv=5, scoring="accuracy")  # array of 5 accuracy values
    gb_cv = cross_val_score(gb, X, y, cv=5, scoring="accuracy")

    # ── STEP 5: Pick the winner ──
    # Whichever model has higher MEAN cross-validation accuracy wins
    best = rf if rf_cv.mean() >= gb_cv.mean() else gb
    best_name = "RandomForest" if rf_cv.mean() >= gb_cv.mean() else "GradientBoosting"

    # ── STEP 6: Evaluate the winner on the held-out test set ──
    y_pred = best.predict(X_test)

    # classification_report computes precision, recall, F1 for EACH condition:
    #   - Precision: of all patients predicted as "Migraine", how many actually had Migraine?
    #   - Recall: of all actual Migraine patients, how many did the model correctly identify?
    #   - F1: harmonic mean of precision and recall (single quality number)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

    # ── STEP 7: Generate visualisation charts ──

    # CHART 1: Confusion Matrix
    # A 10×10 grid showing: for each actual condition (row), how many patients
    # were predicted as each condition (column). Perfect classifier = all on diagonal.
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9); plt.yticks(fontsize=9)
    plt.tight_layout(); fig.savefig(os.path.join(STATIC, "confusion_matrix.png"), dpi=150); plt.close()

    # CHART 2: Feature Importance
    # Shows the top 15 symptoms ranked by how much the Random Forest relies on them.
    # "Gini importance" = how much each feature reduces prediction uncertainty across all trees.
    if hasattr(best, "feature_importances_"):
        imp = best.feature_importances_  # array of 35 importance values
        top_idx = np.argsort(imp)[-15:]  # indices of top 15 (ascending, so last = most important)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(top_idx)), imp[top_idx], color="#e85d04")
        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels([feature_cols[i].replace("_", " ").title() for i in top_idx])
        ax.set_xlabel("Importance"); ax.set_title("Top Symptom Feature Importances", fontweight="bold")
        plt.tight_layout(); fig.savefig(os.path.join(STATIC, "feature_importance.png"), dpi=150); plt.close()

    # CHART 3: Model Comparison
    # Bar chart comparing RF vs GB cross-validation accuracy with error bars
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Random Forest", "Gradient Boosting"],
                  [rf_cv.mean(), gb_cv.mean()], yerr=[rf_cv.std(), gb_cv.std()],
                  color=["#0077b6", "#e85d04"], capsize=5)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.1)
    ax.set_title("Model Comparison (5-Fold CV)", fontweight="bold")
    for bar, m in zip(bars, [rf_cv.mean(), gb_cv.mean()]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{m:.3f}", ha="center", fontweight="bold")
    plt.tight_layout(); fig.savefig(os.path.join(STATIC, "model_comparison.png"), dpi=150); plt.close()

    # ── STEP 8: Save trained artifacts to disk ──
    # pickle.dump serialises Python objects to binary files
    # These are loaded by app.py on startup so training only happens once
    pickle.dump(best, open(os.path.join(MODELS, "symptom_clf.pkl"), "wb"))
    pickle.dump(le, open(os.path.join(MODELS, "label_encoder.pkl"), "wb"))
    pickle.dump(feature_cols, open(os.path.join(MODELS, "feature_cols.pkl"), "wb"))

    # ── STEP 9: Collect metrics for the UI dashboard ──
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
# STRATEGY 2: COLLABORATIVE FILTERING
# ────────────────────────────────────────────────────────────────
#
# CONCEPT:
#   If User A and User B rated similar medicines similarly, then
#   medicines that User A liked (but User B hasn't tried) might
#   also be good for User B.
#
# ALGORITHM:
#   TruncatedSVD decomposes the user-medicine rating matrix into
#   "latent factors" — hidden patterns like "pain relief preference"
#   or "respiratory medicine affinity." Reconstructing the matrix from
#   these factors fills in the missing ratings with predictions.
#
# ANALOGY:
#   Netflix uses this exact approach: decompose the user-movie rating
#   matrix to predict what movies you'd like but haven't watched yet.
#   Here we do the same with users and medicines.
#

def train_collaborative_model(ratings_df):
    """
    Build user-item matrix, decompose with SVD, reconstruct predicted ratings.

    INPUTS:
      ratings_df: DataFrame with columns [user_id, medicine, rating]

    OUTPUTS (saved to models/):
      collab_model.pkl — dictionary containing the reconstructed rating matrix

    OUTPUTS (saved to static/):
      svd_variance.png      — how much information SVD captures
      rating_distribution.png — histogram of all ratings

    RETURNS:
      collab_data dictionary
    """

    # ── STEP 1: Build the user-medicine rating matrix ──
    # pivot_table transforms the long format (user_id, medicine, rating) into a 2D matrix:
    #   Rows = users (200), Columns = medicines (42), Values = ratings
    # fill_value=0 means "no rating" is treated as 0
    pivot = ratings_df.pivot_table(index="user_id", columns="medicine", values="rating", fill_value=0)
    user_ids = pivot.index.tolist()    # [1, 2, 3, ..., 200]
    med_names = pivot.columns.tolist() # ['Acetaminophen', 'Albuterol', ...]
    matrix = pivot.values              # shape: (200, 42) numpy array

    # ── STEP 2: Mean-centre the matrix ──
    # Each user has different rating tendencies (some rate everything high, some low).
    # Subtracting each user's mean removes this bias so SVD can focus on RELATIVE preferences.
    user_means = matrix.mean(axis=1, keepdims=True)  # shape: (200, 1)

    # ── STEP 3: Apply TruncatedSVD ──
    # SVD decomposes the centred matrix into: U × Σ × V^T
    # Where U = user preferences, Σ = importance weights, V = medicine characteristics
    # n_components=15 means we keep the 15 most important "latent factors"
    n_comp = min(15, min(matrix.shape) - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    user_factors = svd.fit_transform(matrix - user_means)  # shape: (200, 15)

    # ── STEP 4: Reconstruct the full matrix ──
    # Multiplying the factors back together + adding means gives predicted ratings
    # for ALL user-medicine pairs, including the ones we didn't have ratings for.
    # These predicted values are the recommendations.
    reconstructed = user_factors @ svd.components_ + user_means  # shape: (200, 42)

    # ── STEP 5: Generate charts ──

    # CHART: SVD Explained Variance
    # Shows what percentage of the total rating variation is captured by each component.
    # Higher = SVD is doing a better job of compressing the information.
    cumvar = np.cumsum(svd.explained_variance_ratio_) * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(cumvar)+1), cumvar, "o-", color="#0077b6", linewidth=2)
    ax.fill_between(range(1, len(cumvar)+1), cumvar, alpha=0.15, color="#0077b6")
    ax.set_xlabel("SVD Components"); ax.set_ylabel("Cumulative Variance (%)")
    ax.set_title("Collaborative Filtering — SVD", fontweight="bold"); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(os.path.join(STATIC, "svd_variance.png"), dpi=150); plt.close()

    # CHART: Rating Distribution
    # Histogram of all 1600 ratings — shows the overall shape of user preferences
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ratings_df["rating"], bins=20, color="#e85d04", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Rating"); ax.set_ylabel("Count")
    ax.set_title("User Rating Distribution", fontweight="bold")
    plt.tight_layout(); fig.savefig(os.path.join(STATIC, "rating_distribution.png"), dpi=150); plt.close()

    # ── STEP 6: Save to disk ──
    collab_data = {
        "user_ids": user_ids,           # list of 200 user IDs
        "med_names": med_names,         # list of 42 medicine names
        "reconstructed": reconstructed, # (200, 42) predicted rating matrix
        "user_means": user_means,       # (200, 1) per-user mean ratings
        "n_components": n_comp,         # 15
        "explained_var": round(cumvar[-1], 2),  # total variance explained (%)
    }
    pickle.dump(collab_data, open(os.path.join(MODELS, "collab_model.pkl"), "wb"))
    return collab_data


# ────────────────────────────────────────────────────────────────
# STRATEGY 3: ASSOCIATION RULES (using pandas — no custom algorithm)
# ────────────────────────────────────────────────────────────────
#
# CONCEPT:
#   If medicines A and B are frequently prescribed together, then a
#   patient already taking A might also benefit from B.
#
# APPROACH:
#   Instead of implementing the Apriori algorithm from scratch, we use
#   simple pandas operations to:
#     1. Count how often each individual medicine appears
#     2. Count how often each pair of medicines appears together
#     3. Compute support, confidence, and lift from the counts
#
# KEY METRICS:
#   - Support = P(A and B) = count(A,B) / total_transactions
#     "How common is this pair overall?"
#
#   - Confidence = P(B | A) = support(A,B) / support(A)
#     "If a patient takes A, what's the probability they also take B?"
#
#   - Lift = confidence / support(B)
#     "Is this association stronger than random chance?"
#     Lift > 1 means positive association. Lift = 1 means no association.
#     Lift < 1 means negative association (medicines appear together LESS than expected).
#

def mine_association_rules(trans_df, min_support=0.05, min_confidence=0.4):
    """
    Mine association rules using simple pandas counting.
    No Apriori implementation needed — just count pairs and compute metrics.

    INPUTS:
      trans_df: DataFrame with "medicines" column (pipe-delimited medicine lists)
      min_support: minimum frequency threshold (default 0.05 = 5% of transactions)
      min_confidence: minimum reliability threshold (default 0.4 = 40%)

    RETURNS:
      (rules_list, item_counts, frequent_pairs)
    """

    # ── STEP 1: Parse each transaction into a Python set of medicine names ──
    # "Ibuprofen|Dextromethorphan|Vitamin C" → {"Ibuprofen", "Dextromethorphan", "Vitamin C"}
    baskets = [set(row.split("|")) for row in trans_df["medicines"]]
    n = len(baskets)  # 500 transactions

    # ── STEP 2: Count individual item frequencies using pandas ──
    # Flatten all baskets into one big list, then use value_counts()
    # This gives us: Ibuprofen → 180, Acetaminophen → 165, etc.
    all_items = []
    for b in baskets:
        all_items.extend(b)
    item_counts = pd.Series(all_items).value_counts()

    # Convert counts to support (fraction of total transactions)
    # support("Ibuprofen") = 180/500 = 0.36
    item_support = item_counts / n

    # ── STEP 3: Count all 2-item pair frequencies using pandas ──
    # For each basket, generate all possible pairs using itertools.combinations
    # sorted() ensures ("A","B") and ("B","A") become the same tuple ("A","B")
    pair_list = []
    for b in baskets:
        for pair in combinations(sorted(b), 2):
            pair_list.append(pair)

    # value_counts() counts how often each pair appears across all transactions
    pair_counts = pd.Series(pair_list).value_counts()
    pair_support = pair_counts / n  # convert to support

    # ── STEP 4: Filter pairs by minimum support ──
    # Only keep pairs that appear in at least 5% of transactions
    freq_pairs = pair_support[pair_support >= min_support]

    # ── STEP 5: Generate rules from frequent pairs ──
    # Each pair (A, B) generates TWO rules: A→B and B→A
    # (because confidence is directional: P(B|A) ≠ P(A|B))
    rules = []
    for (item_a, item_b), sup in freq_pairs.items():
        sup_a = item_support.get(item_a, 0)
        sup_b = item_support.get(item_b, 0)

        # Rule 1: A → B
        # "If patient takes A, recommend B"
        if sup_a > 0:
            conf_ab = sup / sup_a                      # confidence = P(B|A)
            lift_ab = conf_ab / sup_b if sup_b > 0 else 0  # lift = confidence / P(B)
            if conf_ab >= min_confidence:
                rules.append({
                    "antecedent": [item_a],            # the IF part
                    "consequent": [item_b],            # the THEN part
                    "support": round(sup, 4),
                    "confidence": round(conf_ab, 4),
                    "lift": round(lift_ab, 4),
                })

        # Rule 2: B → A
        # "If patient takes B, recommend A"
        if sup_b > 0:
            conf_ba = sup / sup_b
            lift_ba = conf_ba / sup_a if sup_a > 0 else 0
            if conf_ba >= min_confidence:
                rules.append({
                    "antecedent": [item_b],
                    "consequent": [item_a],
                    "support": round(sup, 4),
                    "confidence": round(conf_ba, 4),
                    "lift": round(lift_ba, 4),
                })

    # Sort by lift descending — strongest associations first
    rules.sort(key=lambda r: (-r["lift"], -r["confidence"]))
    return rules, item_counts, freq_pairs


def train_association_rules(trans_df, min_support=0.05, min_confidence=0.4):
    """
    Full pipeline: count pairs → generate rules → save charts + artifacts.

    INPUTS:
      trans_df: prescription transactions DataFrame

    OUTPUTS (saved to models/):
      assoc_rules.pkl — dictionary containing all rules + metadata

    OUTPUTS (saved to static/):
      association_rules_lift.png — top 15 rules by lift
      association_scatter.png   — support vs confidence scatter plot
      item_frequency.png        — most common medicines in prescriptions

    RETURNS:
      assoc_data dictionary
    """
    rules, item_counts, freq_pairs = mine_association_rules(trans_df, min_support, min_confidence)
    n = len(trans_df)

    print(f"    Transactions: {n}, unique items: {len(item_counts)}")
    print(f"    Frequent pairs (support >= {min_support}): {len(freq_pairs)}")
    print(f"    Association rules (confidence >= {min_confidence}): {len(rules)}")

    # ── CHART 1: Top Rules by Lift ──
    # Shows the strongest co-prescription patterns.
    # Orange bars = lift > 2 (strong association). Blue = moderate.
    if rules:
        top = rules[:15]
        labels = [f"{', '.join(r['antecedent'])} -> {', '.join(r['consequent'])}" for r in top]
        lifts = [r["lift"] for r in top]
        confs = [r["confidence"] for r in top]

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(range(len(labels)), lifts,
                       color=["#e85d04" if l > 2 else "#0077b6" for l in lifts], alpha=0.85)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Lift"); ax.set_title("Top Association Rules by Lift", fontweight="bold")
        ax.invert_yaxis()
        for i, (bar, conf) in enumerate(zip(bars, confs)):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    f"conf={conf:.0%}", va="center", fontsize=7, color="#666")
        plt.tight_layout(); fig.savefig(os.path.join(STATIC, "association_rules_lift.png"), dpi=150); plt.close()

    # ── CHART 2: Support vs Confidence scatter ──
    # Each dot is one rule. X = support, Y = confidence, colour = lift.
    # Rules in the top-right corner are the best (common AND reliable).
    if len(rules) > 3:
        fig, ax = plt.subplots(figsize=(8, 6))
        s = ax.scatter([r["support"] for r in rules], [r["confidence"] for r in rules],
                       c=[r["lift"] for r in rules], cmap="YlOrRd", s=60, alpha=0.8, edgecolors="black", linewidth=0.3)
        plt.colorbar(s, ax=ax, label="Lift")
        ax.set_xlabel("Support"); ax.set_ylabel("Confidence")
        ax.set_title("Association Rules — Support vs Confidence", fontweight="bold"); ax.grid(alpha=0.2)
        plt.tight_layout(); fig.savefig(os.path.join(STATIC, "association_scatter.png"), dpi=150); plt.close()

    # ── CHART 3: Item Frequency ──
    # Bar chart showing the 20 most common medicines across all transactions.
    top_items = item_counts.head(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(top_items)), top_items.values, color="#0077b6", edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(top_items)))
    ax.set_xticklabels(top_items.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Frequency"); ax.set_title("Medicine Frequency in Prescriptions", fontweight="bold")
    plt.tight_layout(); fig.savefig(os.path.join(STATIC, "item_frequency.png"), dpi=150); plt.close()

    # ── Save to disk ──
    assoc_data = {
        "rules": rules,                    # list of rule dictionaries
        "n_transactions": n,               # 500
        "n_freq_pairs": len(freq_pairs),   # number of frequent pairs found
        "n_rules": len(rules),             # number of rules above confidence threshold
        "min_support": min_support,        # threshold used
        "min_confidence": min_confidence,  # threshold used
    }
    pickle.dump(assoc_data, open(os.path.join(MODELS, "assoc_rules.pkl"), "wb"))
    return assoc_data


# ────────────────────────────────────────────────────────────────
# RECOMMENDATION FUNCTIONS
# ────────────────────────────────────────────────────────────────
#
# These 4 functions are called by app.py at runtime (NOT during training).
# They take user input and return ranked medicine suggestions.
#

def recommend_content_based(symptoms_dict, clf, le, feature_cols, meta_df, top_n=5):
    """
    Content-Based: Symptoms → predict condition → lookup medicines.

    FLOW:
      1. Convert selected symptoms into a binary vector [0,1,0,1,1,0,...]
      2. Feed it through the trained classifier
      3. Get probability distribution over 10 conditions
      4. Take top 3 conditions by probability
      5. For each, look up medicines from conditions_meta.csv

    INPUTS:
      symptoms_dict: {symptom_name: 0 or 1} for all 35 symptoms
      clf: trained classifier (loaded from symptom_clf.pkl)
      le: label encoder (loaded from label_encoder.pkl)
      feature_cols: ordered list of 35 symptom names
      meta_df: conditions_meta.csv DataFrame

    RETURNS:
      list of dicts: [{condition, confidence, severity, medicines}, ...]
    """
    # Build the input vector in the EXACT order the classifier expects
    vec = np.array([[symptoms_dict.get(f, 0) for f in feature_cols]])  # shape: (1, 35)

    # predict_proba returns probabilities for each class
    # e.g., [0.02, 0.01, 0.03, 0.01, 0.01, 0.75, 0.05, 0.02, 0.08, 0.02]
    proba = clf.predict_proba(vec)[0]

    # Get indices of top 3 conditions (sorted descending by probability)
    top_idx = np.argsort(proba)[::-1][:3]

    results = []
    for idx in top_idx:
        cond = le.classes_[idx]  # convert integer back to condition name
        matched = meta_df[meta_df["condition"] == cond]
        if matched.empty:
            # Safety: skip if condition not found in meta table
            continue
        row = matched.iloc[0]  # lookup in meta table
        results.append({
            "condition": cond,
            "confidence": round(float(proba[idx]), 3),
            "severity": row["severity"],
            "medicines": row["medicines"].split("|")[:top_n],
        })
    return results


def recommend_collaborative(user_id, collab_data, top_n=5):
    """
    Collaborative: User ID → predicted ratings → top medicines.

    FLOW:
      1. Find this user's row in the reconstructed rating matrix
      2. Sort all 42 medicines by predicted rating (descending)
      3. Return top N

    INPUTS:
      user_id: integer user ID (1-200)
      collab_data: dictionary loaded from collab_model.pkl

    RETURNS:
      list of dicts: [{medicine, predicted_rating}, ...]
    """
    if user_id not in collab_data["user_ids"]:
        return []
    uid_idx = collab_data["user_ids"].index(user_id)       # row index in matrix
    scores = collab_data["reconstructed"][uid_idx]          # 42 predicted ratings
    top_idx = np.argsort(scores)[::-1][:top_n]             # indices of top N
    return [{"medicine": collab_data["med_names"][i],
             "predicted_rating": round(float(scores[i]), 2)} for i in top_idx]


def recommend_association(selected_medicines, assoc_data, top_n=5):
    """
    Association Rules: Current medicines → matching rules → co-prescribed suggestions.

    FLOW:
      1. Take the set of medicines the user already takes
      2. Loop through all rules
      3. If a rule's antecedent (IF part) is a subset of what the user takes, the rule "fires"
      4. The consequent (THEN part) becomes a recommendation candidate
      5. Score each candidate by lift × confidence
      6. Return top N unique recommendations

    INPUTS:
      selected_medicines: list of medicine names the user currently takes
      assoc_data: dictionary loaded from assoc_rules.pkl

    RETURNS:
      list of dicts: [{medicine, score, support, confidence, lift, triggered_by}, ...]
    """
    selected = set(selected_medicines)
    if not selected or not assoc_data["rules"]:
        return []

    scored = {}
    for rule in assoc_data["rules"]:
        ante = set(rule["antecedent"])

        # Does this rule's IF-part match what the user is taking?
        if ante.issubset(selected):
            for med in rule["consequent"]:
                # Don't recommend medicines the user already takes
                if med not in selected:
                    score = rule["lift"] * rule["confidence"]
                    # Keep the highest-scoring rule for each medicine
                    if med not in scored or score > scored[med]["score"]:
                        scored[med] = {
                            "medicine": med, "score": round(score, 3),
                            "support": rule["support"], "confidence": rule["confidence"],
                            "lift": rule["lift"], "triggered_by": rule["antecedent"],
                        }

    # Sort by score descending and return top N
    return sorted(scored.values(), key=lambda x: -x["score"])[:top_n]


def recommend_hybrid(symptoms_dict, user_id, clf, le, feature_cols, meta_df, collab_data,
                     top_n=5, content_weight=0.6):
    """
    Hybrid: Blend content-based + collaborative scores.

    FLOW:
      1. Run content-based → get condition predictions + medicines
      2. Run collaborative → get predicted ratings
      3. Normalise both sets of scores to a common scale
      4. Blend: final_score = 0.6 × content_score + 0.4 × collab_score
      5. Sort by blended score

    WHY BLEND:
      Content-based is strong when symptoms are clear but weak for new conditions.
      Collaborative is strong when user history exists but weak for new users.
      Blending combines their strengths.

    INPUTS:
      symptoms_dict: binary symptom vector
      user_id: integer user ID
      content_weight: how much to weight content-based (default 0.6 = 60%)

    RETURNS:
      list of dicts: [{medicine, hybrid_score}, ...]
    """
    # Get results from both strategies (ask for more than top_n so we have overlap candidates)
    cb = recommend_content_based(symptoms_dict, clf, le, feature_cols, meta_df, top_n=20)
    cf = recommend_collaborative(user_id, collab_data, top_n=20)

    # Score each medicine from content-based results
    # Higher confidence condition + higher position in medicine list = higher score
    scores = {}
    for r in cb:
        for i, med in enumerate(r["medicines"]):
            # (1 - i*0.1) gives position decay: 1st med = 1.0, 2nd = 0.9, 3rd = 0.8...
            scores[med] = scores.get(med, 0) + r["confidence"] * (1 - i*0.1) * content_weight

    # Add collaborative scores (normalised to 0-1 range)
    if cf:
        max_r = max(r["predicted_rating"] for r in cf) or 1
        for r in cf:
            scores[r["medicine"]] = scores.get(r["medicine"], 0) + (r["predicted_rating"]/max_r) * (1-content_weight)

    # Sort by total blended score
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    return [{"medicine": m, "hybrid_score": round(s, 3)} for m, s in ranked]


# ────────────────────────────────────────────────────────────────
# MAIN: TRAIN ALL MODELS (runs when you execute: python engine.py)
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MEDICINE RECOMMENDATION ENGINE — TRAINING")
    print("=" * 60)

    # Load all 4 datasets
    symptom_df, ratings_df, meta_df, trans_df = load_data()

    # Train Strategy 1: Content-Based Classifier
    print("\n[1/3] Training classifier ...")
    clf, le, feature_cols, metrics = train_symptom_classifier(symptom_df)
    print(f"      Best: {metrics['best_model']}  Acc: {metrics['test_accuracy']}")
    print(f"      RF CV: {metrics['rf_cv_mean']}  GB CV: {metrics['gb_cv_mean']}")

    # Train Strategy 2: Collaborative Filtering
    print("\n[2/3] Training collaborative filtering ...")
    collab = train_collaborative_model(ratings_df)
    print(f"      SVD components: {collab['n_components']}  Variance: {collab['explained_var']}%")

    # Train Strategy 3: Association Rules
    print("\n[3/3] Mining association rules ...")
    assoc = train_association_rules(trans_df)

    # Save combined metrics (used by the analytics dashboard)
    metrics["assoc"] = {
        "n_transactions": assoc["n_transactions"],
        "n_freq_pairs": assoc["n_freq_pairs"],
        "n_rules": assoc["n_rules"],
        "min_support": assoc["min_support"],
        "min_confidence": assoc["min_confidence"],
    }
    json.dump(metrics, open(os.path.join(MODELS, "metrics.json"), "w"), indent=2)

    # ── Demo: test all 4 recommendation functions ──
    print("\n" + "-" * 60)

    # Build a test symptom vector (Migraine symptoms)
    test_sym = {f: 0 for f in feature_cols}
    test_sym["headache"] = 1; test_sym["nausea"] = 1; test_sym["light_sensitivity"] = 1

    print("  Content-Based (headache + nausea + light_sensitivity):")
    for r in recommend_content_based(test_sym, clf, le, feature_cols, meta_df):
        print(f"    {r['condition']} ({r['confidence']:.0%}) -> {', '.join(r['medicines'])}")

    print("  Collaborative (user 1):")
    for r in recommend_collaborative(1, collab):
        print(f"    {r['medicine']} — rating {r['predicted_rating']}")

    print("  Association (taking: Ibuprofen, Dextromethorphan):")
    for r in recommend_association(["Ibuprofen", "Dextromethorphan"], assoc):
        print(f"    {r['medicine']} — score {r['score']} (lift={r['lift']})")

    print("  Hybrid (user 1 + symptoms):")
    for r in recommend_hybrid(test_sym, 1, clf, le, feature_cols, meta_df, collab):
        print(f"    {r['medicine']} — score {r['hybrid_score']}")

    print("\n" + "=" * 60)
    print("  Done. Run: python app.py")
    print("=" * 60)