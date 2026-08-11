"""
demo_recommendations.py — Step-by-Step Demonstration of All 4 Recommendation Functions
========================================================================================

PURPOSE:
    This file demonstrates each of the 4 recommendation functions from engine.py
    with REAL sample inputs, intermediate print statements at every step, and
    annotated outputs. It is designed for presenting or teaching how the system works.

HOW TO RUN:
    1. First make sure models are trained:
         python data/generate_data.py
         python engine.py

    2. Then run this demo:
         python demo_recommendations.py

WHAT IT SHOWS:
    For each of the 4 functions:
      - The exact sample input being passed
      - Every intermediate variable with its value printed
      - The final output formatted clearly
      - Plain-English explanation of what happened

NO CODE CHANGES TO ENGINE.PY NEEDED — this file imports the trained artifacts
and calls verbose wrapper functions that add print statements around the
original logic.
"""

import os, pickle, json
import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────
# LOAD TRAINED ARTIFACTS (same as app.py does on startup)
# ────────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
DATA = os.path.join(BASE, "data")

clf = pickle.load(open(os.path.join(MODELS, "symptom_clf.pkl"), "rb"))
le = pickle.load(open(os.path.join(MODELS, "label_encoder.pkl"), "rb"))
feature_cols = pickle.load(open(os.path.join(MODELS, "feature_cols.pkl"), "rb"))
collab_data = pickle.load(open(os.path.join(MODELS, "collab_model.pkl"), "rb"))
assoc_data = pickle.load(open(os.path.join(MODELS, "assoc_rules.pkl"), "rb"))
meta_df = pd.read_csv(os.path.join(DATA, "conditions_meta.csv"))


def banner(title):
    """Print a visible section banner."""
    print("\n" + "=" * 70)
    print(f"  DEMO: {title}")
    print("=" * 70)


def sub(title):
    """Print a sub-section header."""
    print(f"\n  ── {title} ──")


# ════════════════════════════════════════════════════════════════════
#  DEMO 1: CONTENT-BASED RECOMMENDATION
# ════════════════════════════════════════════════════════════════════

def demo_content_based():
    """
    SAMPLE INPUT:
        symptoms selected = ["headache", "nausea", "light_sensitivity"]
        (typical Migraine symptoms)

    EXPECTED OUTPUT:
        Top 3 predicted conditions with confidence scores and medicines.
        Migraine should be #1 with high confidence.
    """
    banner("CONTENT-BASED RECOMMENDATION")

    # ── SAMPLE INPUT ──
    selected_symptoms = ["headache", "nausea", "light_sensitivity"]

    sub("STEP 1: Sample Input")
    print(f"    Selected symptoms: {selected_symptoms}")
    print(f"    Total symptom features in model: {len(feature_cols)}")

    # ── BUILD SYMPTOM VECTOR ──
    # Create a dict with 1 for selected symptoms, 0 for everything else
    symptoms_dict = {f: (1 if f in selected_symptoms else 0) for f in feature_cols}

    sub("STEP 2: Build Binary Symptom Vector")
    active = {k: v for k, v in symptoms_dict.items() if v == 1}
    inactive_count = sum(1 for v in symptoms_dict.values() if v == 0)
    print(f"    Active symptoms (=1):  {active}")
    print(f"    Inactive symptoms (=0): {inactive_count} features set to 0")

    # ── CONVERT TO NUMPY ARRAY ──
    # The classifier expects a 2D numpy array with shape (1, 35)
    vec = np.array([[symptoms_dict.get(f, 0) for f in feature_cols]])

    sub("STEP 3: Convert to Numpy Array for Classifier")
    print(f"    Vector shape: {vec.shape}")
    print(f"    Vector (showing non-zero positions only):")
    for i, f in enumerate(feature_cols):
        if vec[0][i] == 1:
            print(f"      Position {i:2d}: {f:30s} = 1")

    # ── PREDICT PROBABILITIES ──
    # clf.predict_proba returns probability for each of the 10 conditions
    proba = clf.predict_proba(vec)[0]

    sub("STEP 4: Classifier Predicts Probabilities for All 10 Conditions")
    print(f"    Probability distribution (all 10 conditions):")
    for i, cond_name in enumerate(le.classes_):
        bar = "█" * int(proba[i] * 50)  # simple visual bar
        print(f"      {cond_name:20s}  {proba[i]:.4f}  ({proba[i]*100:5.1f}%)  {bar}")

    # ── SELECT TOP 3 ──
    top_idx = np.argsort(proba)[::-1][:3]

    sub("STEP 5: Select Top 3 Conditions by Probability")
    print(f"    Sorted indices (descending): {np.argsort(proba)[::-1].tolist()}")
    print(f"    Top 3 indices selected: {top_idx.tolist()}")

    # ── LOOKUP MEDICINES ──
    sub("STEP 6: Look Up Medicines from conditions_meta.csv")
    results = []
    for rank, idx in enumerate(top_idx, 1):
        cond = le.classes_[idx]
        conf = round(float(proba[idx]), 3)
        matched = meta_df[meta_df["condition"] == cond]
        if matched.empty:
            print(f"\n    Rank #{rank}: {cond} — NOT FOUND in conditions_meta.csv (skipped)")
            continue
        row = matched.iloc[0]
        medicines = row["medicines"].split("|")[:5]
        severity = row["severity"]

        print(f"\n    Rank #{rank}: {cond}")
        print(f"      Confidence: {conf} ({conf*100:.1f}%)")
        print(f"      Severity:   {severity}")
        print(f"      Medicines:  {medicines}")

        results.append({
            "condition": cond, "confidence": conf,
            "severity": severity, "medicines": medicines,
        })

    # ── FINAL OUTPUT ──
    sub("FINAL OUTPUT (what the API returns as JSON)")
    print(f"    {json.dumps(results, indent=6)}")

    return results


# ════════════════════════════════════════════════════════════════════
#  DEMO 2: COLLABORATIVE FILTERING RECOMMENDATION
# ════════════════════════════════════════════════════════════════════

def demo_collaborative():
    """
    SAMPLE INPUT:
        user_id = 1

    EXPECTED OUTPUT:
        Top 5 medicines ranked by predicted rating for User 1.
        Medicines related to User 1's assigned conditions will rank highest.
    """
    banner("COLLABORATIVE FILTERING RECOMMENDATION")

    # ── SAMPLE INPUT ──
    user_id = 1

    sub("STEP 1: Sample Input")
    print(f"    User ID: {user_id}")
    print(f"    Total users in model: {len(collab_data['user_ids'])}")
    print(f"    Total medicines in model: {len(collab_data['med_names'])}")

    # ── FIND USER IN MATRIX ──
    uid_idx = collab_data["user_ids"].index(user_id)

    sub("STEP 2: Locate User in Reconstructed Rating Matrix")
    print(f"    User {user_id} is at row index: {uid_idx}")
    print(f"    Matrix shape: {collab_data['reconstructed'].shape}  (users × medicines)")
    print(f"    User's mean rating: {collab_data['user_means'][uid_idx][0]:.3f}")

    # ── GET ALL PREDICTED RATINGS ──
    scores = collab_data["reconstructed"][uid_idx]

    sub("STEP 3: Extract This User's Predicted Ratings (all 42 medicines)")
    print(f"    Rating range: min={scores.min():.3f}, max={scores.max():.3f}")
    print(f"    Sample ratings (first 10 medicines):")
    for i in range(min(10, len(scores))):
        print(f"      {collab_data['med_names'][i]:25s}  predicted rating = {scores[i]:.3f}")
    print(f"      ... ({len(scores) - 10} more medicines)")

    # ── SORT AND SELECT TOP 5 ──
    top_n = 5
    top_idx = np.argsort(scores)[::-1][:top_n]

    sub(f"STEP 4: Sort by Predicted Rating, Select Top {top_n}")
    print(f"    All 42 medicines sorted (showing top {top_n}):")
    results = []
    for rank, i in enumerate(top_idx, 1):
        med = collab_data["med_names"][i]
        rating = round(float(scores[i]), 2)
        stars = "★" * int(round(rating)) + "☆" * (5 - int(round(rating)))
        print(f"      #{rank}  {med:25s}  rating = {rating}  {stars}")
        results.append({"medicine": med, "predicted_rating": rating})

    # ── ALSO SHOW BOTTOM 3 (for contrast) ──
    sub("BONUS: Bottom 3 Medicines (lowest predicted ratings)")
    bottom_idx = np.argsort(scores)[:3]
    for i in bottom_idx:
        med = collab_data["med_names"][i]
        rating = round(float(scores[i]), 2)
        print(f"      {med:25s}  rating = {rating}  (not relevant for this user)")

    # ── FINAL OUTPUT ──
    sub("FINAL OUTPUT (what the API returns as JSON)")
    print(f"    {json.dumps(results, indent=6)}")

    return results


# ════════════════════════════════════════════════════════════════════
#  DEMO 3: ASSOCIATION RULES RECOMMENDATION
# ════════════════════════════════════════════════════════════════════

def demo_association():
    """
    SAMPLE INPUT:
        selected_medicines = ["Ibuprofen", "Dextromethorphan"]
        (typical Common Cold / Flu medicines)

    EXPECTED OUTPUT:
        Co-prescribed medicines found by matching rules.
        Vitamin C and Acetaminophen are likely matches (Cold/Flu combos).
    """
    banner("ASSOCIATION RULES RECOMMENDATION")

    # ── SAMPLE INPUT ──
    selected_medicines = ["Ibuprofen", "Dextromethorphan"]

    sub("STEP 1: Sample Input")
    print(f"    Medicines user is currently taking: {selected_medicines}")
    print(f"    Total rules in model: {len(assoc_data['rules'])}")

    selected = set(selected_medicines)

    # ── SCAN ALL RULES ──
    sub("STEP 2: Scan All Rules — Check Which Ones Fire")
    print(f"    A rule 'fires' when its antecedent is a SUBSET of the user's medicines.")
    print(f"    User's medicine set: {selected}")
    print()

    fired_count = 0
    skipped_count = 0
    scored = {}

    for i, rule in enumerate(assoc_data["rules"]):
        ante = set(rule["antecedent"])
        cons = rule["consequent"]
        is_subset = ante.issubset(selected)

        if is_subset:
            fired_count += 1
            for med in cons:
                if med not in selected:
                    score = round(rule["lift"] * rule["confidence"], 3)
                    print(f"    ✓ FIRES  Rule #{i+1:2d}: {rule['antecedent']} → {cons}")
                    print(f"               antecedent {ante} ⊆ user's {selected} = True")
                    print(f"               support={rule['support']}, confidence={rule['confidence']}, lift={rule['lift']}")
                    print(f"               score = lift × confidence = {rule['lift']} × {rule['confidence']} = {score}")
                    print(f"               → Candidate: {med}")
                    print()

                    if med not in scored or score > scored[med]["score"]:
                        scored[med] = {
                            "medicine": med, "score": score,
                            "support": rule["support"], "confidence": rule["confidence"],
                            "lift": rule["lift"], "triggered_by": rule["antecedent"],
                        }
                else:
                    print(f"    ✓ FIRES  Rule #{i+1:2d}: {rule['antecedent']} → {cons}")
                    print(f"               BUT {med} is already in user's medicines → SKIPPED")
                    print()
        else:
            skipped_count += 1
            # Only show first 3 skipped rules as examples
            if skipped_count <= 3:
                print(f"    ✗ SKIP   Rule #{i+1:2d}: {rule['antecedent']} → {cons}")
                print(f"               antecedent {ante} ⊄ user's {selected}")
            elif skipped_count == 4:
                print(f"    ... (skipping {len(assoc_data['rules']) - fired_count - 3} more non-matching rules)")

    sub("STEP 3: Rule Matching Summary")
    print(f"    Total rules scanned:  {len(assoc_data['rules'])}")
    print(f"    Rules that fired:     {fired_count}")
    print(f"    Rules skipped:        {skipped_count}")
    print(f"    Unique candidates:    {len(scored)}")

    # ── RANK CANDIDATES ──
    top_n = 5
    results = sorted(scored.values(), key=lambda x: -x["score"])[:top_n]

    sub(f"STEP 4: Rank Candidates by Score (top {top_n})")
    if results:
        for rank, r in enumerate(results, 1):
            print(f"    #{rank}  {r['medicine']:25s}  score={r['score']}")
            print(f"         triggered by: {r['triggered_by']}")
            print(f"         support={r['support']}, confidence={r['confidence']}, lift={r['lift']}")
    else:
        print("    No recommendations found. Try selecting different medicines.")

    # ── FINAL OUTPUT ──
    sub("FINAL OUTPUT (what the API returns as JSON)")
    print(f"    {json.dumps(results, indent=6, default=str)}")

    return results


# ════════════════════════════════════════════════════════════════════
#  DEMO 4: HYBRID RECOMMENDATION
# ════════════════════════════════════════════════════════════════════

def demo_hybrid():
    """
    SAMPLE INPUT:
        symptoms  = ["headache", "nausea", "light_sensitivity"]
        user_id   = 1
        content_weight = 0.6 (60% content-based, 40% collaborative)

    EXPECTED OUTPUT:
        Blended ranked list of medicines combining both strategies.
    """
    banner("HYBRID RECOMMENDATION (Content 60% + Collaborative 40%)")

    # ── SAMPLE INPUT ──
    selected_symptoms = ["headache", "nausea", "light_sensitivity"]
    user_id = 1
    content_weight = 0.6

    sub("STEP 1: Sample Input")
    print(f"    Symptoms: {selected_symptoms}")
    print(f"    User ID:  {user_id}")
    print(f"    Blend weights: content={content_weight}, collaborative={1-content_weight}")

    symptoms_dict = {f: (1 if f in selected_symptoms else 0) for f in feature_cols}

    # ── RUN CONTENT-BASED (internally) ──
    sub("STEP 2: Run Content-Based Strategy (for up to 20 medicines)")
    vec = np.array([[symptoms_dict.get(f, 0) for f in feature_cols]])
    proba = clf.predict_proba(vec)[0]
    top_idx = np.argsort(proba)[::-1][:3]

    cb_results = []
    for idx in top_idx:
        cond = le.classes_[idx]
        conf = round(float(proba[idx]), 3)
        matched = meta_df[meta_df["condition"] == cond]
        if matched.empty:
            continue
        row = matched.iloc[0]
        meds = row["medicines"].split("|")[:5]
        cb_results.append({"condition": cond, "confidence": conf, "medicines": meds})
        print(f"    {cond:20s} confidence={conf}  medicines={meds}")

    # ── RUN COLLABORATIVE (internally) ──
    sub("STEP 3: Run Collaborative Strategy (for up to 20 medicines)")
    uid_idx = collab_data["user_ids"].index(user_id)
    all_scores = collab_data["reconstructed"][uid_idx]
    cf_top_idx = np.argsort(all_scores)[::-1][:20]

    cf_results = []
    for i in cf_top_idx[:5]:  # show top 5 for display
        med = collab_data["med_names"][i]
        rating = round(float(all_scores[i]), 2)
        cf_results.append({"medicine": med, "predicted_rating": rating})
        print(f"    {med:25s}  predicted_rating={rating}")
    print(f"    ... (up to 20 medicines used internally)")

    # ── BLEND SCORES ──
    sub("STEP 4: Compute Blended Scores")
    print(f"    Formula: hybrid_score = (content_score × {content_weight}) + (collab_score × {1-content_weight})")
    print()

    scores = {}

    # Content-based scores
    print(f"    4a. Content-Based Scores (weight = {content_weight}):")
    for r in cb_results:
        for i, med in enumerate(r["medicines"]):
            position_decay = 1 - i * 0.1  # 1st=1.0, 2nd=0.9, 3rd=0.8...
            content_score = r["confidence"] * position_decay * content_weight
            scores[med] = scores.get(med, 0) + content_score
            print(f"        {med:25s}  conf={r['confidence']} × decay={position_decay:.1f} × weight={content_weight} = +{content_score:.4f}")

    # Collaborative scores
    print(f"\n    4b. Collaborative Scores (weight = {1-content_weight}):")
    cf_full = [{"medicine": collab_data["med_names"][i],
                "predicted_rating": float(all_scores[i])} for i in cf_top_idx]
    max_r = max(r["predicted_rating"] for r in cf_full) or 1
    print(f"        Max predicted rating for normalisation: {max_r:.3f}")
    for r in cf_full[:8]:  # show top 8 for display
        normalised = r["predicted_rating"] / max_r
        collab_score = normalised * (1 - content_weight)
        scores[r["medicine"]] = scores.get(r["medicine"], 0) + collab_score
        print(f"        {r['medicine']:25s}  rating={r['predicted_rating']:.2f} / {max_r:.2f} × {1-content_weight} = +{collab_score:.4f}")
    print(f"        ... (up to 20 medicines scored)")

    # ── FINAL BLEND ──
    sub("STEP 5: Final Blended Ranking")
    top_n = 5
    ranked = sorted(scores.items(), key=lambda x: -x[1])

    print(f"    All medicines with their total hybrid scores:")
    for rank, (med, s) in enumerate(ranked[:10], 1):  # show top 10
        marker = " ← SELECTED" if rank <= top_n else ""
        print(f"      #{rank:2d}  {med:25s}  hybrid_score = {s:.4f}{marker}")
    if len(ranked) > 10:
        print(f"      ... ({len(ranked) - 10} more medicines below)")

    results = [{"medicine": m, "hybrid_score": round(s, 3)} for m, s in ranked[:top_n]]

    # ── FINAL OUTPUT ──
    sub("FINAL OUTPUT (what the API returns as JSON)")
    print(f"    {json.dumps(results, indent=6)}")

    return results


# ════════════════════════════════════════════════════════════════════
#  DEMO 5: ASSOCIATION RULE MINING (mine_association_rules internals)
# ════════════════════════════════════════════════════════════════════

def demo_mine_association_rules():
    """
    Shows the internal workings of mine_association_rules() step by step
    using a TINY 5-transaction example, then the real 500-transaction data.
    """
    banner("ASSOCIATION RULE MINING — INTERNAL STEP-BY-STEP")

    from itertools import combinations

    # ── TINY EXAMPLE (5 transactions) ──
    sub("PART A: Tiny Example (5 transactions, easy to follow)")

    tiny_baskets = [
        {"Ibuprofen", "Acetaminophen", "Vitamin C"},
        {"Ibuprofen", "Acetaminophen", "Dextromethorphan"},
        {"Ibuprofen", "Acetaminophen"},
        {"Buspirone", "Escitalopram", "Hydroxyzine"},
        {"Buspirone", "Hydroxyzine"},
    ]
    n = len(tiny_baskets)

    print(f"\n    Transactions ({n} total):")
    for i, b in enumerate(tiny_baskets):
        print(f"      T{i+1}: {sorted(b)}")

    # Step 2: Count singles
    print(f"\n    STEP 2: Count Individual Items")
    all_items = []
    for b in tiny_baskets:
        all_items.extend(b)
    item_counts = pd.Series(all_items).value_counts()
    item_support = item_counts / n
    for item, count in item_counts.items():
        print(f"      {item:20s}  count={count}  support={count}/{n} = {count/n:.2f}")

    # Step 3: Count pairs
    print(f"\n    STEP 3: Count Pairs (using itertools.combinations)")
    pair_list = []
    for i, b in enumerate(tiny_baskets):
        pairs = list(combinations(sorted(b), 2))
        print(f"      T{i+1} {sorted(b)} → pairs: {pairs}")
        pair_list.extend(pairs)

    pair_counts = pd.Series(pair_list).value_counts()
    pair_support = pair_counts / n
    print(f"\n    Pair frequencies:")
    for pair, count in pair_counts.items():
        print(f"      {str(pair):50s}  count={count}  support={count/n:.2f}")

    # Step 4: Filter
    min_support = 0.40  # 40% for tiny example (= at least 2 out of 5)
    freq_pairs = pair_support[pair_support >= min_support]
    print(f"\n    STEP 4: Filter by min_support={min_support}")
    print(f"    Pairs kept: {len(freq_pairs)} out of {len(pair_counts)}")
    for pair, sup in freq_pairs.items():
        print(f"      ✓ {str(pair):50s}  support={sup:.2f}  (≥ {min_support})")

    # Step 5: Compute rules
    min_confidence = 0.40
    print(f"\n    STEP 5: Generate Rules (min_confidence={min_confidence})")
    rules = []
    for (a, b), sup in freq_pairs.items():
        sup_a = item_support.get(a, 0)
        sup_b = item_support.get(b, 0)

        conf_ab = sup / sup_a
        lift_ab = conf_ab / sup_b if sup_b > 0 else 0
        conf_ba = sup / sup_b
        lift_ba = conf_ba / sup_a if sup_a > 0 else 0

        print(f"\n      Pair: ({a}, {b})  joint_support={sup:.2f}")
        print(f"        Rule {a} → {b}:")
        print(f"          confidence = {sup:.2f} / {sup_a:.2f} = {conf_ab:.3f}  {'✓ PASS' if conf_ab >= min_confidence else '✗ FAIL'}")
        if conf_ab >= min_confidence:
            print(f"          lift = {conf_ab:.3f} / {sup_b:.2f} = {lift_ab:.3f}")
            rules.append({"rule": f"{a} → {b}", "conf": round(conf_ab, 3), "lift": round(lift_ab, 3)})

        print(f"        Rule {b} → {a}:")
        print(f"          confidence = {sup:.2f} / {sup_b:.2f} = {conf_ba:.3f}  {'✓ PASS' if conf_ba >= min_confidence else '✗ FAIL'}")
        if conf_ba >= min_confidence:
            print(f"          lift = {conf_ba:.3f} / {sup_a:.2f} = {lift_ba:.3f}")
            rules.append({"rule": f"{b} → {a}", "conf": round(conf_ba, 3), "lift": round(lift_ba, 3)})

    print(f"\n    RESULT: {len(rules)} rules generated from tiny example:")
    for r in sorted(rules, key=lambda x: -x["lift"]):
        print(f"      {r['rule']:40s}  confidence={r['conf']}  lift={r['lift']}")

    # ── REAL DATA SUMMARY ──
    sub("PART B: Real Data (500 transactions)")
    trans_df = pd.read_csv(os.path.join(DATA, "prescription_transactions.csv"))
    baskets = [set(row.split("|")) for row in trans_df["medicines"]]
    n_real = len(baskets)

    all_items_real = []
    for b in baskets:
        all_items_real.extend(b)
    item_counts_real = pd.Series(all_items_real).value_counts()

    pair_list_real = []
    for b in baskets:
        for pair in combinations(sorted(b), 2):
            pair_list_real.append(pair)
    pair_counts_real = pd.Series(pair_list_real).value_counts()
    freq_pairs_real = (pair_counts_real / n_real)
    freq_pairs_real = freq_pairs_real[freq_pairs_real >= 0.05]

    print(f"    Transactions:        {n_real}")
    print(f"    Unique items:        {len(item_counts_real)}")
    print(f"    Total unique pairs:  {len(pair_counts_real)}")
    print(f"    Frequent pairs:      {len(freq_pairs_real)} (support ≥ 0.05)")
    print(f"    Rules generated:     {len(assoc_data['rules'])} (confidence ≥ 0.40)")
    print(f"\n    Top 5 item frequencies:")
    for item, count in item_counts_real.head(5).items():
        print(f"      {item:25s}  count={count}  support={count/n_real:.3f}")
    print(f"\n    Top 5 pair frequencies:")
    for pair, sup in (pair_counts_real / n_real).head(5).items():
        print(f"      {str(pair):50s}  support={sup:.3f}")
    print(f"\n    Top 5 rules by lift:")
    for r in assoc_data['rules'][:5]:
        print(f"      {str(r['antecedent']):20s} → {str(r['consequent']):20s}  "
              f"sup={r['support']}  conf={r['confidence']}  lift={r['lift']}")


# ════════════════════════════════════════════════════════════════════
#  MAIN: RUN ALL DEMOS
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║   MedRec — Complete Recommendation Function Demonstration          ║")
    print("║   Showing sample inputs, intermediate steps, and outputs           ║")
    print("╚" + "═" * 68 + "╝")

    # Demo 1: Content-Based
    demo_content_based()

    # Demo 2: Collaborative Filtering
    demo_collaborative()

    # Demo 3: Association Rules
    demo_association()

    # Demo 4: Hybrid
    demo_hybrid()

    # Demo 5: Mining internals
    demo_mine_association_rules()

    print("\n" + "=" * 70)
    print("  ALL DEMOS COMPLETE")
    print("=" * 70)
    print("""
  Summary of sample inputs used:
  ┌──────────────────────┬───────────────────────────────────────────────┐
  │ Function             │ Sample Input                                  │
  ├──────────────────────┼───────────────────────────────────────────────┤
  │ Content-Based        │ symptoms: headache, nausea, light_sensitivity │
  │ Collaborative        │ user_id: 1                                    │
  │ Association Rules    │ medicines: Ibuprofen, Dextromethorphan        │
  │ Hybrid               │ symptoms + user_id: same as above combined    │
  │ Mining Internals     │ 5 tiny transactions + full 500 transactions   │
  └──────────────────────┴───────────────────────────────────────────────┘
    """)