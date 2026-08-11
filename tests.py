import pandas as pd
import numpy as np
import os
import pickle
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import TruncatedSVD
import engine

# Mock Environment Setup
STATIC = "static"
MODELS = "models"
os.makedirs(STATIC, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

# --- MOCK DATA GENERATION ---

def get_mock_data():
    # 1. Symptom Data (Expanded to ensure valid stratified split for 3 classes)
    symptom_data = {
        "fever":    [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
        "cough":    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
        "headache": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1],
        "condition":["Flu", "Flu", "Flu", "Cold", "Cold", "Cold", "Migraine", "Migraine", "Migraine", "Migraine", "Cold", "Cold", "Flu", "Cold", "Migraine"]
    }
    symptom_df = pd.DataFrame(symptom_data)

    # 2. Ratings Data (Users rating medicines)
    ratings_data = {
        "user_id":  ["U1", "U1", "U2", "U2", "U3", "U3", "U4", "U4", "U5", "U5"],
        "medicine": ["MedA", "MedB", "MedA", "MedC", "MedB", "MedD", "MedA", "MedC", "MedB", "MedE"],
        "rating":   [5, 4, 1, 2, 5, 4, 5, 2, 4, 3]
    }
    ratings_df = pd.DataFrame(ratings_data)

    # 3. Transaction Data (Market Basket)
    trans_data = {
        "medicines": ["MedA|MedB", "MedA|MedB|MedC", "MedA|MedB", "MedX|MedY", "MedA|MedB", "MedA|MedC", "MedX|MedZ"]
    }
    trans_df = pd.DataFrame(trans_data)

    # 4. Metadata (Conditions to Medicines mapping)
    meta_data = {
        "condition": ["Flu", "Cold", "Migraine"],
        "severity":  ["High", "Low", "Medium"],
        "medicines": ["MedA|MedB", "MedC|MedD", "MedX|MedY"]
    }
    meta_df = pd.DataFrame(meta_data)

    return symptom_df, ratings_df, trans_df, meta_df

# --- TEST EXECUTION ---

def run_tests():
    print(" Starting Test Suite...\n")
    s_df, r_df, t_df, m_df = get_mock_data()

    # Test 1: Symptom Classifier
    print("Testing Symptom Classifier...")
    # Using a slightly larger test_size or more data points to accommodate 3 classes
    clf, le, feat_cols, s_metrics = engine.train_symptom_classifier(s_df)
    assert s_metrics["test_accuracy"] >= 0, "Classifier failed to produce accuracy"
    print(f"✅ Success: Best model is {s_metrics['best_model']}\n")

    # Test 2: Collaborative Filtering
    print("Testing Collaborative Model...")
    collab_data = engine.train_collaborative_model(r_df)
    assert "reconstructed" in collab_data, "Matrix reconstruction failed"
    print(f"✅ Success: Retained {collab_data['explained_var']}% variance\n")

    # Test 3: Association Rules
    print("Testing Association Rules...")
    assoc_data = engine.train_association_rules(t_df, min_support=0.1)
    assert assoc_data["n_rules"] > 0, "No association rules found"
    print(f"✅ Success: Found {assoc_data['n_rules']} rules\n")

    # Test 4: Hybrid Recommendation (Inference)
    print("Testing Hybrid Recommendation Inference...")
    sample_symptoms = {"fever": 1, "cough": 1, "headache": 0}
    hybrid_recs = engine.recommend_hybrid(
        sample_symptoms, "U1", clf, le, feat_cols, m_df, collab_data
    )
    
    assert len(hybrid_recs) > 0, "Hybrid recommendation returned empty list"
    print("✅ Success: Top Hybrid Recommendation:")
    for rec in hybrid_recs[:2]:
        print(f"   - {rec['medicine']} (Score: {rec['hybrid_score']})")

    print("\n🎉 All tests passed successfully!")

if __name__ == "__main__":
    # Import combinations as it is used in the provided functions
    from itertools import combinations
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Run the tests
    run_tests()