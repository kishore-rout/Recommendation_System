# 💊 MedRec — Intelligent Medicine Recommendation Engine

<p align="center">
  <img src="https://img.shields.io/badge/AI-Medicine%20Recommendation-7C3AED?style=for-the-badge" alt="AI">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Pandas-Data%20Science-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Status-Completed-22C55E?style=for-the-badge" alt="Status">
</p>

<p align="center">
  <b>An end-to-end Machine Learning recommendation system that combines symptom-based classification, collaborative filtering, association-rule mining, and hybrid ranking.</b>
</p>

---

## 🌟 Project Overview

**MedRec** is an intelligent medicine recommendation engine designed to demonstrate how multiple Machine Learning recommendation strategies can work together in a single application.

Instead of depending on one algorithm, the system combines three complementary approaches:

- 🩺 **Content-Based Recommendation** — predicts likely conditions from symptoms.
- 👥 **Collaborative Filtering** — recommends medicines using user-medicine rating patterns.
- 🔗 **Association Rules** — discovers medicines frequently prescribed together.
- 🧠 **Hybrid Recommendation** — combines content and collaborative scores into one ranked list.

The project also includes a complete demonstration script that exposes the intermediate calculations behind every recommendation strategy, making it suitable for **ML project presentations, learning, portfolio demonstrations, and technical interviews**.

> ⚠️ **Disclaimer:** This is an educational Machine Learning project and is not a medical diagnostic or prescribing system. Real medical decisions should be made by qualified healthcare professionals.

---

## 🎯 Business Problem

Traditional recommendation systems often rely on only one source of information.

For medicine-related recommendation scenarios, different users may provide different types of information:

| User Information | Useful Strategy |
|---|---|
| Symptoms | 🩺 Content-Based |
| Previous medicine ratings | 👥 Collaborative Filtering |
| Current medicines | 🔗 Association Rules |
| Symptoms + user history | 🧠 Hybrid |

MedRec addresses this by combining multiple signals into a single recommendation engine.

### 💡 Core Idea

```text
                    ┌──────────────────────┐
                    │      User Input      │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       🩺 Symptoms       👥 User History    🔗 Medicines
              │                │                │
              ▼                ▼                ▼
       Random Forest /    TruncatedSVD     Association
       Gradient Boosting                  Rule Mining
              │                │                │
              └────────────────┼────────────────┘
                               │
                               ▼
                    🧠 Recommendation Layer
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
      Individual Results                 Hybrid Ranking
              │                                 │
              └────────────────┬────────────────┘
                               ▼
                    💊 Ranked Suggestions
```

---

# 🚀 Key Features

### 🩺 1. Symptom-Based Prediction

Accepts a binary symptom vector and predicts the most probable conditions.

**Models compared:**
- Random Forest
- Gradient Boosting

The system uses **5-fold cross-validation** to select the stronger model.

---

### 👥 2. Collaborative Filtering

Uses historical user-medicine ratings to identify hidden preference patterns.

**Technique:**
- User-Medicine Matrix
- Mean Centering
- Truncated SVD
- Latent Factor Reconstruction

The reconstructed matrix provides predicted ratings for medicines that a user may prefer.

---

### 🔗 3. Association Rule Mining

Discovers frequently co-occurring medicines from prescription transactions.

The system calculates:

- Support
- Confidence
- Lift

Example:

```text
Ibuprofen → Vitamin C

Support      = P(Ibuprofen AND Vitamin C)
Confidence   = P(Vitamin C | Ibuprofen)
Lift         = Confidence / P(Vitamin C)
```

---

### 🧠 4. Hybrid Recommendation

Combines:

```text
60% Content-Based
        +
40% Collaborative Filtering
        =
Hybrid Recommendation Score
```

The final medicines are ranked according to the combined score.

---

### 📊 5. ML Evaluation & Visualization

The training pipeline automatically generates **8 visualization charts**:

| Chart | Purpose |
|---|---|
| `confusion_matrix.png` | Condition classification performance |
| `feature_importance.png` | Important symptoms |
| `model_comparison.png` | RF vs GB comparison |
| `svd_variance.png` | SVD variance analysis |
| `rating_distribution.png` | User rating distribution |
| `association_rules_lift.png` | Strongest medicine associations |
| `association_scatter.png` | Support vs confidence |
| `item_frequency.png` | Medicine frequency |

---

### 🔍 6. Explainable Demo Pipeline

`demo_recommendations.py` shows:

- Input values
- Binary symptom vector
- Model probabilities
- Top predictions
- SVD reconstructed ratings
- Association-rule matching
- Support / confidence / lift
- Hybrid score calculation
- Final recommendations

This makes the project easy to explain during an interview.

---

# 🏗️ System Architecture

```text
                     ┌───────────────────────┐
                     │      CSV Datasets     │
                     └───────────┬───────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
       Symptom Dataset     Rating Dataset    Transaction Dataset
              │                  │                  │
              ▼                  ▼                  ▼
       ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
       │ Classifier  │    │ Truncated   │    │ Association  │
       │ RF / GB     │    │ SVD         │    │ Rules        │
       └──────┬──────┘    └──────┬──────┘    └──────┬───────┘
              │                  │                  │
              ▼                  ▼                  ▼
       Condition Scores    Predicted Ratings    Rule Scores
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 │
                                 ▼
                     ┌────────────────────────┐
                     │ Recommendation Engine  │
                     └───────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        Content-Based      Collaborative       Association
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 │
                                 ▼
                         🧠 Hybrid Ranking
                                 │
                                 ▼
                       💊 Top N Suggestions
```

---

# 🧠 Machine Learning Techniques

## 1️⃣ Random Forest

The symptom classifier uses a Random Forest consisting of multiple decision trees.

```text
Symptoms
   │
   ├── Tree 1 ──┐
   ├── Tree 2 ──┤
   ├── Tree 3 ──┤
   ├── ...      ├──► Voting ──► Condition
   └── Tree 150 ┘
```

Configuration:

```python
RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
```

### Why Random Forest?

- Handles binary features naturally
- No feature scaling required
- Captures non-linear relationships
- Provides feature importance
- Provides class probabilities
- Robust against overfitting when appropriately constrained

---

## 2️⃣ Gradient Boosting

Gradient Boosting is trained as the second candidate model.

```python
GradientBoostingClassifier(
    n_estimators=120,
    max_depth=5,
    random_state=42
)
```

It builds trees sequentially, with later trees attempting to correct errors made by earlier trees.

---

## 3️⃣ Model Selection

Both classifiers are evaluated using **5-fold cross-validation**.

```text
Dataset
   │
   ├── Fold 1 → Train / Validate
   ├── Fold 2 → Train / Validate
   ├── Fold 3 → Train / Validate
   ├── Fold 4 → Train / Validate
   └── Fold 5 → Train / Validate
              │
              ▼
        Mean Accuracy
              │
              ▼
       Best Model Selected
```

The model with the higher mean CV accuracy is saved as the final symptom classifier.

---

# 👥 Collaborative Filtering

The collaborative component converts the rating dataset into a user-medicine matrix.

```text
             Medicine
          M1   M2   M3   M4
       ┌────────────────────
User 1 │  5    4    0    3
User 2 │  4    0    5    2
User 3 │  0    5    4    0
User 4 │  3    2    0    5
       └────────────────────
```

### Processing

```text
Ratings
   ↓
Pivot Table
   ↓
User Mean Centering
   ↓
TruncatedSVD
   ↓
Latent Factors
   ↓
Matrix Reconstruction
   ↓
Predicted Ratings
   ↓
Top Medicines
```

The implementation retains up to **15 latent components**, subject to the matrix dimensions.

---

# 🔗 Association Rule Mining

Instead of implementing Apriori from scratch, the project uses:

- Pandas
- `itertools.combinations`
- Frequency counting

### Example Basket

```text
Transaction:
{
    Ibuprofen,
    Acetaminophen,
    Vitamin C
}
```

Generated pairs:

```text
Ibuprofen + Acetaminophen
Ibuprofen + Vitamin C
Acetaminophen + Vitamin C
```

### Metrics

#### Support

```text
Support(A,B) = Count(A,B) / Total Transactions
```

Measures how frequently the pair appears.

#### Confidence

```text
Confidence(A→B) = Support(A,B) / Support(A)
```

Measures how often B occurs when A occurs.

#### Lift

```text
Lift(A→B) = Confidence(A→B) / Support(B)
```

Interpretation:

```text
Lift > 1  → Positive association
Lift = 1  → No meaningful association
Lift < 1  → Negative association
```

---

# 🧠 Hybrid Recommendation

The hybrid system combines two recommendation signals.

### Content Score

For each medicine:

```text
Condition Confidence
        ×
Position Decay
        ×
Content Weight
```

Position decay:

```text
1st medicine → 1.0
2nd medicine → 0.9
3rd medicine → 0.8
4th medicine → 0.7
5th medicine → 0.6
```

### Collaborative Score

Predicted ratings are normalized:

```text
Normalized Rating = Predicted Rating / Maximum Rating
```

Then:

```text
Hybrid Score =
    0.6 × Content Score
    +
    0.4 × Collaborative Score
```

Finally, medicines are sorted by the hybrid score.

---

# 📁 Project Structure

```text
MedRec/
│
├── 📄 engine.py
├── 📄 demo_recommendations.py
├── 📄 app.py
├── 📄 README.md
├── 📄 requirements.txt
│
├── 📂 data/
│   ├── symptom_condition.csv
│   ├── user_ratings.csv
│   ├── conditions_meta.csv
│   └── prescription_transactions.csv
│
├── 📂 models/
│   ├── symptom_clf.pkl
│   ├── label_encoder.pkl
│   ├── feature_cols.pkl
│   ├── collab_model.pkl
│   ├── assoc_rules.pkl
│   └── metrics.json
│
└── 📂 static/
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── model_comparison.png
    ├── svd_variance.png
    ├── rating_distribution.png
    ├── association_rules_lift.png
    ├── association_scatter.png
    └── item_frequency.png
```

---

# 📊 Dataset Overview

The training pipeline works with four CSV datasets.

| Dataset | Approx. Size | Purpose |
|---|---:|---|
| `symptom_condition.csv` | 2,000 × 36 | Symptom → condition classification |
| `user_ratings.csv` | 1,600 × 3 | Collaborative filtering |
| `conditions_meta.csv` | 10 × 4 | Condition and medicine metadata |
| `prescription_transactions.csv` | 500 × 4 | Association-rule mining |

### Symptom Dataset

```text
35 symptom features
+
1 condition label
```

### Rating Dataset

```text
user_id
medicine
rating
```

### Condition Metadata

```text
condition
medicines
severity
symptoms
```

### Transaction Dataset

```text
transaction_id
medicines
conditions
basket_size
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd MedRec
```

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

Example `requirements.txt`:

```text
numpy
pandas
scikit-learn
matplotlib
seaborn
```

If your `app.py` uses additional libraries such as Flask or Streamlit, add the corresponding package to `requirements.txt`.

---

# ▶️ How to Run

## Step 1 — Generate Data

If your project includes `generate_data.py`:

```bash
python data/generate_data.py
```

This creates the four CSV datasets.

---

## Step 2 — Train the Models

Run:

```bash
python engine.py
```

The training pipeline performs:

```text
[1/3] Symptom Classifier
        ↓
Random Forest vs Gradient Boosting
        ↓
Best model saved

[2/3] Collaborative Filtering
        ↓
TruncatedSVD
        ↓
Predicted rating matrix saved

[3/3] Association Rules
        ↓
Support + Confidence + Lift
        ↓
Rules saved
```

It also generates:

- 8 charts
- model artifacts
- evaluation metrics

---

## Step 3 — Run Recommendation Demo

```bash
python demo_recommendations.py
```

The demo executes:

```text
Demo 1 → Content-Based
Demo 2 → Collaborative Filtering
Demo 3 → Association Rules
Demo 4 → Hybrid Recommendation
Demo 5 → Association Rule Mining Internals
```

---

## Step 4 — Run Application

After training:

```bash
python app.py
```

The application can load the saved model artifacts without retraining the models every time.

---

# 🧪 Sample Inputs

## 🩺 Content-Based

```python
[
    "headache",
    "nausea",
    "light_sensitivity"
]
```

Expected behavior:

```text
Symptoms
   ↓
Classifier
   ↓
Top 3 Conditions
   ↓
Condition Metadata
   ↓
Medicine Suggestions
```

---

## 👥 Collaborative Filtering

```python
user_id = 1
```

Expected behavior:

```text
User 1
  ↓
Predicted Ratings
  ↓
Sort Descending
  ↓
Top 5 Medicines
```

---

## 🔗 Association Rules

```python
[
    "Ibuprofen",
    "Dextromethorphan"
]
```

Expected behavior:

```text
Current Medicines
       ↓
Match Antecedents
       ↓
Calculate Rule Scores
       ↓
Remove Existing Medicines
       ↓
Top Recommendations
```

---

## 🧠 Hybrid

```python
symptoms = [
    "headache",
    "nausea",
    "light_sensitivity"
]

user_id = 1
```

Weights:

```text
Content-Based      = 60%
Collaborative      = 40%
```

---

# 📈 Evaluation

The project stores model evaluation results in:

```text
models/metrics.json
```

Example structure:

```json
{
    "best_model": "RandomForest",
    "rf_cv_mean": 0.95,
    "gb_cv_mean": 0.94,
    "test_accuracy": 0.96,
    "per_class": {
        "Condition": {
            "precision": 0.95,
            "recall": 0.96,
            "f1": 0.95
        }
    },
    "assoc": {
        "n_transactions": 500,
        "n_freq_pairs": 20,
        "n_rules": 35,
        "min_support": 0.05,
        "min_confidence": 0.4
    }
}
```

> The actual numbers depend on the generated dataset and should be taken from the project's generated `metrics.json`, rather than hard-coded in the README.

---

# 📊 Generated Visualizations

### 1. Confusion Matrix

Shows correct and incorrect predictions across the condition classes.

### 2. Feature Importance

Identifies the symptoms contributing most to the tree-based classifier.

### 3. Model Comparison

Compares 5-fold cross-validation accuracy between Random Forest and Gradient Boosting.

### 4. SVD Variance

Shows cumulative variance captured by the latent SVD components.

### 5. Rating Distribution

Visualizes the distribution of user-medicine ratings.

### 6. Association Rule Lift

Ranks the strongest medicine associations by lift.

### 7. Support vs Confidence

Visualizes association-rule quality.

### 8. Medicine Frequency

Shows the most frequently occurring medicines in the prescription transactions.

---

# 🧩 Recommendation Functions

The core recommendation layer contains four reusable functions:

| Function | Input | Output |
|---|---|---|
| `recommend_content_based()` | Symptoms | Conditions + medicines |
| `recommend_collaborative()` | User ID | Ranked medicines |
| `recommend_association()` | Current medicines | Co-prescribed medicines |
| `recommend_hybrid()` | Symptoms + User ID | Blended medicine ranking |

This modular design allows `app.py` to call the recommendation functions directly.

---

# 💾 Model Artifacts

After running `engine.py`, the following artifacts are created:

### `symptom_clf.pkl`

Stores the winning symptom classification model.

### `label_encoder.pkl`

Converts encoded condition labels back to their original names.

### `feature_cols.pkl`

Stores the exact feature order expected by the classifier.

### `collab_model.pkl`

Stores:

- User IDs
- Medicine names
- Reconstructed rating matrix
- User means
- SVD component count
- Explained variance

### `assoc_rules.pkl`

Stores:

- Association rules
- Transaction count
- Frequent pair count
- Rule count
- Support threshold
- Confidence threshold

### `metrics.json`

Stores the evaluation metrics used by the dashboard/application.

---

# 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Language | 🐍 Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Classification | Random Forest, Gradient Boosting |
| Recommendation | Content-Based, Collaborative Filtering, Hybrid |
| Dimensionality Reduction | TruncatedSVD |
| Encoding | LabelEncoder |
| Association Mining | Pandas + itertools |
| Visualization | Matplotlib, Seaborn |
| Model Serialization | Pickle |
| Metrics Storage | JSON |
| Application | `app.py` |

---

# 💡 Why This Project Is Different

Many beginner ML projects stop at:

```text
Dataset → Model → Prediction
```

MedRec goes further:

```text
Dataset
   ↓
Multiple ML Strategies
   ↓
Model Comparison
   ↓
Recommendation Layer
   ↓
Hybrid Ranking
   ↓
Model Serialization
   ↓
Visualization
   ↓
Demo / Application Integration
```

### 🔥 Portfolio Highlights

- ✅ Multiple recommendation algorithms
- ✅ Classification model comparison
- ✅ Cross-validation
- ✅ Dimensionality reduction
- ✅ Association-rule metrics
- ✅ Hybrid recommendation
- ✅ Explainable intermediate outputs
- ✅ Model persistence with Pickle
- ✅ Automated visualization generation
- ✅ Modular architecture
- ✅ Application-ready recommendation functions

---

# 🎤 Interview Explanation

### "Tell me about your project."

> **MedRec is an end-to-end Machine Learning-based medicine recommendation engine that combines three recommendation strategies. First, I use Random Forest and Gradient Boosting to classify conditions from 35 binary symptom features and select the better model using 5-fold cross-validation. Second, I use TruncatedSVD-based collaborative filtering to learn latent user-medicine preferences from historical ratings. Third, I mine association rules from prescription transactions using support, confidence, and lift. Finally, I built a hybrid recommendation function that combines content-based and collaborative scores using a 60-40 weighting scheme. I also created a demonstration pipeline that prints the intermediate calculations so the recommendation process is transparent and easy to explain.**

---

# 📌 Future Improvements

The current system can be extended with:

- 🔹 Real clinical datasets
- 🔹 More advanced NLP for free-text symptoms
- 🔹 XGBoost / LightGBM model comparison
- 🔹 Hyperparameter optimization
- 🔹 Neural collaborative filtering
- 🔹 Proper implicit-feedback recommendation
- 🔹 Matrix factorization with missing-value handling
- 🔹 Multi-item association rules
- 🔹 Explainable AI with SHAP
- 🔹 Confidence calibration
- 🔹 User authentication
- 🔹 Real-time recommendation API
- 🔹 Docker deployment
- 🔹 Cloud deployment
- 🔹 Medical knowledge graph integration
- 🔹 LLM/RAG-based medical information assistant with verified sources

---

# ⚠️ Medical Safety Notice

This project is intended **only for educational, demonstration, and Machine Learning portfolio purposes**.

The recommendations produced by the system are based on synthetic/demo-style datasets and machine-learning patterns. They should **not** be treated as medical advice, diagnosis, treatment guidance, or prescriptions.

Always consult a qualified healthcare professional for real medical decisions.

---

# 👨‍💻 Project Skills Demonstrated

```text
Python
│
├── Data Processing
│   ├── Pandas
│   └── NumPy
│
├── Machine Learning
│   ├── Random Forest
│   ├── Gradient Boosting
│   ├── Cross Validation
│   └── Classification Metrics
│
├── Recommendation Systems
│   ├── Content-Based
│   ├── Collaborative Filtering
│   ├── Association Rules
│   └── Hybrid Recommendation
│
├── Unsupervised Learning
│   └── TruncatedSVD
│
├── Data Visualization
│   ├── Matplotlib
│   └── Seaborn
│
└── Deployment Preparation
    ├── Pickle Model Artifacts
    ├── JSON Metrics
    └── Modular Recommendation API
```

---

# ⭐ Project Summary

**MedRec** demonstrates how multiple Machine Learning techniques can be integrated into a single recommendation pipeline.

```text
🩺 Symptoms
     +
👥 User Preferences
     +
🔗 Prescription Patterns
     ↓
🧠 Multiple ML Models
     ↓
📊 Scoring & Ranking
     ↓
💊 Personalized Recommendations
```

<p align="center">
  <b>Built with Python • Machine Learning • Recommendation Systems • Data Science</b>
</p>

<p align="center">
  ⭐ If you found this project useful, consider giving the repository a star!
</p>
