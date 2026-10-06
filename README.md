# CKD Risk Prediction & Explainable ML Pipeline

This repository contains a research-oriented machine learning pipeline for **Chronic Kidney Disease (CKD) risk stratification** using demographic, lifestyle, family-history, and health-related questionnaire features.

The main analysis is implemented in `CKD_Model.ipynb`. The notebook builds a binary CKD-risk label from a composite risk score, compares multiple machine-learning models and class-imbalance strategies, evaluates calibration and classification performance, and performs explainability, robustness, ablation, decision-curve, and subgroup/fairness analyses.

> **Important:** This is a research/ML analysis pipeline. The risk labels in this notebook are derived from the dataset's composite risk features and clustering procedure; they are **not equivalent to a clinical CKD diagnosis** and should not be used as a standalone medical diagnostic tool.

---

## Project Overview

The pipeline follows these major stages:

1. Load and inspect the CKD dataset.
2. Clean column names and remove selected redundant/proxy variables.
3. Construct a composite risk score.
4. Derive binary `Low Risk` / `High Risk` labels using K-Means clustering.
5. Validate the clustering structure using statistical and stability analyses.
6. Split the data into stratified training and test sets.
7. Train and tune five ML algorithms using two class-imbalance strategies:
   - Class weighting
   - SMOTE
8. Compare models using F1 score and ROC-AUC.
9. Select and calibrate the best cross-validation model.
10. Optimize the classification threshold using training-set out-of-fold predictions.
11. Evaluate the final model on an untouched test set.
12. Perform nested cross-validation as a robustness check.
13. Generate ROC, Precision-Recall, calibration, learning-curve, and confusion-matrix analyses.
14. Perform SHAP and LIME explainability analyses.
15. Compare SHAP and LIME explanations and assess explanation stability.
16. Test perturbation sensitivity of local explanations.
17. Compare alternative GMM-derived risk labels with K-Means labels.
18. Perform model-comparison statistical tests.
19. Run decision-curve analysis.
20. Conduct proxy-feature ablation and subgroup/fairness analyses.

---

## Dataset

The notebook expects the input dataset:

```text
ckd_data.xlsx
```

The dataset used in the notebook contains:

- **1,002 observations**
- **27 original columns**
- Age range: **20–60 years**
- No missing values were observed in the displayed dataset inspection.

The original variables include demographic, socioeconomic, anthropometric, lifestyle, family-history, personal-health, environmental, and kidney-related factors.

Examples include:

- `Age`
- `Gender`
- `Socioeconomic Status`
- `Weight (kg)`
- `Height (cm)`
- `Bmi`
- `Bmi Category`
- `Edema (swelling)`
- `Urination frequency`
- `Sleep duration`
- `Water intake`
- `Salt (sodium) intake`
- `Sweets/sugary drinks intake`
- `Smoking Status`
- `Alcohol Consumption`
- `Fast food/processed food intake`
- `Tea/Coffee (caffeine) intake`
- `Daily stress level`
- `Herbal/traditional supplement use`
- `Sedentary hours/day`
- `Family history of hypertension`
- `Family history of diabetes`
- `Personal hypertension`
- `Personal diabetes`
- `Painkiller (NSAID) use frequency`
- `Environmental pollution exposure`
- `History of kidney stones/UTI`

---

## Risk-Label Construction

A composite score called `New_Total_Risk` is created from six variables:

```text
Personal hypertension
Personal diabetes
Edema (swelling)
Urination frequency
History of kidney stones/UTI
Painkiller (NSAID) use frequency
```

The resulting score ranges from **0 to 15**.

A two-cluster K-Means model is then fitted to this one-dimensional composite score:

```python
KMeans(n_clusters=2, random_state=42, n_init=10)
```

The clusters are ordered by their centroids and mapped to:

```text
0 → Low Risk
1 → High Risk
```

In the notebook output:

- Low Risk: **610 participants (60.9%)**
- High Risk: **392 participants (39.1%)**

The K-Means solution has a reported silhouette score of **0.6817**.

The notebook also validates the two-cluster structure with a Gaussian Mixture Model (GMM). The GMM produced the same participant assignments as K-Means in the reported analysis:

- Adjusted Rand Index (ARI): **1.0000**
- Relabeled participants: **0 / 1002**

A 100-bootstrap GMM stability analysis reported mean ARI **0.9982 ± 0.0182**.

---

## Feature Selection for Prediction

The six variables used to construct `New_Total_Risk` are excluded from the downstream predictors to reduce direct target leakage.

The final predictive feature set contains **18 predictors**:

```text
Age
Gender
Socioeconomic Status
Bmi
Sleep duration
Water intake
Salt (sodium) intake
Sweets/sugary drinks intake
Smoking Status
Alcohol Consumption
Fast food/processed food intake
Tea/Coffee (caffeine) intake
Daily stress level
Herbal/traditional supplement use
Sedentary hours/day
Family history of hypertension
Family history of diabetes
Environmental pollution exposure
```

The data are divided using a stratified 80/20 train-test split:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=40,
    stratify=y
)
```

The resulting test set contains **201 observations**.

---

## Machine Learning Models

Five classification algorithms are evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. Support Vector Machine (SVM)
5. LightGBM

### Strategy A — Class Weighting

Class imbalance is handled using model-specific class weighting or XGBoost's `scale_pos_weight`.

Hyperparameters are optimized with:

- `RandomizedSearchCV`
- 40 random configurations per model
- 10-fold stratified cross-validation
- F1 as the refit metric
- ROC-AUC as an additional evaluation metric

### Strategy B — SMOTE

A second pipeline uses:

```python
SMOTE(random_state=42)
```

inside an imbalanced-learn pipeline.

This is evaluated with the same general cross-validation and randomized hyperparameter-search framework. SVM uses a lighter search configuration with 5-fold CV.

---

## Cross-Validation Results

The notebook reports the following mean cross-validation performance:

| Model | Strategy | Mean F1 | Mean ROC-AUC |
|---|---|---:|---:|
| Logistic Regression | Class Weighting | 0.7384 | 0.8746 |
| Random Forest | Class Weighting | 0.7445 | 0.8735 |
| XGBoost | Class Weighting | 0.7535 | 0.8705 |
| SVM | Class Weighting | 0.7307 | 0.8692 |
| LightGBM | Class Weighting | 0.7522 | 0.8736 |
| Logistic Regression | SMOTE | 0.7222 | 0.8718 |
| Random Forest | SMOTE | 0.7352 | 0.8724 |
| XGBoost | SMOTE | 0.7504 | 0.8735 |
| LightGBM | SMOTE | 0.7300 | 0.8665 |
| SVM | SMOTE | 0.7189 | 0.8658 |

The notebook selects the **XGBoost + Class Weighting** configuration based on the highest recorded mean F1 during the model-search comparison:

```text
CV F1:      0.7535 ± 0.0494
CV ROC-AUC: 0.8705 ± 0.0314
```

Selected XGBoost parameters reported by the notebook include:

```text
n_estimators       = 100
learning_rate      = 0.005
max_depth          = 3
subsample          = 0.7
colsample_bytree   = 0.8
min_child_weight   = 5
reg_alpha          = 1.0
reg_lambda         = 5.0
scale_pos_weight   ≈ 1.559
objective          = binary:logistic
```

---

## Calibration and Threshold Optimization

The selected estimator is calibrated using isotonic regression:

```python
CalibratedClassifierCV(
    estimator=best_estimator,
    cv=5,
    method="isotonic"
)
```

Instead of automatically using a probability threshold of 0.5, the notebook estimates an operating threshold from training-set out-of-fold predictions using **Youden's J statistic**.

The reported threshold is:

```text
0.4109
```

This threshold is then fixed and applied to the untouched test set.

---

## Final Test-Set Performance

The reported final test set contains:

```text
201 observations
```

Confusion matrix:

```text
[[101, 21],
 [ 15, 64]]
```

Therefore:

| Metric | Value |
|---|---:|
| Sensitivity | 0.810 |
| Specificity | 0.828 |
| Precision | 0.753 |
| NPV | 0.871 |
| Accuracy | 0.821 |
| F1-score | 0.780 |

The notebook also reports a Brier score of:

```text
0.1191
```

Performance visualizations include ROC, Precision-Recall, and calibration curves.

---

## Nested Cross-Validation

A separate 5-fold nested cross-validation analysis is used as a robustness check for the XGBoost + Class Weighting configuration.

Reported nested-CV results:

```text
Nested CV F1:       0.7479 ± 0.0160
Nested CV ROC-AUC:  0.8684 ± 0.0103
```

The notebook compares these values with the non-nested estimates and reports estimated optimistic bias of:

```text
F1:      +0.0057
ROC-AUC: +0.0021
```

---

## Statistical Model Comparison

The notebook reruns 10-fold CV using the already-tuned model configurations and performs paired comparisons.

It calculates:

- Mean fold-level difference
- Paired t-test p-value
- Wilcoxon signed-rank p-value
- Paired Cohen's d

Comparisons include XGBoost against:

- Logistic Regression
- Random Forest
- SVM
- LightGBM

A separate McNemar's test is also used to compare calibrated XGBoost predictions with a tuned Logistic Regression baseline.

Reported McNemar test:

```text
p-value = 0.1185
```

---

## Explainable AI (XAI)

The notebook includes both global and local explainability.

### SHAP

SHAP is used for global feature importance and local explanations.

The analysis includes:

- Global SHAP feature importance
- Local waterfall explanations
- Patient-level explanations

### LIME

LIME is used for local patient-level explanations.

The notebook evaluates:

- Low-risk cases
- Medium-risk cases
- High-risk cases
- Repeated LIME runs
- Top-feature frequency
- Pairwise Jaccard stability

Reported average LIME stability:

```text
Mean Jaccard stability: 0.451
```

The reported average SHAP-LIME agreement is:

```text
Average overlap: 3.7 / 5 features
Average Jaccard agreement: 0.587
```

---

## Perturbation Sensitivity

The notebook tests local explanation robustness by changing selected ordinal input features by ±1 level while keeping the random seed fixed.

The reported overall mean Jaccard similarity is:

```text
0.483
```

Results are also exported for further analysis.

---

## Risk-Tier Analysis

Predicted probabilities are additionally grouped into three risk tiers:

```text
Low Risk:      < 0.3
Medium Risk:   0.3–0.6
High Risk:     > 0.6
```

The notebook creates a cross-tabulation of these tiers against the binary outcome and generates a risk-tier visualization.

---

## Decision Curve Analysis

Decision Curve Analysis (DCA) is included to investigate model net benefit across a range of probability thresholds.

The notebook uses the `dcurves` package and generates:

```text
decision_curve.png
```

---

## Ablation Analysis

The notebook includes a proxy-learning ablation analysis in which selected strong proxy predictors are removed and model performance is reassessed.

The results are saved as:

```text
proxy_learning_ablation_results.csv
```

This is intended to investigate whether model performance depends strongly on particular proxy variables.

---

## Subgroup and Fairness Analysis

The notebook includes subgroup performance analysis across:

- Gender
- Age groups
- Socioeconomic-status groups

Corresponding CSV files are generated:

```text
fairness_gender_subgroups.csv
fairness_age_subgroups.csv
fairness_ses_subgroups.csv
```

The analysis reports subgroup performance with confidence intervals.

---

## Generated Outputs

Depending on the executed cells, the notebook generates files such as:

```text
participant_characteristics_table.xlsx
lime_perturbation_sensitivity.csv
proxy_learning_ablation_results.csv
fairness_gender_subgroups.csv
fairness_age_subgroups.csv
fairness_ses_subgroups.csv
learning_curve.png
Figure_2A_Confusion_Matrix.png
precision_recall_curve.png
decision_curve.png
```

Additional curve/figure files are also produced by the visualization cells, including ROC and calibration outputs.

---

## Required Python Packages

The notebook imports the following major packages:

```text
pandas
numpy
scipy
scikit-learn
imbalanced-learn
xgboost
lightgbm
matplotlib
seaborn
shap
lime
statsmodels
dcurves
diptest
Pillow
```

The notebook installs some packages directly when needed, including:

```python
!pip install diptest
!pip install dcurves
```

A typical environment can be prepared with:

```bash
pip install pandas numpy scipy scikit-learn imbalanced-learn xgboost lightgbm matplotlib seaborn shap lime statsmodels dcurves diptest pillow
```

---

## How to Run

### 1. Clone/download the project

Place the following files in the same working directory:

```text
CKD_Pipeline.ipynb
ckd_data.xlsx
```

### 2. Install dependencies

```bash
pip install pandas numpy scipy scikit-learn imbalanced-learn xgboost lightgbm matplotlib seaborn shap lime statsmodels dcurves diptest pillow
```

### 3. Open the notebook

Using Jupyter:

```bash
jupyter notebook CKD_Pipeline.ipynb
```

Or open it in Google Colab and upload/mount `ckd_data.xlsx`.

### 4. Run the notebook sequentially

Run cells from top to bottom because later sections depend on variables, fitted models, and results created earlier in the notebook.

---

## Reproducibility

The notebook uses fixed random seeds in major stochastic procedures, including:

```text
random_state = 42
```

and the train-test split uses:

```text
random_state = 40
```

The pipeline uses stratified cross-validation for classification experiments.

---

## Main Pipeline Flow

```text
ckd_data.xlsx
      │
      ▼
Data Loading & Inspection
      │
      ▼
Column Cleaning
      │
      ▼
Feature Removal
      │
      ▼
Composite Risk Score (0–15)
      │
      ▼
K-Means Risk Labeling
      │
      ├──────────────► GMM Validation
      │
      ▼
Low Risk / High Risk
      │
      ▼
18 Predictive Features
      │
      ▼
Stratified Train/Test Split
      │
      ├──────────────► Class Weighting
      │
      └──────────────► SMOTE
      │
      ▼
LR / RF / XGBoost / SVM / LightGBM
      │
      ▼
Randomized Hyperparameter Search
      │
      ▼
Cross-Validation Evaluation
      │
      ▼
Model Selection
      │
      ▼
Isotonic Calibration
      │
      ▼
Training-CV Threshold Optimization
      │
      ▼
Final Test Evaluation
      │
      ├── ROC / PR / Calibration
      ├── Confusion Matrix
      ├── Nested CV
      ├── SHAP
      ├── LIME
      ├── Perturbation Analysis
      ├── Decision Curve Analysis
      ├── Ablation Analysis
      └── Subgroup/Fairness Analysis
```

---

## Notes and Limitations

- The binary target is **constructed from questionnaire-derived risk variables**, rather than being directly based on a clinically confirmed CKD diagnosis in the notebook.
- Several categorical variables are encoded numerically; their interpretation depends on the value-label mapping defined in the notebook.
- The notebook contains checks for undocumented category codes and reports whether such codes are present.
- Model performance reported here is specific to the supplied dataset and experimental setup.
- Cross-validation performance and test-set performance should not be interpreted as evidence of clinical effectiveness without independent external validation.
- The notebook performs multiple robustness and explainability analyses, but these do not replace prospective or external clinical validation.
- Predictions should not be used for diagnosis or treatment decisions without appropriate clinical validation and oversight.

---

## File Structure

A simple project layout is:

```text
.
├── CKD_Model.ipynb
├── ckd_data.xlsx
├── README.md
│
├── participant_characteristics_table.xlsx
├── lime_perturbation_sensitivity.csv
├── proxy_learning_ablation_results.csv
├── fairness_gender_subgroups.csv
├── fairness_age_subgroups.csv
├── fairness_ses_subgroups.csv
│
├── learning_curve.png
├── Figure_2A_Confusion_Matrix.png
├── precision_recall_curve.png
└── decision_curve.png
```

---


