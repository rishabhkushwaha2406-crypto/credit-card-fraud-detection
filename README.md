# Credit Card Fraud Detection

**Python · Pandas · Scikit-learn · imbalanced-learn · XGBoost**

A machine-learning pipeline that flags potentially fraudulent credit-card transactions, plus a live dashboard that compares the models and scores transactions in the browser.

**Live demo:** _add your Vercel or Netlify URL here_

## What it does

- **Cleans and preprocesses** transaction data: removes duplicates/missing rows, engineers `LogAmount` and `Hour`, standardises features, and uses a stratified 75/25 train/test split.
- **Handles extreme class imbalance** (~0.17% fraud) three ways, applied to training data only:
  - class weights (`class_weight="balanced"` / XGBoost `scale_pos_weight`)
  - **SMOTE** oversampling
  - random **undersampling** (1:10 ratio)
- **Trains and compares** Logistic Regression, Random Forest and XGBoost across all three strategies (9 pipelines).
- **Evaluates** with precision, recall, F1, ROC-AUC and PR-AUC, plus confusion matrices, so missed fraud cases are visible rather than hidden behind accuracy.

## Results (current run)

| Model | Strategy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | Class weights | 1.000 | 0.915 | **0.956** | 0.971 |
| Random Forest | SMOTE | 1.000 | 0.894 | 0.944 | 0.969 |
| XGBoost | Undersampling | 1.000 | 0.894 | 0.944 | 0.967 |
| XGBoost | SMOTE | 0.955 | 0.894 | 0.923 | 0.966 |
| Random Forest | Class weights | 1.000 | 0.830 | 0.907 | 0.963 |
| Logistic Regression | Undersampling | 0.896 | 0.915 | 0.905 | 0.966 |
| Random Forest | Undersampling | 0.932 | 0.872 | 0.901 | 0.948 |
| Logistic Regression | SMOTE | 0.115 | 0.915 | 0.204 | 0.961 |
| Logistic Regression | Class weights | 0.091 | 0.915 | 0.166 | 0.966 |

Key takeaway: every model scores ROC-AUC > 0.94, yet F1 ranges from 0.17 to 0.96. On highly imbalanced data, precision/recall and PR-AUC are the metrics that separate good fraud models from bad ones.

> **Data note:** these numbers come from a synthetic dataset generated with the same schema (`Time`, `V1`–`V28`, `Amount`, `Class`) and fraud rate as the [Kaggle ULB Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). To run on the real data, download `creditcard.csv` into `data/` and re-run the pipeline — the code picks it up automatically.

## Run it

```bash
pip install -r requirements.txt
python -m src.train      # trains 9 pipelines, saves models/ and reports/results.json
python web/build.py      # rebuilds the dashboard at web/index.html
```

## Project structure

```
src/data.py          loading, cleaning, feature engineering, synthetic fallback
src/train.py         imbalance strategies, model training, evaluation, exports
reports/results.json full metrics, ROC/PR curves, feature importance
web/template.html    dashboard source
web/build.py         inlines results into web/index.html
web/index.html       static site deployed to GitHub Pages
.github/workflows/   GitHub Pages deployment
```

The dashboard's transaction scorer runs the class-weighted logistic regression entirely in JavaScript using the exported scaler and coefficients — no backend needed.

## Deploy

The dashboard is a static site in `web/` with no build step.

- **Vercel:** import the repo; `vercel.json` sets the output directory to `web`. Click Deploy.
- **Netlify:** import the repo; `netlify.toml` publishes `web`. Click Deploy.
- **GitHub Pages:** Settings → Pages → Source: GitHub Actions (workflow included).
