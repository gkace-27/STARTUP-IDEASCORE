# StartupIdeaScore

A Flask web app that predicts startup success using a trained machine-learning model.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

## Deploy on Render

If the repository contains `StartupIdeaScore` as a subfolder, set the Render root directory to `StartupIdeaScore`. If the GitHub repository itself is this project folder, leave the root directory blank.

Use:

```text
Build command: pip install -r requirements.txt
Start command: gunicorn app:app
```

## Retrain

```bash
python train.py
```

The trainer compares Logistic Regression, Decision Tree, Random Forest, Extra Trees, and XGBoost, then saves the best pipeline to `model.pkl`.
