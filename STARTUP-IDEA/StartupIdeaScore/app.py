from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


def build_features(payload: dict) -> pd.DataFrame:
	funding_total = float(payload.get("funding_total_usd") or 0)
	funding_rounds = float(payload.get("funding_rounds") or 0)
	founded_year = int(payload.get("founded_year") or 0) or np.nan
	first_funding_year = int(payload.get("first_funding_year") or 0) or np.nan
	last_funding_year = int(payload.get("last_funding_year") or 0) or np.nan

	funding_per_round = funding_total / funding_rounds if funding_rounds else np.nan
	return pd.DataFrame([{
		"funding_total_usd": funding_total,
		"funding_rounds": funding_rounds,
		"log_funding": np.log1p(max(funding_total, 0)),
		"funding_per_round": funding_per_round,
		"log_funding_per_round": np.log1p(max(funding_per_round, 0)) if pd.notna(funding_per_round) else np.nan,
		"founded_year": founded_year,
		"first_funding_year": first_funding_year,
		"last_funding_year": last_funding_year,
		"startup_age_at_last_funding": last_funding_year - founded_year if pd.notna(last_funding_year) and pd.notna(founded_year) else np.nan,
		"funding_duration_years": last_funding_year - first_funding_year if pd.notna(last_funding_year) and pd.notna(first_funding_year) else np.nan,
		"main_category": (payload.get("category_list") or "Unknown").split("|")[0].strip(),
		"country_code": payload.get("country_code") or "Unknown",
		"state_code": payload.get("state_code") or "Unknown",
	}])


@app.get("/")
def home():
	return render_template("index.html")


@app.get("/health")
def health():
	return jsonify({"status": "ok", "model_loaded": model is not None})


@app.post("/predict")
def predict():
	payload = request.get_json(silent=True) or request.form.to_dict()
	try:
		features = build_features(payload)
		prediction = int(model.predict(features)[0])
		probability = float(model.predict_proba(features)[0][1])
		return jsonify({
			"success": True,
			"prediction": prediction,
			"label": "Successful" if prediction else "Failed",
			"probability": round(probability * 100, 2),
		})
	except (TypeError, ValueError, KeyError) as error:
		return jsonify({"success": False, "error": str(error)}), 400


if __name__ == "__main__":
	app.run(debug=True, host="127.0.0.1", port=5000)
