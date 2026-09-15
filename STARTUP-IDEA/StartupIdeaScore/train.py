import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "big_startup_secsees_dataset.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
MODEL_INFO_PATH = BASE_DIR / "model_info.pkl"

RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH, na_values=["-"])

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "status",
    "funding_total_usd",
    "funding_rounds",
    "category_list",
    "country_code",
    "state_code",
    "founded_at",
    "first_funding_at",
    "last_funding_at"
]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Required column missing: {column}"
        )


# ============================================================
# 5. REMOVE DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates()

after = len(df)

print(
    f"Removed duplicates: {before - after}"
)


# ============================================================
# 6. CREATE TARGET
# ============================================================

print("\nCreating target...")

# SUCCESS
# acquired = 1
# ipo = 1

# FAILURE
# closed = 0

df["target"] = df["status"].map({
    "acquired": 1,
    "ipo": 1,
    "closed": 0
})


# Remove operating startups
df = df.dropna(subset=["target"])

df["target"] = df["target"].astype(int)


print("\nTarget distribution:")

print(
    df["target"]
    .value_counts()
    .rename({
        0: "Failed",
        1: "Successful"
    })
)


# ============================================================
# 7. CONVERT NUMERIC COLUMNS
# ============================================================

df["funding_total_usd"] = pd.to_numeric(
    df["funding_total_usd"],
    errors="coerce"
)

df["funding_rounds"] = pd.to_numeric(
    df["funding_rounds"],
    errors="coerce"
)


# ============================================================
# 8. DATE FEATURES
# ============================================================

print("\nCreating date features...")


df["founded_at"] = pd.to_datetime(
    df["founded_at"],
    errors="coerce"
)

df["first_funding_at"] = pd.to_datetime(
    df["first_funding_at"],
    errors="coerce"
)

df["last_funding_at"] = pd.to_datetime(
    df["last_funding_at"],
    errors="coerce"
)


# Year founded
df["founded_year"] = (
    df["founded_at"].dt.year
)


# Year of first funding
df["first_funding_year"] = (
    df["first_funding_at"].dt.year
)


# Year of last funding
df["last_funding_year"] = (
    df["last_funding_at"].dt.year
)


# Startup age when last funding happened
df["startup_age_at_last_funding"] = (
    df["last_funding_year"]
    - df["founded_year"]
)


# Years between first and last funding
df["funding_duration_years"] = (
    df["last_funding_year"]
    - df["first_funding_year"]
)


# ============================================================
# 9. FUNDING FEATURES
# ============================================================

# Log transformation reduces the effect of very large
# funding values.

df["log_funding"] = np.log1p(
    df["funding_total_usd"].clip(lower=0)
)


# Average funding per round

df["funding_per_round"] = (
    df["funding_total_usd"]
    /
    df["funding_rounds"].replace(0, np.nan)
)


df["log_funding_per_round"] = np.log1p(
    df["funding_per_round"].clip(lower=0)
)


# ============================================================
# 10. CATEGORY CLEANING
# ============================================================

df["category_list"] = (
    df["category_list"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)


# Take the first category.
# Example:
# "Software|Analytics|SaaS"
# becomes
# "Software"

df["main_category"] = (
    df["category_list"]
    .str.split("|")
    .str[0]
    .str.strip()
)


# ============================================================
# 11. CLEAN COUNTRY / STATE
# ============================================================

for column in [
    "country_code",
    "state_code"
]:

    df[column] = (
        df[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )


# ============================================================
# 12. SELECT FINAL FEATURES
# ============================================================

features = [

    # Funding
    "funding_total_usd",
    "funding_rounds",
    "log_funding",
    "funding_per_round",
    "log_funding_per_round",

    # Time
    "founded_year",
    "first_funding_year",
    "last_funding_year",
    "startup_age_at_last_funding",
    "funding_duration_years",

    # Business information
    "main_category",
    "country_code",
    "state_code"
]


X = df[features].copy()

y = df["target"]


print("\nFinal features:")

for feature in features:
    print(" -", feature)


# ============================================================
# 13. REMOVE INVALID VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 14. TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y
)


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# 15. DEFINE FEATURE TYPES
# ============================================================

numeric_features = [

    "funding_total_usd",
    "funding_rounds",
    "log_funding",
    "funding_per_round",
    "log_funding_per_round",
    "founded_year",
    "first_funding_year",
    "last_funding_year",
    "startup_age_at_last_funding",
    "funding_duration_years"
]


categorical_features = [

    "main_category",
    "country_code",
    "state_code"
]


# ============================================================
# 16. NUMERIC PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),

    (
        "scaler",
        StandardScaler()
    )

])


# ============================================================
# 17. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),

    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=5
        )
    )

])


# ============================================================
# 18. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer([

    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),

    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )

])


# ============================================================
# 19. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression":

        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),


    "Decision Tree":

        DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),


    "Random Forest":

        RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),


    "Extra Trees":

        ExtraTreesClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    ,

    "XGBoost":

        XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=3,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

}


# ============================================================
# 20. TRAIN MODELS
# ============================================================

results = []

best_model = None
best_model_name = ""
best_accuracy = 0


print("\n")
print("=" * 65)
print("MODEL TRAINING")
print("=" * 65)


for model_name, model in models.items():

    print("\n")
    print("-" * 65)
    print("Training:", model_name)
    print("-" * 65)


    pipeline = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )

    ])


    # Train
    pipeline.fit(
        X_train,
        y_train
    )


    # Predict
    predictions = pipeline.predict(
        X_test
    )


    # Probability
    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]


    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )


    # ROC-AUC
    auc = roc_auc_score(
        y_test,
        probabilities
    )


    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "ROC_AUC": auc

    })


    print(
        "Accuracy:",
        round(accuracy * 100, 2),
        "%"
    )


    print(
        "ROC-AUC:",
        round(auc, 4)
    )


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Failed",
                "Successful"
            ]
        )
    )


    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )


    # Select best model
    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = pipeline

        best_model_name = model_name


# ============================================================
# 21. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)


print("\n")
print("=" * 65)
print("MODEL COMPARISON")
print("=" * 65)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 22. BEST MODEL
# ============================================================

print("\n")
print("=" * 65)
print("BEST MODEL")
print("=" * 65)


print(
    "Model:",
    best_model_name
)


print(
    "Test Accuracy:",
    round(
        best_accuracy * 100,
        2
    ),
    "%"
)


# ============================================================
# 23. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    MODEL_PATH
)


print("\n")
print("=" * 65)
print("MODEL SAVED")
print("=" * 65)


print(
    "Saved as:",
    MODEL_PATH
)


# ============================================================
# 24. SAVE FEATURE INFORMATION
# ============================================================

model_info = {

    "model_name":
        best_model_name,

    "accuracy":
        float(best_accuracy),

    "features":
        features,

    "target":
        "1 = Successful, 0 = Failed",

    "success_status":
        [
            "acquired",
            "ipo"
        ],

    "failure_status":
        [
            "closed"
        ]

}


joblib.dump(
    model_info,
    MODEL_INFO_PATH
)


print(
    "Saved:",
    "model_info.pkl"
)


# ============================================================
# 25. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 65)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 65)

print(
    f"\nBest Model: {best_model_name}"
)

print(
    f"Accuracy: {best_accuracy * 100:.2f}%"
)

print(
    "\nReady for FastAPI integration."
)
