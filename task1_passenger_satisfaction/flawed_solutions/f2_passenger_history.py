# ============================================================================
# FLAWED SOLUTION: PASSENGER HISTORY FEATURE (TARGET LEAKAGE)
#
# This is a copy of the reference solution with ONE deliberate mistake.
# It exists to prove that the checks in tests/ catch this mistake.
# DO NOT use it as a real solution.
#
# THE MISTAKE
#   Adds a 'passenger_history' feature: each passenger's average satisfaction
#   over ALL their training flights, including the flight being predicted.
#   Test passengers have no history, so they get the overall average.
#
# WHY IT'S WRONG
#   The feature is built from the answers themselves, including the row's own
#   answer, so it leaks the target into the features. Validation looks
#   excellent, but new customers have no history, so it's useless on the test.
#
# EXPECTED TO BE CAUGHT BY: test_estimate_not_overconfident
#
# Every changed line is marked with:  # <-- MISTAKE
# ============================================================================

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

TASK_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = TASK_DIR / "data" / "public"
# Where to write results: the outputs folder, unless another folder is given
# on the command line (the tests in Step 4 use this)
OUTPUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else TASK_DIR / "outputs"

SEED = 42
N_FOLDS = 5
TARGET = "satisfied"

# Columns we will NOT use, with the reason for each
NOT_FEATURES = {
    "flight_id": "an id, not information",
    "passenger_id": "an id; test passengers are all new, so memorising people can't help",
    "refund_issued": "decided AFTER the flight (data leakage); always 0 in test",
    TARGET: "the answer",
}
CATEGORICAL = ["home_airport", "loyalty_tier", "travel_type", "cabin_class"]

MODEL_SETTINGS = {
    "n_estimators": 1000,       # fixed number of trees in every fold (see decision 3)
    "learning_rate": 0.01,      # small, careful steps
    "num_leaves": 7,            # small trees: less chance to memorise noise
    "min_child_samples": 100,   # each leaf must contain at least 100 flights
    "cat_smooth": 50,           # be careful with rare airport codes
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "random_state": SEED, "n_jobs": 1,   # n_jobs=1 keeps results exactly repeatable
    "deterministic": True, "force_row_wise": True,
    "verbose": -1,
}


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    features = [c for c in train.columns if c not in NOT_FEATURES]
    print("Using features:", ", ".join(features))
    for col, reason in NOT_FEATURES.items():
        if col != TARGET:
            print(f"   not using {col}: {reason}")

    # Word columns: same list of possible values in train and test
    for col in CATEGORICAL:
        kind = pd.CategoricalDtype(sorted(set(train[col]) | set(test[col])))
        train[col] = train[col].astype(kind)
        test[col] = test[col].astype(kind)

    history = train.groupby("passenger_id")[TARGET].mean()  # <-- MISTAKE
    train["passenger_history"] = train["passenger_id"].map(history)  # <-- MISTAKE
    test["passenger_history"] = train[TARGET].mean()   # new customers: no history  # <-- MISTAKE
    features = features + ["passenger_history"]  # <-- MISTAKE
    X, y, groups = train[features], train[TARGET], train["passenger_id"]
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))

    # GroupKFold: every passenger's flights stay together in one fold
    folds = GroupKFold(n_splits=N_FOLDS)
    for fold, (fit_rows, check_rows) in enumerate(folds.split(X, y, groups), start=1):
        model = lgb.LGBMClassifier(**MODEL_SETTINGS)
        model.fit(X.iloc[fit_rows], y.iloc[fit_rows])
        oof[check_rows] = model.predict_proba(X.iloc[check_rows])[:, 1]
        test_pred += model.predict_proba(test[features])[:, 1] / N_FOLDS

        # Safety check: no passenger may be in both parts of a fold
        overlap = set(groups.iloc[fit_rows]) & set(groups.iloc[check_rows])
        assert not overlap, "a passenger appears on both sides of a fold"
        print(f"fold {fold}: AUC {roc_auc_score(y.iloc[check_rows], oof[check_rows]):.4f}")

    cv_auc = roc_auc_score(y, oof)
    print(f"Cross-validation AUC on unseen passengers: {cv_auc:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"flight_id": test["flight_id"],
                  "satisfied_probability": np.round(test_pred, 6)}).to_csv(
        OUTPUT_DIR / "predictions.csv", index=False, lineterminator="\n")
    with open(OUTPUT_DIR / "cv_estimate.json", "w") as f:
        json.dump({"auc": round(float(cv_auc), 6)}, f)
    print(f"Wrote predictions.csv and cv_estimate.json to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
