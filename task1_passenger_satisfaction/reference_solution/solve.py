# ============================================================================
# TASK 1: REFERENCE SOLUTION
# Run with:   python task1_passenger_satisfaction/reference_solution/solve.py
#
# A reference solution is the task author's own correct answer. It proves the
# task CAN be solved, and its score is used to set the pass mark.
#
# THE TWO DECISIONS THAT MATTER (and why)
#
#   1. DROP refund_issued (and the id columns).
#      Refunds are decided AFTER the flight, mostly for unhappy passengers.
#      In train.csv the column almost gives the answer away; in test.csv it's
#      always 0 because no refund exists yet at prediction time. A model that
#      learns "refund = unhappy" is using information from the future.
#      This is called DATA LEAKAGE.
#      passenger_id and flight_id are just labels, not information about the
#      flight. A model could use passenger_id to memorise people, but every
#      test passenger is new, so that can't help.
#
#   2. VALIDATE WITH GroupKFold, grouped by passenger.
#      The same person flies many times, and each person has their own
#      general mood. With a plain random split, one person's flights land in
#      both the training part and the checking part, so the model "recognises"
#      them and scores too well. GroupKFold keeps ALL of a passenger's flights
#      in the same fold, so every check is on people the model hasn't seen,
#      just like the real test of new customers.
#
#   3. USE A FIXED NUMBER OF TREES (no early stopping).
#      A first version used early stopping, and the folds stopped at very
#      different points (14 trees in one fold, 147 in another). Predictions
#      from such different models aren't on the same scale, so pooling them
#      made the estimate 0.017 too LOW, failing the task's own honesty check.
#      A fixed, gentle model (1000 small trees, learning rate 0.01) gives
#      stable folds. The settings were chosen by cross-validation score only,
#      never by looking at the hidden test answers.
#
# RESULT: own estimate 0.7548, real test AUC 0.7568 (estimate 0.0019 too low;
#         allowed range: 0.030 too low to 0.010 too high). Runs in about 20 seconds.
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
