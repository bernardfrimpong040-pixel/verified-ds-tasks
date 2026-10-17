# ============================================================================
# test_task1.py  —  THE AUTOMATIC CHECKS FOR TASK 1
#
# Run from the main project folder with:
#     pytest task1_passenger_satisfaction/tests -v
# To check a different solution:
#     pytest task1_passenger_satisfaction/tests -v --solution path/to/solve.py
#
# Each function whose name starts with "test_" is one check. pytest runs them
# all and reports PASSED or FAILED for each. A check fails when one of its
# "assert" lines is false; the message after the comma explains why.
#
# These checks are exactly the pass rules written in TASK.md:
#   Rule 1  format          -> test_predictions_file_*, test_estimate_file_*
#   Rule 2  accuracy        -> test_accuracy_meets_pass_mark
#   Rule 3  honest estimate -> test_estimate_not_overconfident,
#                              test_estimate_not_too_pessimistic
#   Rule 4  reproducible    -> test_two_runs_give_identical_files
# plus two checks on the environment itself (the data and the run).
# ============================================================================

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

TASK_DIR = Path(__file__).resolve().parent.parent
EXPECTED_DATA = json.loads((Path(__file__).parent / "expected_data.json").read_text())

PASS_MARK_AUC = 0.745            # Rule 2
MAX_OVERCONFIDENCE = 0.010       # Rule 3: estimate may be at most this much ABOVE the real AUC
MAX_PESSIMISM = 0.030            # Rule 3: estimate may be at most this much BELOW the real AUC
REQUIRED_COLUMNS = ["flight_id", "satisfied_probability"]


# ----------------------------------------------------------------------------
# Small helpers used by several checks
# ----------------------------------------------------------------------------

def first_run(solution_runs):
    """The first run's information; fails clearly if the solution crashed."""
    run = solution_runs["runs"][0]
    assert run["ok"], f"The solution did not run successfully:\n{run['error']}"
    return run


def read_predictions(solution_runs):
    path = first_run(solution_runs)["outputs"] / "predictions.csv"
    assert path.exists(), "The solution did not write outputs/predictions.csv"
    return pd.read_csv(path)


def read_estimate(solution_runs):
    path = first_run(solution_runs)["outputs"] / "cv_estimate.json"
    assert path.exists(), "The solution did not write outputs/cv_estimate.json"
    estimate = json.loads(path.read_text())
    assert isinstance(estimate, dict) and "auc" in estimate, \
        'cv_estimate.json must look like {"auc": 0.75}'
    return float(estimate["auc"])


@pytest.fixture(scope="module")
def real_auc(solution_runs):
    """The solution's real AUC on the hidden test answers (read only by the checks)."""
    predictions = read_predictions(solution_runs)
    assert list(predictions.columns) == REQUIRED_COLUMNS, (
        "The AUC can't be computed until predictions.csv has the right columns "
        "(see test_predictions_file_has_exact_columns)")
    answers = pd.read_csv(TASK_DIR / "data" / "hidden" / "test_labels.csv")
    joined = answers.merge(predictions, on="flight_id", how="left")
    assert joined["satisfied_probability"].notna().all(), \
        "Some test flights have no prediction, so the AUC can't be computed"
    return roc_auc_score(joined["satisfied"], joined["satisfied_probability"])


# ----------------------------------------------------------------------------
# The environment
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("relative_path", sorted(EXPECTED_DATA))
def test_data_matches_official_fingerprint(relative_path, solution_runs):
    """The data must be exactly what make_data.py produces (nothing edited or swapped)."""
    content = (TASK_DIR / relative_path).read_bytes()
    fingerprint = hashlib.sha256(content).hexdigest()
    assert fingerprint == EXPECTED_DATA[relative_path], (
        f"{relative_path} is not the official data. "
        "Delete the data folder and run environment/make_data.py again.")


def test_solution_runs_without_error(solution_runs):
    """The solution must finish without crashing (with the hidden answers out of reach)."""
    run = first_run(solution_runs)
    print(f"solution ran in {run['seconds']:.1f} seconds")


# ----------------------------------------------------------------------------
# Rule 1: file format
# ----------------------------------------------------------------------------

def test_predictions_file_has_exact_columns(solution_runs):
    predictions = read_predictions(solution_runs)
    assert list(predictions.columns) == REQUIRED_COLUMNS, (
        f"Columns must be exactly {REQUIRED_COLUMNS}, found {list(predictions.columns)}")


def test_predictions_file_has_every_test_flight_once(solution_runs):
    predictions = read_predictions(solution_runs)
    test_ids = pd.read_csv(TASK_DIR / "data" / "public" / "test.csv", usecols=["flight_id"])
    assert not predictions["flight_id"].duplicated().any(), "Some flight_ids appear twice"
    missing = set(test_ids["flight_id"]) - set(predictions["flight_id"])
    extra = set(predictions["flight_id"]) - set(test_ids["flight_id"])
    assert not missing, f"{len(missing)} test flights have no prediction"
    assert not extra, f"{len(extra)} flight_ids are not in test.csv"


def test_predictions_file_has_valid_probabilities(solution_runs):
    probabilities = read_predictions(solution_runs)["satisfied_probability"]
    assert probabilities.notna().all(), "Some predictions are empty"
    assert probabilities.between(0, 1).all(), "Every probability must be between 0 and 1"
    assert probabilities.nunique() > 1, "All predictions are the same value"


def test_estimate_file_is_valid(solution_runs):
    estimate = read_estimate(solution_runs)
    assert np.isfinite(estimate) and 0.5 <= estimate <= 1.0, (
        f"The AUC estimate must be a number between 0.5 and 1.0, found {estimate}")


# ----------------------------------------------------------------------------
# Rule 2: accuracy
# ----------------------------------------------------------------------------

def test_accuracy_meets_pass_mark(real_auc):
    print(f"real test AUC: {real_auc:.4f} (pass mark {PASS_MARK_AUC})")
    assert real_auc >= PASS_MARK_AUC, (
        f"Test AUC is {real_auc:.4f}, below the pass mark of {PASS_MARK_AUC}")


# ----------------------------------------------------------------------------
# Rule 3: honest estimate
# ----------------------------------------------------------------------------

def test_estimate_not_overconfident(solution_runs, real_auc):
    """The most important check: flawed validation always OVERestimates."""
    estimate = read_estimate(solution_runs)
    gap = estimate - real_auc
    print(f"estimate {estimate:.4f}, real {real_auc:.4f}, gap {gap:+.4f} (limit +{MAX_OVERCONFIDENCE})")
    assert gap <= MAX_OVERCONFIDENCE, (
        f"Overconfident: estimated {estimate:.4f} but really scored {real_auc:.4f} "
        f"({gap:+.4f}, more than {MAX_OVERCONFIDENCE} too high). This usually means the "
        "validation let the model see the same passengers twice, or used information "
        "that isn't available at prediction time.")


def test_estimate_not_too_pessimistic(solution_runs, real_auc):
    estimate = read_estimate(solution_runs)
    gap = estimate - real_auc
    assert gap >= -MAX_PESSIMISM, (
        f"Too pessimistic: estimated {estimate:.4f} but really scored {real_auc:.4f} "
        f"({gap:+.4f}, more than {MAX_PESSIMISM} too low)")


# ----------------------------------------------------------------------------
# Rule 4: reproducible
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("file_name", ["predictions.csv", "cv_estimate.json"])
def test_two_runs_give_identical_files(solution_runs, file_name):
    first_run(solution_runs)
    runs = solution_runs["runs"]
    assert len(runs) == 2 and runs[1]["ok"], f"The second run failed:\n{runs[-1]['error']}"
    first = (runs[0]["outputs"] / file_name).read_bytes()
    second = (runs[1]["outputs"] / file_name).read_bytes()
    assert first == second, (
        f"{file_name} was different on the second run. Fix every random seed "
        "(and use n_jobs=1 / deterministic settings if needed).")
