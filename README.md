# Verified Data Science Tasks

- **Author:** Bernard Frimpong ([GitHub](https://github.com/bernardfrimpong040-pixel) · [LinkedIn](https://www.linkedin.com/in/bernard-frimpong-681008222))
- **Tools:** Python, pandas, NumPy, scikit-learn, LightGBM, pytest
- **Related:** [Kaggle Playground S6E10 project](https://github.com/bernardfrimpong040-pixel/kaggle-airline-satisfaction): public LB 0.96049 over five documented rounds

## What this is

A collection of data science tasks built so that **a plausible-looking answer is not enough**.
Each task has:

1. **A reproducible environment:** a script that rebuilds the data byte for byte from a fixed seed
2. **A verified reference solution:** the author's own correct answer, checked against hidden labels
3. **Reliable checks:** automatic `pytest` tests that pass correct solutions and fail
   incomplete or flawed ones
4. **Clear documentation:** the task statement, the reasoning behind the solution, and how it is verified

These tasks are designed for evaluating data scientists and AI coding agents.

## Tasks

| Task | Skill tested | Traps | Reference result |
|---|---|---|---|
| [Task 1: Passenger satisfaction for new customers](task1_passenger_satisfaction/TASK.md) | Validation design, leakage detection | Repeat passengers (grouped data); a column recorded after the outcome | Test AUC 0.7568, own estimate 0.7548 (gap −0.0019) |

## Task 1 at a glance

The data contains frequent flyers (15–30 flights each) with a hidden personal mood, and a
`refund_issued` column decided **after** the flight. The test set contains only **new** passengers.
A solution must reach a test AUC of at least **0.745** and estimate its own score honestly:
**at most 0.010 too high**, at most 0.030 too low.

Every approach below was run with two model settings (the reference's careful settings and
LightGBM's defaults) and judged by those rules:

| Approach | Careful: estimate gap | Default: estimate gap | Result |
|---|---:|---:|---|
| **GroupKFold, no ids, no refund (reference)** | **−0.0019** | **−0.0077** | ✅ **passes** |
| Random K-fold | +0.0201 | +0.0460 | ❌ overconfident |
| Random K-fold + passenger_id | +0.0340 | +0.0931 | ❌ overconfident (default also below 0.745) |
| GroupKFold + `refund_issued` | +0.1396 | +0.1395 | ❌ overconfident |
| Random K-fold + passenger_id + refund | +0.1531 | +0.1820 | ❌ overconfident (default also below 0.745) |

The checks separate correct from flawed approaches with a clear margin: honest solutions
underestimate slightly, flawed ones overestimate by at least 0.020.

### The automatic checks (`tests/`)

13 `pytest` checks, one per rule in `TASK.md`, plus checks on the environment:

| Check | What it verifies |
|---|---|
| Data fingerprint (×3) | The data files are byte-for-byte what `make_data.py` produces (SHA-256) |
| Runs without error | The solution finishes, with the hidden answers **moved out of reach** while it runs |
| Format (×4) | Exact columns, every test flight once, probabilities in [0, 1], valid estimate file |
| Accuracy | Test AUC ≥ 0.745 |
| Not overconfident | Estimate is at most 0.010 above the real test AUC |
| Not too pessimistic | Estimate is at most 0.030 below the real test AUC |
| Reproducible (×2) | Two runs give byte-identical output files |

### Flawed solutions (`flawed_solutions/`)

Eight realistic mistakes, each a copy of the reference with **one** change (marked `# <-- MISTAKE`)
and the check that must catch it. `check_flawed_solutions.py` runs the full test suite on each and
confirms it is caught **by the intended check**. A crash doesn't count as caught.

| Solution | Mistake | Checks passed | Caught by | Verdict |
|---|---|---:|---|---|
| **Reference** | — | **13/13** | — | ✅ passes |
| f1 | Random K-fold instead of GroupKFold | 12/13 | Not overconfident (estimate +0.020 too high) | ✅ caught |
| f2 | Passenger-history feature built from the answers | 12/13 | Not overconfident (estimate 0.92 vs real 0.76) | ✅ caught |
| f3 | Uses `refund_issued` | 12/13 | Not overconfident (+0.140) | ✅ caught |
| f4 | Reads the hidden test answers | 3/13 | Runs without error (answers out of reach) | ✅ caught |
| f5 | Folds shuffled without a seed | 11/13 | Reproducible | ✅ caught |
| f6 | Writes 0/1 labels instead of probabilities | 11/13 | Accuracy (real AUC 0.685) | ✅ caught |
| f7 | Reports the training score as the estimate | 12/13 | Not overconfident (+0.030) | ✅ caught |
| f8 | Wrong column name | 8/13 | Format | ✅ caught |

Five of the eight mistakes pass 11 or 12 of the 13 checks; a single targeted check stops each one.
Full results: [`flawed_solutions/RESULTS.md`](task1_passenger_satisfaction/flawed_solutions/RESULTS.md).

## Repository structure

```
verified-ds-tasks/
├── README.md
├── requirements.txt               # pinned library versions
├── STEP3_GUIDE.md                 # how Task 1 was built and verified
├── STEP4_GUIDE.md                 # the automatic checks and how they were proven
├── STEP5_GUIDE.md                 # flawed solutions and the checker
└── task1_passenger_satisfaction/
    ├── TASK.md                    # the task statement given to the solver
    ├── environment/
    │   └── make_data.py           # rebuilds the data from a fixed seed
    ├── reference_solution/
    │   └── solve.py               # the verified reference solution
    ├── check_flawed_solutions.py  # runs every flawed solution through the tests
    ├── flawed_solutions/          # 8 one-mistake copies of the reference + RESULTS.md
    ├── tests/
    │   ├── conftest.py            # runs the solution twice, hidden answers out of reach
    │   ├── test_task1.py          # the 13 checks
    │   └── expected_data.json     # official data fingerprints
    ├── data/                      # created by make_data.py (not on GitHub)
    │   ├── public/                #   train.csv, test.csv: what the solver sees
    │   └── hidden/                #   test_labels.csv: only the checks may read this
    └── outputs/                   # created by solve.py (not on GitHub)
```

## Reproducing Task 1

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python task1_passenger_satisfaction/environment/make_data.py
python task1_passenger_satisfaction/reference_solution/solve.py
pytest task1_passenger_satisfaction/tests -v
```

To check any other solution: `pytest task1_passenger_satisfaction/tests -v --solution path/to/solve.py`

To confirm every flawed solution is caught: `python task1_passenger_satisfaction/check_flawed_solutions.py`

The data and the reference outputs are identical on every run.

## Roadmap

| Step | What | Status |
|---|---|---|
| 3 | Task 1 environment and reference solution | ✅ Done |
| 4 | Automatic checks with `pytest` | ✅ Done |
| 5 | Deliberately flawed solutions that the checks must catch | ✅ Done |
| 6 | Task 2: a numerical mathematics task | Next |
| 7 | Run an AI coding agent on the tasks and refine them | |
| 8 | Full documentation for each task | |
