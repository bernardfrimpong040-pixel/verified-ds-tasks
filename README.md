# Verified Data Science Tasks

- **Author:** Bernard Frimpong ([GitHub](https://github.com/bernardfrimpong040-pixel) · [LinkedIn](https://www.linkedin.com/in/bernard-frimpong-681008222))
- **Tools:** Python, pandas, NumPy, scikit-learn, LightGBM
- **Related:** [Kaggle Playground S6E10 project](https://github.com/bernardfrimpong040-pixel/kaggle-airline-satisfaction): public LB 0.96049 over five documented rounds

## What this is

A collection of data science tasks built so that **a plausible-looking answer is not enough**.
Each task has:

1. **A reproducible environment:** a script that rebuilds the data byte for byte from a fixed seed
2. **A verified reference solution:** the author's own correct answer, checked against hidden labels
3. **Reliable checks** (coming in Step 4): automatic tests that pass correct solutions and fail
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

## Repository structure

```
verified-ds-tasks/
├── README.md
├── requirements.txt               # pinned library versions
├── STEP3_GUIDE.md                 # how Task 1 was built and verified
└── task1_passenger_satisfaction/
    ├── TASK.md                    # the task statement given to the solver
    ├── environment/
    │   └── make_data.py           # rebuilds the data from a fixed seed
    ├── reference_solution/
    │   └── solve.py               # the verified reference solution
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
```

The data and the reference outputs are identical on every run.

## Roadmap

| Step | What | Status |
|---|---|---|
| 3 | Task 1 environment and reference solution | ✅ Done |
| 4 | Automatic checks with `pytest` | Next |
| 5 | Deliberately flawed solutions that the checks must catch | |
| 6 | Task 2: a numerical mathematics task | |
| 7 | Run an AI coding agent on the tasks and refine them | |
| 8 | Full documentation for each task | |
