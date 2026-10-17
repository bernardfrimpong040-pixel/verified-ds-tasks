# Step 4 Guide: Automatic Checks with pytest

**Goal:** Turn the pass rules in `TASK.md` into automatic tests that pass correct solutions and fail flawed ones.
**Files:** `task1_passenger_satisfaction/tests/conftest.py`, `test_task1.py`, `expected_data.json`
**Result:** the reference passes all **13** checks; four deliberately bad scripts were each caught for the right reason

---

## Part 1: What pytest is

**pytest** is Python's standard testing tool. You write small functions whose names start with
`test_`, each containing `assert` lines:

```python
def test_accuracy_meets_pass_mark(real_auc):
    assert real_auc >= 0.745, "Test AUC is below the pass mark"
```

`assert` means "this must be true". If it is, the test **PASSES**. If not, it **FAILS** and
prints the message after the comma. You run every test with one command:

```powershell
pytest task1_passenger_satisfaction/tests -v
```

(`-v` means "verbose": show each test's name and result.)

## Part 2: How the checks work

**`conftest.py`: the setup.** pytest reads it first. It:

1. Adds a `--solution` option, so any solution can be tested (the default is the reference).
2. Builds the data if it's missing.
3. Runs the solution **twice**, keeping a copy of each run's output files.
4. **Moves the hidden answers out of the task folder while the solution runs**, then puts them back.
   A solution that tries to read them crashes, so it cannot cheat, even by accident.

The two runs happen **once**, however many tests use them (pytest calls this a *session-scoped fixture*).

**`test_task1.py`: the 13 checks**, matching the rules in `TASK.md`:

| # | Check | Rule | What it verifies |
|---|---|---|---|
| 1–3 | `test_data_matches_official_fingerprint` | Environment | Each data file's SHA-256 fingerprint matches `expected_data.json`, so the data hasn't been edited or swapped |
| 4 | `test_solution_runs_without_error` | Environment | The solution finishes (within 10 minutes) with the hidden answers out of reach |
| 5 | `test_predictions_file_has_exact_columns` | 1. Format | Columns are exactly `flight_id, satisfied_probability` |
| 6 | `test_predictions_file_has_every_test_flight_once` | 1. Format | No missing, extra or duplicated flights |
| 7 | `test_predictions_file_has_valid_probabilities` | 1. Format | No blanks, all between 0 and 1, not all the same |
| 8 | `test_estimate_file_is_valid` | 1. Format | `cv_estimate.json` holds an AUC between 0.5 and 1 |
| 9 | `test_accuracy_meets_pass_mark` | 2. Accuracy | Real test AUC ≥ 0.745 |
| 10 | `test_estimate_not_overconfident` | 3. Honesty | Estimate at most **0.010 above** the real AUC |
| 11 | `test_estimate_not_too_pessimistic` | 3. Honesty | Estimate at most 0.030 below the real AUC |
| 12–13 | `test_two_runs_give_identical_files` | 4. Reproducible | Both output files are byte-identical across the two runs |

**What is a fingerprint (SHA-256)?** A 64-character code computed from a file's exact contents.
Change even one character and the code changes completely. Comparing codes is a quick way to
prove two files are identical.

## Part 3: A fix this step needed

Windows and Linux end lines of text differently (`\r\n` vs `\n`). The data written on your laptop
would therefore have had a **different fingerprint** from mine, and check 1–3 would fail for no
real reason. `make_data.py` and `solve.py` now always write `\n` line endings
(`lineterminator="\n"`), so the files are byte-for-byte identical on every computer.

**This means you must rebuild the data once** (see Part 5).

## Part 4: Proving the checks work

A test suite that passes the reference is only half the proof. It must also **fail** bad solutions,
and for the right reason. Four deliberately bad scripts were tried:

| Solution | Passed | What failed | Why that's right |
|---|---:|---|---|
| **Reference** | **13 / 13** | — | |
| Random K-fold validation | 12 / 13 | **Not overconfident** only | Its predictions are fine, but it overestimates itself. Only the honesty check can catch this |
| Reads the hidden answers | Crashed | **Runs without error**, then everything after | The answers were out of reach, so the cheat had nothing to read |
| Random seed from the clock | 11 / 13 | **Reproducible** ×2 | Two runs gave different predictions |
| Wrong column name | Failed | **Format** checks | Clear message: "Columns must be exactly ..." |

The random-K-fold result is the most important: it passed **12 of 13** checks. Without the
honesty check, it would have been accepted. That's why the task asks for an estimate at all.

The hidden answers were put back correctly after every run, including the cheating one.

(These four scripts were quick experiments for this step. Step 5 builds a proper, documented
set of flawed solutions as part of the project.)

## Part 5: How to run it on your laptop

**1. Save the new and updated files** (see the table in the chat message).

**2. Install pytest** (with `(.venv)` showing):
```powershell
pip install -r requirements.txt
```

**3. Rebuild the data** (line endings changed, so the old files' fingerprints won't match):
```powershell
python task1_passenger_satisfaction/environment/make_data.py
```

**4. Run the checks** (about 45 seconds, because the solution runs twice):
```powershell
pytest task1_passenger_satisfaction/tests -v
```
✅ *You should see:* 13 lines ending in `PASSED`, then `13 passed`.

To also see the scores, add `-s`:
```powershell
pytest task1_passenger_satisfaction/tests -v -s
```
✅ *You should see:* `real test AUC: 0.7568` and `estimate 0.7548, real 0.7568, gap -0.0019`.

**5. Update GitHub:**
```powershell
git add .
git commit -m "Step 4: automatic pytest checks for Task 1"
git push
```

---

## New words in this step

| Word | Plain meaning |
|---|---|
| **pytest** | Python's standard tool for running automatic checks |
| **Test** | A function starting with `test_` that passes if all its `assert` lines are true |
| **assert** | "This must be true"; if not, the test fails with the given message |
| **Fixture** | Setup code that tests share, e.g. running the solution once for all checks |
| **Session scope** | The fixture runs once per `pytest` command, not once per test |
| **Parametrize** | Running the same test several times with different inputs (e.g. each data file) |
| **SHA-256 fingerprint** | A code computed from a file's exact contents; any change gives a different code |
| **PASSED / FAILED / ERROR** | Check succeeded / an `assert` was false / the setup for that check failed |
| **Line endings** | The invisible characters ending each line of text; Windows and Linux differ |
