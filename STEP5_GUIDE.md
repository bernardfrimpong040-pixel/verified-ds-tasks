# Step 5 Guide: Flawed Solutions That the Checks Must Catch

**Goal:** Build a documented set of realistic mistakes and prove that the checks catch every one, for the right reason.
**Files:** `task1_passenger_satisfaction/flawed_solutions/` (8 scripts + `RESULTS.md`) and `check_flawed_solutions.py`
**Result:** the reference passes 13/13; **all 8 flawed solutions are caught by the intended check**

---

## Part 1: Why build wrong solutions on purpose?

The posting asks for checks that "reliably distinguish correct solutions from incomplete or
flawed approaches." The only way to show that is to **try** flawed approaches and watch the checks fail them.

Testers call this **mutation testing**: make small deliberate mistakes ("mutants") and confirm
the tests "kill" each one. If a mutant survives, the tests have a blind spot.

**Two rules made the evidence trustworthy:**

1. **One change per solution.** Each flawed solution is a copy of the reference with **exactly
   one** mistake, every changed line marked `# <-- MISTAKE`. If it's caught, that one mistake is the reason.
2. **Caught by the right check.** Each file names the check that should catch it
   (`EXPECTED TO BE CAUGHT BY:`). Failing *some* check isn't enough.

## Part 2: The 8 mistakes

| File | Mistake | Real-world version | Should be caught by |
|---|---|---|---|
| `f1_random_kfold.py` | Plain random K-fold instead of GroupKFold | Forgetting that rows belong to the same customer, patient or device | Not overconfident |
| `f2_passenger_history.py` | Feature = each passenger's average satisfaction, including the row's own answer | "Customer history" features built from the outcome | Not overconfident |
| `f3_uses_refund.py` | Keeps `refund_issued` | Using data recorded after the event you're predicting | Not overconfident |
| `f4_reads_hidden_answers.py` | Reads the hidden test answers | Peeking at the test set, by accident or on purpose | Runs without error |
| `f5_unseeded_folds.py` | `GroupKFold(shuffle=True)` with no seed | Results nobody can reproduce | Reproducible |
| `f6_hard_labels.py` | Writes 0/1 instead of probabilities | Using `predict` where `predict_proba` was needed | Accuracy |
| `f7_training_score_as_estimate.py` | Reports the AUC on the training data | "Marking its own homework" | Not overconfident |
| `f8_wrong_column_name.py` | Column named `probability` | Not following the output specification | Format (exact columns) |

## Part 3: The checker (`check_flawed_solutions.py`)

One command runs the whole test suite on the reference and on every flawed solution:

```powershell
python task1_passenger_satisfaction/check_flawed_solutions.py
```

For each solution it:
1. Runs `pytest` with `--solution` pointing at that file.
2. Reads pytest's **XML report** (`--junitxml`) to get the exact list of failed checks. This is more
   reliable than reading the screen output.
3. Compares the failures with the file's `EXPECTED TO BE CAUGHT BY:` line.
4. Gives a verdict: **PASSES** (reference), **CAUGHT**, **caught, but by a different check**,
   **NOT CAUGHT**, or **CRASHED: proves nothing**.
5. Saves the table to `flawed_solutions/RESULTS.md`, so the evidence is on GitHub.

It takes about 5 minutes (9 solutions, each run twice by the tests).

## Part 4: A bug the checker itself had

The first run reported **"ALL GOOD"**, but two rows didn't look right: `f6_hard_labels` and
`f8_wrong_column_name` had **crashed**. My `# <-- MISTAKE` marker had been placed in the middle of a
line, which turned the rest of the line into a comment and broke the Python code.

The checker had still called them **CAUGHT**, because when a solution crashes, *every* check fails,
including the expected one. A crash proves nothing about the accuracy check or the format check.

**Two fixes:**
1. **The flawed files:** markers moved to the ends of lines, and the builder now **compiles** every
   flawed file to catch syntax errors before anything runs.
2. **The checker:** if a solution crashes when crashing isn't the mistake being tested, the verdict
   is now **"CRASHED: proves nothing"**, never CAUGHT.

**Lesson:** a test result that looks too good still needs checking. "Everything passed" and
"everything failed" are both worth a second look.

## Part 5: The results

| Solution | Checks passed | Failed checks | Verdict |
|---|---:|---|---|
| **Reference** | **13/13** | none | ✅ PASSES |
| f1 Random K-fold | 12/13 | Not overconfident | ✅ CAUGHT |
| f2 Passenger history | 12/13 | Not overconfident | ✅ CAUGHT |
| f3 Uses refund | 12/13 | Not overconfident | ✅ CAUGHT |
| f4 Reads hidden answers | 3/13 | Runs without error (and everything after) | ✅ CAUGHT |
| f5 Unseeded folds | 11/13 | Reproducible ×2 | ✅ CAUGHT |
| f6 0/1 labels | 11/13 | Accuracy, not overconfident | ✅ CAUGHT |
| f7 Training score as estimate | 12/13 | Not overconfident | ✅ CAUGHT |
| f8 Wrong column name | 8/13 | Format, and the checks that need the right column | ✅ CAUGHT |

**The numbers behind the honesty and accuracy failures:**

| Solution | Its own estimate | Real test AUC | Gap (limit +0.010) |
|---|---:|---:|---:|
| **Reference** | **0.7548** | **0.7568** | **−0.0019** ✅ |
| f1 Random K-fold | 0.7766 | 0.7565 | +0.0201 |
| f2 Passenger history | 0.9238 | 0.7559 | **+0.1679** |
| f3 Uses refund | 0.8956 | 0.7559 | +0.1396 |
| f6 0/1 labels | 0.7548 | **0.6853** (below 0.745) | +0.0695 |
| f7 Training score | 0.7868 | 0.7568 | +0.0301 |

**What stands out:**
- **Five of the eight mistakes still pass 11 or 12 of the 13 checks.** Their predictions look fine;
  only one check stops them. A reviewer who only looked at the format or the accuracy would accept them.
- **The honesty check catches four different mistakes** (f1, f2, f3, f7). They have different causes but
  the same symptom: the solution believes it's better than it is.
- **f2 is the most dangerous:** it estimates 0.92 for a model that really scores 0.76. A team would
  ship it with complete confidence.
- **f1 and f7 have perfectly good predictions** (real AUC 0.7565 and 0.7568, the same as the reference).
  Only their *estimate* is wrong. That's exactly the "plausible answer" the posting warns about.

## Part 6: How to run it on your laptop

1. Extract the zip over your `verified-ds-tasks` folder (as in Step 4: choose
   `Desktop\Data Analysis`, then **Replace the files in the destination**).
2. Run the checker (about 5–8 minutes on your laptop):
   ```powershell
   python task1_passenger_satisfaction/check_flawed_solutions.py
   ```
   ✅ *You should see:* `solve.py 13/13 PASSES`, eight `CAUGHT` lines, and
   `ALL GOOD: every flawed solution caught by the right check.`
3. Update GitHub:
   ```powershell
   git add .
   git commit -m "Step 5: eight flawed solutions, all caught by the intended check"
   git push
   ```
   Your run rewrites `flawed_solutions/RESULTS.md` with your date, so GitHub shows evidence from your own laptop.

---

## New words in this step

| Word | Plain meaning |
|---|---|
| **Flawed solution / mutant** | A deliberately wrong version of a solution, used to test the tests |
| **Mutation testing** | Checking that tests fail deliberately broken versions of the code |
| **Killed / survived** | A mutant the tests catch is "killed"; one they miss "survived" (a blind spot) |
| **Target leakage** | A feature built from the answer itself (f2) |
| **JUnit XML report** | A standard file listing every test and its result, which other programs can read |
| **False positive (in testing)** | A check failing for the wrong reason, like a crash counted as "caught" |
