# Step 3 Guide: Building Task 1 (Environment and Reference Solution)

**Goal:** Build a data science task where a plausible-looking answer is not enough, with a
reproducible environment and a verified reference solution.
**Files:** `task1_passenger_satisfaction/environment/make_data.py`, `TASK.md`, `reference_solution/solve.py`
**Result:** reference test AUC **0.7568**, own estimate **0.7548**; every flawed approach tested fails the checks

---

## Part 1: What this job actually involves

The North Hires role asks for people who **create tasks** for AI coding agents. Each task needs:

| Requirement from the posting | Where it is in Task 1 |
|---|---|
| "Build reproducible environments, prepare relevant data" | `make_data.py` rebuilds the data byte for byte from a fixed seed |
| "Implement reference solutions" | `reference_solution/solve.py` |
| "Tests that reliably distinguish correct solutions from flawed approaches" | The pass rules in `TASK.md`, measured in Part 5 (automated in Step 4) |
| "Identify implementation errors, flawed assumptions, data leakage" | Two traps: repeat passengers and a leaky column |
| "Document task requirements, solution reasoning, verification" | `TASK.md`, comments in `solve.py`, this guide |

## Part 2: The task

**Story:** an airline wants to predict, **before a flight**, whether a passenger will say they were
satisfied, and it will use the model on **new customers**.

**The two traps.** These aren't announced as traps, but the task statement gives a careful
solver everything needed to avoid them:

| Trap | What's in the data | The mistake it catches | Why it's a mistake |
|---|---|---|---|
| **1. Repeat passengers** | Frequent flyers (15–30 flights each), each with a hidden personal "mood" | Validating with a plain random split | One person's flights land on both sides of the split, so the model "recognises" them and the score looks too good. Test customers are all new |
| **2. Leaky column** | `refund_issued`, decided **after** the flight, mostly for unhappy passengers | Using it as a feature | It's information from the future. In `test.csv` it's always 0, because no refund exists yet at prediction time |

**What the solver must deliver:** test predictions, plus their **own estimate** of their test AUC.

**The pass rules:**
1. Correct file format
2. Test AUC at least **0.745**
3. Honest estimate: **at most 0.010 too high**, at most 0.030 too low
4. Same files when run twice

Rule 3 is the key one. A flawed approach can still make decent predictions, but it **always
overestimates itself**. In real work, overconfidence is the dangerous mistake: a team trusts a
model that's weaker than it believes.

## Part 3: How the data is made (`make_data.py`)

1. Create **1,500 training passengers** and **2,000 different test passengers**, each with an age,
   home airport, loyalty tier, usual trip type, and a hidden **mood** (never written to the files).
2. Give each passenger **15 to 30 flights**, each with a class, distance, delay and four service ratings.
3. Decide each flight's answer from a **true rule**: a score that adds up class, trip type,
   ratings, delay, loyalty and the person's mood, then turns into a probability.
4. Add `refund_issued`: 60% of unhappy flights get one, 2% of happy ones. In test, always 0.
5. Write `data/public/train.csv` and `test.csv` (what the solver sees), and
   `data/hidden/test_labels.csv` (only the checks may read it).

**Reproducible:** the seed is fixed (`SEED = 2026`), so running the script twice gives
**byte-for-byte identical files**. This was checked by comparing file fingerprints (MD5) from two runs.

## Part 4: The reference solution (`solve.py`)

Three decisions, each explained in the script:

1. **Drop `refund_issued`, `passenger_id` and `flight_id`.** The refund is leakage, and the ids
   carry no information about the flight. Memorising people can't help, since test passengers are new.
2. **GroupKFold by passenger.** All of a person's flights stay in the same fold, so every check is
   on unseen people, just like the real test.
3. **A fixed number of trees** (1,000 small trees, learning rate 0.01, at least 100 flights per leaf).

**Decision 3 came from a mistake the checks caught.** The first version used early stopping. The
folds stopped at very different points (14 trees in one fold, 147 in another), so their predictions
weren't on the same scale, and pooling them made the estimate **0.017 too low**. My own reference
solution failed my own honesty check. Fixed tree counts made the folds consistent. The settings were
chosen by **cross-validation score only**, never by looking at the hidden answers.

## Part 5: Proving the checks work

Building a task means testing the task itself, not just the solution. Three design rounds were
needed:

| Version | Problem found | Fix |
|---|---|---|
| 1. 6,000 passengers, 1–6 flights each | Trap 1 barely worked: random CV overestimated by only 0.001–0.002 | More flights per person, home airport added |
| 2. 1,500 passengers, 6–16 flights | With a careful model, random CV still passed (+0.009, inside the limit) | Stronger mood, more flights; but a small test set made honest scores noisy |
| 3. **1,500 train / 2,000 test passengers, 15–30 flights** | None. Honest solutions pass, every flawed one fails | Asymmetric honesty rule (+0.010 / −0.030) |

**Final check:** every approach, run with two model settings, judged by the pass rules:

| Approach | Careful model: estimate − actual | Default model: estimate − actual | Result |
|---|---:|---:|---|
| **GroupKFold, no ids, no refund** | **−0.0019** | **−0.0077** | ✅ **passes** |
| Random K-fold | +0.0201 | +0.0460 | ❌ |
| Random K-fold + passenger_id | +0.0340 | +0.0931 | ❌ (default also scores 0.7318, below 0.745) |
| GroupKFold + refund_issued | +0.1396 | +0.1395 | ❌ |
| Random K-fold + passenger_id + refund | +0.1531 | +0.1820 | ❌ (default also scores 0.7344) |

Honest solutions **under**estimate slightly; flawed ones **over**estimate by at least **0.020**,
double the allowed 0.010. That margin is what makes the check reliable.

## Part 6: How to set it up on your laptop

This is a **new project** with its own folder, `.venv` and GitHub repository.

1. Unzip `verified-ds-tasks.zip` into `Desktop\Data Analysis\`, next to `kaggle-airline-satisfaction`.
2. In VS Code: **File → New Window**, then **File → Open Folder** → `verified-ds-tasks`.
3. In the terminal:
   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
4. Build the data:
   ```powershell
   python task1_passenger_satisfaction/environment/make_data.py
   ```
   ✅ *You should see:* `train.csv: 33,638 flights by 1,500 passengers (57.0% satisfied)`,
   `test.csv: 45,077 flights by 2,000 NEW passengers`, and `passengers in both train and test: 0`.
5. Run the reference solution (about 20 seconds):
   ```powershell
   python task1_passenger_satisfaction/reference_solution/solve.py
   ```
   ✅ *You should see:* five fold scores and `Cross-validation AUC on unseen passengers: 0.7548`.
6. Create a **public** GitHub repository called `verified-ds-tasks`, with **no** README, .gitignore
   or licence ticked, then:
   ```powershell
   git init
   git add .
   git commit -m "Step 3: Task 1 environment, task statement and reference solution"
   git branch -M main
   git remote add origin https://github.com/bernardfrimpong040-pixel/verified-ds-tasks.git
   git push -u origin main
   ```
   Check `git status` before pushing: **no `data/`** and **no `outputs/`** folders.

---

## New words in this step

| Word | Plain meaning |
|---|---|
| **Task** | A problem with a statement, data, and automatic rules for judging answers |
| **Reference solution** | The task author's own verified answer; proves the task is solvable |
| **Reproducible environment** | A setup that produces exactly the same data and results every time |
| **Seed** | The fixed starting number for random generation |
| **Synthetic data** | Data generated by a program, so its true rules are known |
| **Grouped data** | Rows that belong together (here, flights by the same person) |
| **GroupKFold** | Cross-validation that keeps each group entirely inside one fold |
| **Data leakage** | Using information that wouldn't be available at prediction time |
| **Hidden labels** | The true test answers, kept away from the solver and used only for checking |
| **Overconfidence** | Estimating a model's score higher than it really is |
