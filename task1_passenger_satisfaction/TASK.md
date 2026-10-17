# Task 1: Predict Passenger Satisfaction for New Customers

## Background

SkyLine Air sends a short survey after every flight. The airline wants to predict, **before a
flight**, how likely each passenger is to say they were satisfied, so that staff can offer extra
help to passengers who are likely to be unhappy.

The model will be used on **passengers who have never flown with SkyLine Air before**.
Every passenger in `test.csv` is a new customer who does not appear in `train.csv`.

## Data

Two files are in `data/public/`:

| File | Rows | Contents |
|---|---|---|
| `train.csv` | 33,638 flights by 1,500 passengers | Past flights **with** the survey answer (`satisfied`) |
| `test.csv` | 45,077 flights by 2,000 new passengers | Flights by new customers, **without** the answer |

Each passenger appears on many flights (they are frequent flyers).

### Columns

| Column | Description |
|---|---|
| `flight_id` | Unique id for each flight record |
| `passenger_id` | Unique id for each passenger (the same person has the same id on every flight) |
| `age` | Passenger's age in years |
| `home_airport` | Airport code where the passenger lives |
| `loyalty_tier` | `none`, `silver` or `gold` |
| `travel_type` | `business` or `personal` trip |
| `cabin_class` | `business` or `economy` |
| `flight_distance` | Miles |
| `departure_delay_min` | Departure delay in minutes |
| `online_boarding`, `seat_comfort`, `inflight_wifi`, `food_and_drink` | Service ratings, 1 (worst) to 5 (best) |
| `refund_issued` | 1 if the customer-care team issued a refund **after the flight**, else 0. Refund decisions are made after the survey is returned, so at prediction time this is not yet known and is recorded as 0 |
| `satisfied` | **Target** (train only): 1 = satisfied, 0 = not satisfied |

## What to deliver

Write a Python script, `solution/solve.py`, that reads only the files in `data/public/` and writes
two files to `outputs/`:

**1. `outputs/predictions.csv`**: one row per flight in `test.csv`, with exactly these columns:

```
flight_id,satisfied_probability
16552,0.83
16553,0.41
...
```

`satisfied_probability` is your predicted probability (between 0 and 1) that the passenger is satisfied.

**2. `outputs/cv_estimate.json`**: your own estimate of the ROC AUC your predictions will score
on the test set:

```json
{"auc": 0.75}
```

This estimate must come from your own validation on `train.csv`. You do not have the test answers.

## How your solution is judged

Your predictions are scored against the true test answers (hidden from you) using **ROC AUC**.
A solution passes only if **all** of the following hold:

1. `predictions.csv` has the exact columns above, one row for every `flight_id` in `test.csv`,
   no duplicates, no missing values, and every probability between 0 and 1.
2. **Accuracy:** the test ROC AUC is at least **0.745**.
3. **Honest estimate:** your `cv_estimate.json` AUC must be **no more than 0.010 above** and
   **no more than 0.030 below** your actual test AUC. Overconfidence is limited more strictly
   because it is the more harmful mistake: a model that looks good in validation but not on new
   customers fails, even if its test score is acceptable.
4. **Reproducible:** running `python solution/solve.py` twice produces identical files.

## Rules

- Use only `data/public/`. Do not read or create anything in `data/hidden/`.
- Allowed libraries: those in `requirements.txt` (pandas, numpy, scikit-learn, lightgbm).
- The script must finish within 10 minutes on a normal laptop.
