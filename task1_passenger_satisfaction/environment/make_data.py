# ============================================================================
# TASK 1 ENVIRONMENT: BUILD THE DATA
# Run with:   python task1_passenger_satisfaction/environment/make_data.py
#
# WHAT THIS SCRIPT DOES
#   It CREATES a realistic airline-satisfaction dataset from scratch, using
#   random numbers with a FIXED SEED. A fixed seed means the "random" numbers
#   come out exactly the same every time, so anyone who runs this script gets
#   exactly the same files, byte for byte. That is what "reproducible" means.
#
# WHY MAKE OUR OWN DATA?
#   1. Anyone can rebuild it: no downloads, no logins, no licence problems.
#   2. We know the TRUE rules behind the data, so we can plant traps on
#      purpose and check whether a solution falls into them.
#
# THE TWO TRAPS (the solver is NOT told about these directly)
#   TRAP 1 - REPEAT PASSENGERS. Each passenger flies several times, and each
#     person has their own hidden "mood" that makes them happier or grumpier
#     on every flight. The test set contains only NEW passengers. A solution
#     that validates with a plain random split lets the model "recognise"
#     people it has already seen (age + home airport almost identify a person,
#     and passenger_id identifies them exactly), so its score looks better than
#     it really is.
#   TRAP 2 - A LEAKY COLUMN. "refund_issued" is decided AFTER the flight,
#     mostly for unhappy passengers. In the training file it gives the answer
#     away. In the test file it is always 0, because at prediction time no
#     refund has been decided yet. A model that relies on it collapses.
#
# FILES IT WRITES
#   data/public/train.csv   - what the solver gets, WITH the answers
#   data/public/test.csv    - what the solver predicts, WITHOUT the answers
#   data/hidden/test_labels.csv - the true test answers. ONLY the tests
#                                 (Step 4) may read this. The solver never sees it.
# ============================================================================

from pathlib import Path
import numpy as np
import pandas as pd

SEED = 2026                 # the fixed seed: change it and you get different data
N_TRAIN_PASSENGERS = 1500   # people in the training set
N_TEST_PASSENGERS = 2000    # DIFFERENT people (a large test set keeps the score stable)
MIN_FLIGHTS, MAX_FLIGHTS = 15, 30  # frequent flyers: each person flies 15 to 30 times

AIRPORTS = ["ATL", "BOS", "CLE", "CMH", "DEN", "DFW", "DTW", "EWR", "IAH", "JFK", "LAS",
            "LAX", "MCO", "MIA", "MSP", "ORD", "PHL", "PHX", "PIT", "SEA", "SFO", "SLC"]

TASK_DIR = Path(__file__).resolve().parent.parent
PUBLIC_DIR = TASK_DIR / "data" / "public"
HIDDEN_DIR = TASK_DIR / "data" / "hidden"


def make_passengers(rng, n, first_id):
    """One row per PERSON: things that stay the same on every flight."""
    return pd.DataFrame({
        "passenger_id": np.arange(first_id, first_id + n),
        "age": rng.integers(18, 80, n),
        "home_airport": rng.choice(AIRPORTS, n),
        "loyalty_tier": rng.choice(["none", "silver", "gold"], n, p=[0.6, 0.3, 0.1]),
        # Each person's usual reason for flying and usual class
        "usual_travel": rng.choice(["business", "personal"], n, p=[0.6, 0.4]),
        # The HIDDEN mood: some people are just happier or grumpier.
        # It is never written to the files; it only affects the answers.
        "mood": rng.normal(0, 2.5, n),
    })


def make_flights(rng, passengers):
    """One row per FLIGHT. Each passenger gets between 1 and 6 flights."""
    n_flights = rng.integers(MIN_FLIGHTS, MAX_FLIGHTS + 1, len(passengers))
    flights = passengers.loc[passengers.index.repeat(n_flights)].reset_index(drop=True)
    n = len(flights)

    # Trip type: usually the person's usual reason, sometimes not
    keep_usual = rng.random(n) < 0.85
    other = np.where(flights["usual_travel"] == "business", "personal", "business")
    flights["travel_type"] = np.where(keep_usual, flights["usual_travel"], other)

    # Class: business travellers fly Business class more often
    business_trip = flights["travel_type"] == "business"
    p_business_class = np.where(business_trip, 0.65, 0.12)
    flights["cabin_class"] = np.where(rng.random(n) < p_business_class, "business", "economy")

    flights["flight_distance"] = rng.integers(150, 4500, n)
    flights["departure_delay_min"] = np.round(rng.exponential(14, n))

    # Service ratings, 1 to 5. Business class tends to rate higher.
    boost = np.where(flights["cabin_class"] == "business", 0.8, 0.0)
    for rating in ["online_boarding", "seat_comfort", "inflight_wifi", "food_and_drink"]:
        raw = rng.normal(3.0 + boost, 1.1, n)
        flights[rating] = np.clip(np.round(raw), 1, 5).astype(int)
    return flights


def decide_satisfaction(rng, flights):
    """
    The TRUE rule behind the answers (the solver has to learn this from data).
    A score adds up good and bad things; a higher score means more likely satisfied.
    """
    score = (
        -6.8
        + 1.1 * (flights["cabin_class"] == "business")
        + 0.9 * (flights["travel_type"] == "business")
        + 0.9 * flights["online_boarding"]
        + 0.6 * flights["seat_comfort"]
        + 0.35 * flights["inflight_wifi"]
        + 0.15 * flights["food_and_drink"]
        - 0.02 * flights["departure_delay_min"]
        + 0.4 * (flights["loyalty_tier"] == "gold")
        + flights["mood"]                      # TRAP 1: the hidden personal mood
    )
    probability = 1 / (1 + np.exp(-score))     # turn the score into a chance (0 to 1)
    return (rng.random(len(flights)) < probability).astype(int)


def add_refund(rng, flights):
    """
    TRAP 2: refunds are decided AFTER the flight, mostly for unhappy passengers.
    60% of unsatisfied passengers get one; only 2% of satisfied ones do.
    """
    p_refund = np.where(flights["satisfied"] == 1, 0.02, 0.60)
    return (rng.random(len(flights)) < p_refund).astype(int)


def main():
    rng = np.random.default_rng(SEED)

    train_people = make_passengers(rng, N_TRAIN_PASSENGERS, first_id=100000)
    test_people = make_passengers(rng, N_TEST_PASSENGERS, first_id=200000)

    train = make_flights(rng, train_people)
    test = make_flights(rng, test_people)

    train["satisfied"] = decide_satisfaction(rng, train)
    test["satisfied"] = decide_satisfaction(rng, test)

    train["refund_issued"] = add_refund(rng, train)
    test["refund_issued"] = 0    # at prediction time, no refund has been decided yet

    # Shuffle the rows so passengers aren't neatly grouped together, then
    # give every flight its own id
    train = train.sample(frac=1, random_state=SEED).reset_index(drop=True)
    test = test.sample(frac=1, random_state=SEED + 1).reset_index(drop=True)
    train.insert(0, "flight_id", np.arange(1, len(train) + 1))
    test.insert(0, "flight_id", np.arange(len(train) + 1, len(train) + len(test) + 1))

    columns = ["flight_id", "passenger_id", "age", "home_airport", "loyalty_tier", "travel_type",
               "cabin_class", "flight_distance", "departure_delay_min",
               "online_boarding", "seat_comfort", "inflight_wifi", "food_and_drink",
               "refund_issued"]

    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    HIDDEN_DIR.mkdir(parents=True, exist_ok=True)
    # lineterminator="\n" writes the same line endings on Windows, Mac and Linux,
    # so the files are byte-for-byte identical on every computer (the tests check this)
    train[columns + ["satisfied"]].to_csv(PUBLIC_DIR / "train.csv", index=False, lineterminator="\n")
    test[columns].to_csv(PUBLIC_DIR / "test.csv", index=False, lineterminator="\n")
    test[["flight_id", "satisfied"]].to_csv(HIDDEN_DIR / "test_labels.csv", index=False,
                                            lineterminator="\n")

    print(f"train.csv: {len(train):,} flights by {train['passenger_id'].nunique():,} passengers "
          f"({train['satisfied'].mean():.1%} satisfied)")
    print(f"test.csv:  {len(test):,} flights by {test['passenger_id'].nunique():,} NEW passengers")
    shared = set(train["passenger_id"]) & set(test["passenger_id"])
    print(f"passengers in both train and test: {len(shared)}")
    print(f"Files written to {PUBLIC_DIR} and {HIDDEN_DIR}")


if __name__ == "__main__":
    main()
