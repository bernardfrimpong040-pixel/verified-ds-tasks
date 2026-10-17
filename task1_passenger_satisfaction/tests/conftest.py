# ============================================================================
# conftest.py  —  SETUP FOR THE TASK 1 TESTS
#
# pytest reads this file automatically before running the tests. It does
# three jobs:
#
#   1. Adds a --solution option, so you can choose which solution to test:
#        pytest task1_passenger_satisfaction/tests                (tests the reference)
#        pytest task1_passenger_satisfaction/tests --solution path/to/solve.py
#
#   2. Makes sure the data exists (builds it with make_data.py if not).
#
#   3. Runs the solution TWICE, saving each run's output files, so the tests
#      can check the files and check that both runs are identical.
#      While the solution runs, the hidden answers are MOVED OUT of the task
#      folder. A solution that tries to read them simply crashes, so it can't
#      cheat, even by accident.
# ============================================================================

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

TASK_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = TASK_DIR / "data"
HIDDEN_DIR = DATA_DIR / "hidden"
OUTPUT_DIR = TASK_DIR / "outputs"
DEFAULT_SOLUTION = TASK_DIR / "reference_solution" / "solve.py"
TIME_LIMIT_SECONDS = 600     # TASK.md allows 10 minutes

DATA_FILES = ["data/public/train.csv", "data/public/test.csv", "data/hidden/test_labels.csv"]


def pytest_addoption(parser):
    parser.addoption("--solution", action="store", default=str(DEFAULT_SOLUTION),
                     help="path to the solve.py to test (default: the reference solution)")


def make_sure_data_exists():
    """Build the data if any file is missing (make_data.py always gives identical files)."""
    if not all((TASK_DIR / f).exists() for f in DATA_FILES):
        subprocess.run([sys.executable, str(TASK_DIR / "environment" / "make_data.py")],
                       check=True, capture_output=True, text=True)


def run_solution_once(solution, hidden_parking_spot):
    """Run the solution with the hidden answers moved away. Returns what happened."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)              # start clean, so old files can't count

    shutil.move(str(HIDDEN_DIR), str(hidden_parking_spot))   # hide the answers
    try:
        start = time.time()
        result = subprocess.run([sys.executable, str(solution)], cwd=TASK_DIR,
                                capture_output=True, text=True, timeout=TIME_LIMIT_SECONDS)
        seconds = time.time() - start
    except subprocess.TimeoutExpired:
        return {"ok": False, "seconds": TIME_LIMIT_SECONDS,
                "error": f"the solution took longer than {TIME_LIMIT_SECONDS} seconds"}
    finally:
        shutil.move(str(hidden_parking_spot), str(HIDDEN_DIR))  # always put them back

    return {"ok": result.returncode == 0, "seconds": seconds,
            "error": result.stderr[-2000:]}    # the last part of any error message


@pytest.fixture(scope="session")
def solution_runs(request):
    """
    Runs the solution twice and keeps a copy of each run's outputs.
    "scope=session" means this happens ONCE, however many tests use it.
    """
    solution = Path(request.config.getoption("--solution")).resolve()
    assert solution.exists(), f"Solution file not found: {solution}"

    make_sure_data_exists()
    workspace = Path(tempfile.mkdtemp(prefix="task1_checks_"))
    runs = []
    for number in (1, 2):
        info = run_solution_once(solution, workspace / f"hidden_while_run_{number}")
        copy = workspace / f"outputs_run_{number}"
        if OUTPUT_DIR.exists():
            shutil.copytree(OUTPUT_DIR, copy)
        info["outputs"] = copy
        runs.append(info)
        if not info["ok"]:
            break                               # no point running a broken solution again

    yield {"solution": solution, "runs": runs}
    shutil.rmtree(workspace, ignore_errors=True)
