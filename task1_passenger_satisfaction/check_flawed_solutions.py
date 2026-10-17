# ============================================================================
# CHECK THE CHECKS: run every flawed solution through the tests
# Run from the main project folder with:
#     python task1_passenger_satisfaction/check_flawed_solutions.py
#
# WHY THIS EXISTS
#   A test suite is only trustworthy if it FAILS bad solutions, and fails
#   them for the RIGHT reason. This script runs the full test suite against:
#     - the reference solution, which must pass every check, and
#     - every file in flawed_solutions/, each of which must be caught by the
#       check named on its "EXPECTED TO BE CAUGHT BY:" line.
#
#   "Caught for the right reason" matters: a leaky solution that only failed
#   because of a typo would prove nothing about the leakage check. So if a
#   solution CRASHES (when crashing isn't the mistake being tested), it is
#   reported as "CRASHED: proves nothing", not as caught.
#
# HOW LONG: about 45-60 seconds per solution, so 6-9 minutes in total.
#
# OUTPUT: a table on screen, and the same table saved to
#         flawed_solutions/RESULTS.md (so the evidence is on GitHub).
# ============================================================================

import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
TESTS_DIR = TASK_DIR / "tests"
FLAWED_DIR = TASK_DIR / "flawed_solutions"
REFERENCE = TASK_DIR / "reference_solution" / "solve.py"


def expected_check(script):
    """Read the 'EXPECTED TO BE CAUGHT BY:' line from a flawed solution's header."""
    for line in script.read_text().splitlines():
        if "EXPECTED TO BE CAUGHT BY:" in line:
            return line.split("EXPECTED TO BE CAUGHT BY:")[1].strip()
    raise SystemExit(f"{script.name} has no 'EXPECTED TO BE CAUGHT BY:' line")


def title_of(script):
    """Read the 'FLAWED SOLUTION:' title from the header."""
    for line in script.read_text().splitlines():
        if "FLAWED SOLUTION:" in line:
            return line.split("FLAWED SOLUTION:")[1].strip().capitalize()
    return script.stem


def run_tests(solution):
    """
    Run pytest on one solution and return (checks passed, list of failed checks).
    pytest writes a small XML report (--junitxml) listing every check and
    whether it failed; we read that report instead of the screen output.
    """
    report = Path(tempfile.mkdtemp()) / "report.xml"
    subprocess.run([sys.executable, "-m", "pytest", str(TESTS_DIR), "-q",
                    "-p", "no:cacheprovider", f"--solution={solution}",
                    f"--junitxml={report}"],
                   cwd=TASK_DIR.parent, capture_output=True, text=True)
    passed, failed = 0, []
    for case in ET.parse(report).getroot().iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            failed.append(case.get("name"))
        else:
            passed += 1
    return passed, failed


def base_name(check):
    """'test_two_runs_give_identical_files[predictions.csv]' -> 'test_two_runs_give_identical_files'"""
    return check.split("[")[0]


def main():
    rows = []
    start = time.time()

    # 1. The reference must pass everything
    print("Testing the reference solution...")
    passed, failed = run_tests(REFERENCE)
    ok = not failed
    rows.append(("reference_solution/solve.py", "Reference solution", "(nothing)",
                 f"{passed}/{passed + len(failed)}", "none" if ok else ", ".join(failed),
                 "PASSES" if ok else "PROBLEM: the reference failed"))
    print(f"   {passed} passed, {len(failed)} failed -> {'OK' if ok else 'PROBLEM'}")

    # 2. Every flawed solution must be caught by its expected check
    for script in sorted(FLAWED_DIR.glob("f*.py")):
        expected = expected_check(script)
        print(f"Testing {script.name} (should be caught by {expected})...")
        passed, failed = run_tests(script)
        failed_names = {base_name(name) for name in failed}
        crashed = "test_solution_runs_without_error" in failed_names
        if crashed and expected != "test_solution_runs_without_error":
            # It crashed, so every later check fails too. That proves nothing about
            # the check this mistake was meant to test, so it does NOT count.
            verdict = "CRASHED: proves nothing"
        elif expected in failed_names:
            verdict = "CAUGHT"
        elif failed:
            verdict = "caught, but by a different check"
        else:
            verdict = "NOT CAUGHT"
        rows.append((f"flawed_solutions/{script.name}", title_of(script), expected,
                     f"{passed}/{passed + len(failed)}",
                     ", ".join(failed) if failed else "none", verdict))
        print(f"   {passed} passed, {len(failed)} failed -> {verdict}")

    all_good = rows[0][5] == "PASSES" and all(r[5] == "CAUGHT" for r in rows[1:])

    # 3. Save the evidence as a Markdown table
    lines = [
        "# Results: are the flawed solutions caught?",
        "",
        f"Generated by `check_flawed_solutions.py` on {datetime.now():%Y-%m-%d %H:%M}.",
        "",
        "| Solution | Mistake | Must be caught by | Checks passed | Checks failed | Verdict |",
        "|---|---|---|---:|---|---|",
    ]
    for path, title, expected, score, failed, verdict in rows:
        mark = "✅" if verdict in ("PASSES", "CAUGHT") else "❌"
        lines.append(f"| `{path}` | {title} | `{expected}` | {score} | {failed} | {mark} {verdict} |")
    lines += ["", "**Overall: " + ("every flawed solution was caught by the right check, and the "
                                    "reference passed everything.**" if all_good else
                                    "PROBLEM: see the rows marked ❌.**")]
    (FLAWED_DIR / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 72)
    for path, title, expected, score, failed, verdict in rows:
        print(f"{Path(path).name:<38}{score:>7}   {verdict}")
    print("=" * 72)
    print("ALL GOOD: every flawed solution caught by the right check." if all_good
          else "PROBLEM: see above.")
    print(f"Saved the table to {FLAWED_DIR / 'RESULTS.md'}  ({(time.time() - start) / 60:.1f} minutes)")
    sys.exit(0 if all_good else 1)


if __name__ == "__main__":
    main()
