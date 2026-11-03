"""
Hallucination report: reads results_<model>.csv in this folder and shows how
well each model catches answers that are wrong or made up.

Run after the models:
    python hallucination/report.py
"""

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = ["correct", "incorrect", "unsupported"]

files = sorted(HERE.glob("results_*.csv"))
if not files:
    raise SystemExit("No results yet. Run: python run_tests.py laya --data hallucination")

for path in files:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    name = path.stem.replace("results_", "")
    grid = {e: {g: 0 for g in LABELS} for e in LABELS}
    for r in rows:
        got = r["got"].strip().lower()
        if got in LABELS:
            grid[r["expected"]][got] += 1

    def caught(label):
        total = sum(grid[label].values())
        hits = sum(grid[label][g] for g in LABELS if g != "correct")
        return hits, total

    made_up, made_up_n = caught("unsupported")
    wrong, wrong_n = caught("incorrect")
    good_n = sum(grid["correct"].values())
    alarms = good_n - grid["correct"]["correct"]
    exact = sum(grid[l][l] for l in LABELS)

    print(f"\n=== {name} ===")
    print(f"Exact label right:        {exact}/{len(rows)}")
    print(f"Made-up answers caught:   {made_up}/{made_up_n}   (unsupported, not passed as correct)")
    print(f"Wrong answers caught:     {wrong}/{wrong_n}   (incorrect, not passed as correct)")
    print(f"False alarms:             {alarms}/{good_n}   (correct answers flagged as bad)")
    print("\nConfusion table (rows = expected, columns = model said)")
    print(f"{'':14}" + "".join(f"{g:>13}" for g in LABELS))
    for e in LABELS:
        print(f"{e:14}" + "".join(f"{grid[e][g]:>13}" for g in LABELS))
