"""
Run the shared test cases against one model.

The tests live in two CSV files next to this script, so every script and every
model reads the same ones:
    questions.csv   one row per group: the question and its options
    testcases.csv   one row per example: group, text, expected answer

Run (in the environment that has the model's package):
    python run_tests.py laya        # needs: source ~/laya-env/bin/activate
    python run_tests.py gliner      # needs: source ~/gliner-env/bin/activate
    python run_tests.py jev         # needs an API key, see below

Jev is hosted, so it sends the texts to a server and needs
    export OPENROUTER_API_KEY="your-key"     (or TYPESAFE_API_KEY)

Another test set:  python run_tests.py laya --data hallucination

Each run prints a scorecard and saves results_<model>.csv, including the
response time of every case in milliseconds (the "ms" column). One untimed
warm-up question is sent first, so model loading is not counted.

Editing the tests: open the CSV files in a spreadsheet or text editor.
  * Add an example: add a row to testcases.csv with an existing group name.
  * Add a group: add a row to questions.csv, then examples that use it.
  * options are separated by " | ". For type "choice" each option is
    "name: description" (description optional). For type "score" list the
    levels from lowest to highest; expected is the level's text.
"""

import csv
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning, module="laya")

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- reading the CSVs

def load_questions(path):
    questions = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names, descriptions = [], {}
            for part in row["options"].split("|"):
                part = part.strip()
                if not part:
                    continue
                name, _, desc = part.partition(":")
                name = name.strip()
                names.append(name)
                descriptions[name] = desc.strip() or name
            if len(names) < 2:
                sys.exit(f"questions.csv: group '{row['group']}' needs 2+ options")
            questions[row["group"].strip()] = {
                "task": row["task"].strip(), "type": row["type"].strip().lower(),
                "instructions": row["instructions"].strip(),
                "names": names, "descriptions": descriptions}
    return questions


def load_cases(path, questions):
    cases = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            group, expected = row["group"].strip(), row["expected"].strip()
            if group not in questions:
                sys.exit(f"testcases.csv row {row['id']}: unknown group '{group}'")
            if expected not in questions[group]["names"]:
                sys.exit(f"testcases.csv row {row['id']}: expected '{expected}' "
                         f"is not an option of group '{group}'")
            cases.append((row["id"], group, row["text"], expected))
    return cases


# ---------------------------------------------------------------- the models
# Every model gets ask(text, question) and returns (answer name, probability or None).

def systemone_question(q):
    """The request format shared by Laya and Jev."""
    if q["type"] == "score":
        return {"type": "score", "instructions": q["instructions"],
                "criteria": q["names"]}
    return {"type": "choice", "instructions": q["instructions"],
            "criteria": q["descriptions"]}


def read_systemone(answer, q):
    if q["type"] == "score":
        level = max(0, min(len(q["names"]) - 1, round(answer["score"])))
        return q["names"][level], answer.get("probabilities", {}).get(str(level))
    pick = answer["choice"]
    return pick, answer.get("probabilities", {}).get(pick)


class Laya:
    def __init__(self):
        from laya import Router
        self.router = Router()

    def ask(self, text, q):
        out = self.router.predict(text, {"q": systemone_question(q)})
        return read_systemone(out["answers"]["q"], q)


class Jev:
    def __init__(self):
        import requests
        self.requests = requests
        self.key = (os.environ.get("TYPESAFE_API_KEY")
                    or os.environ.get("OPENROUTER_API_KEY") or "").strip()
        if not self.key:
            sys.exit('Set your key first:  export OPENROUTER_API_KEY="your-key"')
        if self.key.startswith("sk-or-"):
            self.url = "https://openrouter.ai/api/v1/systemone"
            self.model = "typesafe/jev-1.13"
        else:
            self.url = "https://api.typesafe.ai/v1/systemone"
            self.model = "jev-latest"

    def ask(self, text, q):
        r = self.requests.post(
            self.url, timeout=60, headers={"Authorization": f"Bearer {self.key}"},
            json={"model": self.model, "state": text,
                  "questions": {"q": systemone_question(q)}})
        if r.status_code != 200:
            sys.exit(f"Jev returned {r.status_code}: {r.text[:300]}")
        return read_systemone(r.json()["answers"]["q"], q)


class Gliner:
    """Takes a task name and label names only (no descriptions)."""

    def __init__(self):
        from gliner2 import AutoExtractor
        self.extractor = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide")
        self.with_confidence = True

    def ask(self, text, q):
        schema = {q["task"]: q["names"]}
        if self.with_confidence:
            try:
                raw = self.extractor.classify_text(text, schema,
                                                   include_confidence=True)
            except TypeError:
                self.with_confidence = False
                raw = self.extractor.classify_text(text, schema)
        else:
            raw = self.extractor.classify_text(text, schema)
        value = raw.get(q["task"], raw) if isinstance(raw, dict) else raw
        if isinstance(value, list):
            value = value[0] if value else ""
        if isinstance(value, dict):
            return (str(value.get("label", value.get("text", ""))),
                    value.get("confidence", value.get("score")))
        return str(value), None


class OpenDecider:
    """Same question format as Laya and Jev, runs locally."""

    def __init__(self):
        from opendecider import load
        self.model = load("manjunathshiva/opendecider-nano")

    def ask(self, text, q):
        out = self.model.system_one(text, {"q": systemone_question(q)})
        return read_systemone(out["answers"]["q"], q)


MODELS = {"laya": Laya, "jev": Jev, "gliner": Gliner, "opendecider": OpenDecider}


# ---------------------------------------------------------------- running

def main():
    data = HERE
    if "--data" in sys.argv:
        i = sys.argv.index("--data")
        data = (HERE / sys.argv[i + 1]).resolve()
        del sys.argv[i:i + 2]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args or args[0] not in MODELS:
        sys.exit("Usage: python run_tests.py laya | jev | gliner | opendecider")
    name = args[0]

    questions = load_questions(data / "questions.csv")
    cases = load_cases(data / "testcases.csv", questions)
    print(f"{len(cases)} test cases in {len(questions)} groups. Loading {name} ...")
    model = MODELS[name]()

    # Untimed warm-up, so the first real case does not pay for model start-up.
    model.ask(cases[0][2], questions[cases[0][1]])

    rows, times = [], []
    score = {g: [0, 0] for g in questions}
    slow = {g: [] for g in questions}
    for case_id, group, text, expected in cases:
        start = time.perf_counter()
        got, prob = model.ask(text, questions[group])
        ms = (time.perf_counter() - start) * 1000
        times.append(ms)
        slow[group].append(ms)
        ok = str(got).strip().lower() == expected.lower()
        score[group][0] += ok
        score[group][1] += 1
        rows.append([case_id, group, " ".join(text.split())[:90], expected, got,
                     "" if prob is None else f"{prob:.2f}",
                     "PASS" if ok else "FAIL", f"{ms:.0f}"])

    total = sum(s[0] for s in score.values())
    print(f"\nSCORECARD for {name}: {total}/{len(cases)} correct")
    for group, (right, n) in score.items():
        if n:
            print(f"  {group:<14}{right}/{n:<6}{statistics.mean(slow[group]):7.0f} ms average")

    print("\nRESPONSE TIME")
    print(f"Average time per question: {statistics.mean(times):.0f} ms")
    print(f"Median: {statistics.median(times):.0f} ms   "
          f"Fastest: {min(times):.0f} ms   Slowest: {max(times):.0f} ms")

    print("\nFAILED CASES")
    for r in rows:
        if r[6] == "FAIL":
            print(f"  #{r[0]} [{r[1]}] expected={r[3]} got={r[4]} {r[5]}  {r[2][:55]}")

    out = data / f"results_{name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "group", "text", "expected", "got", "probability",
                    "result", "ms"])
        w.writerows(rows)
    print(f"\nSaved to {out.name}")


if __name__ == "__main__":
    main()
