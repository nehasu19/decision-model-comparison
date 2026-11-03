"""
Streamlit showcase: run Laya, GLiNER and Jev on the same test cases and compare.

Put this file in the same folder as run_tests.py, questions.csv and testcases.csv.

Setup (once), in any environment:
    pip install streamlit

Run:
    streamlit run app.py

Each model runs in its own Python environment, so the app starts run_tests.py
three times as separate processes. The paths to those environments are in the
sidebar. Jev is hosted: its test texts are sent to a server and it needs a key.
"""

import json
import os
import re
import subprocess
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
HOME = Path.home()
TIMINGS = HERE / "timings.json"

MODELS = {
    "laya": {"label": "Laya", "where": "local, open source", "color": "#2a78d6",
             "python": HOME / "laya-env/bin/python"},
    "gliner": {"label": "GLiNER2.5-Decide", "where": "local, open source",
               "color": "#1baf7a", "python": HOME / "gliner-env/bin/python"},
    "opendecider": {"label": "OpenDecider nano", "where": "local, open source",
                    "color": "#4a3aa7", "python": HOME / "opendecider-env/bin/python"},
    "jev": {"label": "Jev", "where": "hosted, paid API", "color": "#eb6834",
            "python": HOME / "laya-env/bin/python"},
}

st.set_page_config(page_title="Decision model comparison", layout="wide")


# ---------------------------------------------------------------- helpers

def run_model(name, python, api_key, log_box):
    """Run run_tests.py for one model in its own environment; stream its output."""
    env = os.environ.copy()
    if api_key:
        env["OPENROUTER_API_KEY"] = api_key
        env.pop("TYPESAFE_API_KEY", None)
    try:
        process = subprocess.Popen(
            [str(python), "run_tests.py", name], cwd=HERE, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except OSError as error:
        return False, f"Could not start {python}: {error}"
    lines = []
    for line in process.stdout:
        lines.append(line.rstrip())
        log_box.code("\n".join(lines[-12:]) or " ")
    process.wait()
    return process.returncode == 0, "\n".join(lines)


def read_time(output):
    found = re.search(r"Average time per question:\s*([\d.]+)\s*ms", output)
    return float(found.group(1)) if found else None


def load_timings():
    try:
        return json.loads(TIMINGS.read_text())
    except (OSError, ValueError):
        return {}


def load_results():
    results = {}
    for name in MODELS:
        path = HERE / f"results_{name}.csv"
        if path.exists():
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame["correct"] = frame["result"] == "PASS"
            if "ms" in frame:
                frame["ms"] = pd.to_numeric(frame["ms"], errors="coerce")
            results[name] = frame
    return results


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Settings")
    chosen = [name for name, m in MODELS.items()
              if st.checkbox(m["label"], value=True, key=f"use_{name}")]
    api_key = st.text_input(
        "OpenRouter or TypeSafe key (for Jev)", type="password",
        value=os.environ.get("OPENROUTER_API_KEY",
                             os.environ.get("TYPESAFE_API_KEY", "")),
        help="Used only for this run. It is not saved to disk.")
    with st.expander("Python environment for each model"):
        pythons = {name: st.text_input(m["label"], value=str(m["python"]),
                                       key=f"py_{name}")
                   for name, m in MODELS.items()}
    st.caption("Jev sends the test texts to an external server. "
               "Laya and GLiNER run on this machine.")

# ---------------------------------------------------------------- header and run

st.title("Decision models, side by side")
st.write("The same test cases, sent to three models that pick an answer from a "
         "fixed set of options.")

cases_file = HERE / "testcases.csv"
if not cases_file.exists() or not (HERE / "run_tests.py").exists():
    st.error("Put app.py in the same folder as run_tests.py, questions.csv "
             "and testcases.csv.")
    st.stop()

cases = pd.read_csv(cases_file, dtype=str, keep_default_na=False)
st.caption(f"{len(cases)} test cases in {cases['group'].nunique()} groups, "
           "read from testcases.csv.")

if st.button("Run all selected models", type="primary", disabled=not chosen):
    timings = load_timings()
    for name in chosen:
        label = MODELS[name]["label"]
        if name == "jev" and not api_key:
            st.warning("Jev skipped: enter an API key in the sidebar.")
            continue
        with st.status(f"Running {label} ...", expanded=True) as status:
            ok, output = run_model(name, pythons[name], api_key, st.empty())
            if ok:
                timings[name] = read_time(output)
                status.update(label=f"{label} finished", state="complete",
                              expanded=False)
            else:
                status.update(label=f"{label} failed", state="error")
                st.code(output[-2500:] or "no output")
    TIMINGS.write_text(json.dumps(timings))

# ---------------------------------------------------------------- comparison

results = load_results()
current = dict(zip(cases["id"], cases["expected"]))
stale = [name for name, frame in results.items()
         if dict(zip(frame["id"], frame["expected"])) != current]
for name in stale:
    del results[name]
if stale:
    st.warning("Saved results for " + ", ".join(MODELS[n]["label"] for n in stale)
               + " come from an older version of testcases.csv and are hidden. "
               "Run those models again.")
if not results:
    st.info("No results yet. Press the button to run the models.")
    st.stop()

timings = load_timings()
st.header("Comparison")

tiles = st.columns(len(results))
for tile, (name, frame) in zip(tiles, results.items()):
    right, total = int(frame["correct"].sum()), len(frame)
    with tile:
        st.metric(MODELS[name]["label"], f"{right}/{total}",
                  f"{right / total:.0%} correct", delta_color="off")
        if "ms" in frame and frame["ms"].notna().any():
            st.caption(f"{MODELS[name]['where']} · average {frame['ms'].mean():.0f} ms, "
                       f"median {frame['ms'].median():.0f} ms per question")
        else:
            speed = timings.get(name)
            st.caption(MODELS[name]["where"]
                       + (f" · {speed:.0f} ms per question" if speed else ""))

st.subheader("Correct answers by test group")
order = list(dict.fromkeys(cases["group"]))
by_group = pd.concat(
    [frame.groupby("group")["correct"].agg(correct="sum", total="count")
     .reset_index().assign(model=MODELS[name]["label"])
     for name, frame in results.items()])
by_group["share"] = by_group["correct"] / by_group["total"]
by_group["label"] = (by_group["correct"].astype(int).astype(str) + "/"
                     + by_group["total"].astype(str))
labels = [MODELS[n]["label"] for n in results]
chart = (
    alt.Chart(by_group)
    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    .encode(
        x=alt.X("group:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("model:N", sort=labels),
        y=alt.Y("share:Q", title="Correct", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", sort=labels, title=None,
                        scale=alt.Scale(domain=labels,
                                        range=[MODELS[n]["color"] for n in results]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("model:N", title="Model"),
                 alt.Tooltip("group:N", title="Group"),
                 alt.Tooltip("label:N", title="Correct")])
    .properties(height=360))
st.altair_chart(chart, use_container_width=True)

timed = {name: frame for name, frame in results.items()
         if "ms" in frame and frame["ms"].notna().any()}
if timed:
    st.subheader("Response time by test group")
    by_time = pd.concat(
        [frame.groupby("group")["ms"].mean().reset_index()
         .assign(model=MODELS[name]["label"]) for name, frame in timed.items()])
    time_labels = [MODELS[n]["label"] for n in timed]
    time_chart = (
        alt.Chart(by_time)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("group:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("model:N", sort=time_labels),
            y=alt.Y("ms:Q", title="Average milliseconds per question"),
            color=alt.Color("model:N", sort=time_labels, title=None,
                            scale=alt.Scale(domain=time_labels,
                                            range=[MODELS[n]["color"] for n in timed]),
                            legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("model:N", title="Model"),
                     alt.Tooltip("group:N", title="Group"),
                     alt.Tooltip("ms:Q", title="Average ms", format=".0f")])
        .properties(height=320))
    st.altair_chart(time_chart, use_container_width=True)
    st.caption("Time from sending a question to receiving the answer. Jev's time "
               "includes the internet round trip; the local models' times depend "
               "on this machine. One untimed warm-up question is sent before each run.")

st.subheader("Every test case")
show_time = bool(timed) and st.checkbox("Show response time for each case")
table = cases[["id", "group", "text", "expected"]].copy()
table["text"] = table["text"].str.slice(0, 400)
for name, frame in results.items():
    answers = frame.set_index("id")
    label = MODELS[name]["label"]
    table[label] = table["id"].map(
        lambda i: (("PASS  " if answers.at[i, "correct"] else "FAIL  ")
                   + answers.at[i, "got"]) if i in answers.index else "not run")
    if show_time and name in timed:
        table[label + " ms"] = table["id"].map(
            lambda i: f"{answers.at[i, 'ms']:.0f}" if i in answers.index else "")

model_columns = [MODELS[n]["label"] for n in results]
passed = table[model_columns].apply(lambda col: col.str.startswith("PASS"))
view = st.radio("Show", ["All cases", "Where the models disagree",
                         "Where any model failed", "Where all models failed",
                         "Where all models passed"], horizontal=True)
if view == "Where the models disagree":
    table = table[passed.nunique(axis=1) > 1]
elif view == "Where any model failed":
    table = table[~passed.all(axis=1)]
elif view == "Where all models failed":
    table = table[~passed.any(axis=1)]
elif view == "Where all models passed":
    table = table[passed.all(axis=1)]
groups_shown = st.multiselect("Test groups", order, default=order)
table = table[table["group"].isin(groups_shown)]
st.dataframe(table, use_container_width=True, hide_index=True)
st.caption(f"{len(table)} rows shown. Each cell is the result and the model's "
           "answer.")

st.info("Small test with made-up examples: a first impression, not a benchmark. "
        "GLiNER receives option names only, while Laya and Jev also receive a "
        "description for each option. Jev's speed includes the network trip.")
