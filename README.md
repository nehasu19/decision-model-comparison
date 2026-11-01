# Decision model comparison: Laya, GLiNER2.5-Decide and Jev

A small test bench that sends the same test cases to three "decision models"
and compares the answers in a Streamlit page.

A decision model does not write text. You give it a short text and a fixed set
of options, and it picks one.

| Model | Where it runs | Notes |
|---|---|---|
| Laya | On your machine | Open source (Apache 2.0) |
| GLiNER2.5-Decide | On your machine | Open source (Apache 2.0), English only |
| Jev | TypeSafe's servers, via OpenRouter | Hosted and paid; needs an API key |

> Status: built and run on Ubuntu with Python 3.10. The macOS steps below
> follow the same procedure but have not been tested on a Mac. The GLiNER
> setup in particular was still being completed when this was written.

## What is in the folder

| File | Purpose |
|---|---|
| `app.py` | The Streamlit page: one button runs the models, then shows the comparison |
| `run_tests.py` | Runs the test cases against one model and saves `results_<model>.csv` |
| `testcases.csv` | The test cases: id, group, text, expected answer (74 rows) |
| `questions.csv` | One row per test group: the question and its options (10 rows) |

The ten test groups: intent routing, sentiment, spam, complaint (yes/no),
urgency rating, 15 topics, four languages (German, Spanish, Dutch, French),
long text, arithmetic, and RAG answer correctness.

## Requirements (macOS)

- macOS with Python 3.10 or newer. Check with `python3 --version`.
  If it is missing or older, install it from python.org or with
  `brew install python`.
- About 5 GB of free disk space for the libraries and model downloads.
- Internet access for the first run (model downloads) and for Jev.
- For Jev only: an OpenRouter account with a little credit, and an API key
  from https://openrouter.ai/settings/keys

No NVIDIA GPU is needed. On a Mac the local models run on the CPU or on
Apple's GPU, which is slower than a dedicated GPU but fine for 74 short cases.

## Setup

Open Terminal and go to the project folder:

```bash
cd path/to/this/folder
```

Each local model gets its own Python environment so their libraries cannot
clash. The app expects them at `~/laya-env` and `~/gliner-env`.

### 1. Laya environment (also used for the app and for Jev)

```bash
python3 -m venv ~/laya-env
source ~/laya-env/bin/activate
python -m pip install --upgrade pip
pip install laya streamlit requests
deactivate
```

### 2. GLiNER environment

```bash
python3 -m venv ~/gliner-env
source ~/gliner-env/bin/activate
python -m pip install --upgrade pip
pip install gliner2 torch transformers sentencepiece protobuf peft accelerate safetensors huggingface_hub
deactivate
```

The extra packages are listed because `pip install gliner2` alone did not pull
them in on the original machine. If a run stops with
`ModuleNotFoundError: No module named 'X'`, activate this environment and run
`pip install X`.

### 3. API key for Jev (optional)

You can skip Jev by unticking it in the app. To include it, create your own
key at OpenRouter. Enter it in the app's sidebar, or set it in the terminal
before starting the app:

```bash
export OPENROUTER_API_KEY="your-key"
```

Never put the key in a file in this folder, and never commit or share it.

## Check each model from the terminal first

This finds setup problems before the app hides them behind a button.

```bash
source ~/laya-env/bin/activate
python run_tests.py laya
python run_tests.py jev        # needs the key
deactivate

source ~/gliner-env/bin/activate
python run_tests.py gliner
deactivate
```

Each run prints a scorecard and writes `results_<model>.csv`. The first run of
each local model downloads its weights, which takes a few minutes.

## Run the app

```bash
source ~/laya-env/bin/activate
streamlit run app.py
```

- The first time, Streamlit asks for an email address. Leave it blank and
  press Enter.
- The page opens at http://localhost:8501
- Tick the models in the sidebar, enter the key if you want Jev, and press
  "Run all selected models".
- Keep the terminal open while using the page. Press Ctrl+C there to stop it.

After the runs you get a score per model, a chart per test group, and a table
of every case with filters (models disagree, any failed, all failed, all
passed, and by test group).

If your environments are somewhere other than `~/laya-env` and `~/gliner-env`,
change the paths under "Python environment for each model" in the sidebar.

## Editing the test cases

Open the CSV files in Numbers, Excel or a text editor. Save as CSV in UTF-8.

- Add an example: add a row to `testcases.csv` using an existing group name.
- Add a test group: add a row to `questions.csv`, then examples that use it.
- Options are separated by ` | `. For type `choice` each option is
  `name: description` (description optional). For type `score` list the levels
  from lowest to highest, and use the level text as the expected answer.

`run_tests.py` checks both files on start and reports the row if a group name
or expected answer does not match. After changing the test cases, run every
model again; the app hides results from an older version of the file.

## Reading the results fairly

- The examples are few and made up. Treat scores as a first impression, not a
  benchmark.
- GLiNER receives option names only. Laya and Jev also receive a description
  per option, so the comparison slightly favours them.
- GLiNER2.5-Decide is English only, so expect misses in the languages group.
- Laya reads a limited amount of text, so the long-text group is expected to
  expose that.
- Jev's time per question includes the network round trip.
- Jev sends the test texts to an external server. Do not add confidential text
  to `testcases.csv` if you run Jev.

## Troubleshooting

| Problem | Fix |
|---|---|
| `command not found: python` | Activate an environment first, or use `python3` |
| `ModuleNotFoundError: No module named 'X'` | Activate the right environment, then `pip install X` |
| `No module named 'gliner2'` or `'laya'` | You are in the other environment; activate the matching one |
| Jev returns 401 | The key is wrong or not set; create a new one at OpenRouter |
| Jev returns 402 | The OpenRouter account has no credit |
| pip fails with `AssertionError` in the resolver | Run `python -m pip install --upgrade pip` and retry |
| Laya hangs while loading | Start it with `USE_TF=0 python run_tests.py laya` |
| The app shows "results come from an older version" | Run that model again |
| A model fails inside the app | Run it from the terminal as shown above to see the full error |

## Sources

- Laya: https://huggingface.co/convaiinnovations/laya
- GLiNER2.5-Decide: https://huggingface.co/fastino/GLiNER2.5-Decide
- Jev on OpenRouter: https://openrouter.ai/docs/guides/community/jev.md
