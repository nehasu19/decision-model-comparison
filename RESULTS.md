# Results

Latest runs, November 2026. All cases are short and made up, so treat these as
a first impression, not a benchmark.

## Main test set: 74 cases in 10 groups

| Test group (cases) | Laya | GLiNER2.5-Decide | Jev |
|---|---|---|---|
| Intent routing (8) | 6 | 8 | 8 |
| Sentiment (6) | 6 | 6 | 6 |
| Spam and phishing (6) | 6 | 6 | 6 |
| Complaint, yes or no (6) | 6 | 6 | 6 |
| Urgency rating (6) | 4 | 6 | 6 |
| Choosing among 15 topics (8) | 8 | 8 | 8 |
| Four languages (12) | 10 | 12 | 12 |
| Long text (3) | 2 | 3 | 3 |
| Arithmetic and counting (6) | 3 | 3 | 6 |
| RAG answer correctness (13) | 9 | 5 | 13 |
| **All 74 cases** | **60 (81%)** | **63 (85%)** | **74 (100%)** |

OpenDecider nano has been run on this set; its scores are still to be added.

## Hallucination test set: 48 cases

Each case gives a context, a question and an answer. The model labels the
answer `correct`, `incorrect` (contradicts the context) or `unsupported`
(states something the context never mentions, i.e. made up). "Caught" means
the bad answer was not passed as correct.

| | Laya | GLiNER2.5-Decide | OpenDecider nano |
|---|---|---|---|
| Exact label right | **31/48** | 23/48 | 26/48 |
| Made-up answers caught | **4/16** | 3/16 | 0/16 |
| Wrong answers caught | **15/16** | 8/16 | 10/16 |
| False alarms on correct answers | 0/16 | 1/16 | 0/16 |
| Average response time | 285 ms | 601 ms | not recorded |

Jev has not been run on this set yet.

## Main findings

- Jev answered all 74 main cases correctly; the local models lost points on
  arithmetic and RAG answer correctness.
- None of the three local models reliably catches made-up answers. They spot
  answers that contradict the context, but not answers that add facts the
  context never mentions.
- Laya is the best local model at checking answers against a context, mainly
  because it catches contradictions (15 of 16).
- OpenDecider never chose `unsupported`, so it caught none of the made-up
  answers, despite strong results on its authors' own benchmark.

## Reproduce

```bash
python run_tests.py laya                         # main set
python run_tests.py laya --data hallucination    # hallucination set
python hallucination/report.py                   # hallucination summary
```

Replace `laya` with `gliner`, `opendecider` or `jev`. See the README for setup.
