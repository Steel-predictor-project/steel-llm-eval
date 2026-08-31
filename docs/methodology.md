# Methodology

This document describes exactly how the benchmark is constructed, run, and scored, so the results can be reproduced and critiqued.

## 1. The question

Given **only a steel's chemical composition** (plus whether it is powder-metallurgy, and its test hardness where recorded), how well can a large language model predict two performance properties that we have **objective laboratory measurements** for?

- **Edge retention** — how long the edge keeps cutting (wear resistance).
- **Toughness** — resistance to chipping and breaking under impact.

We deliberately restrict ground truth to *measured* quantities. No subjective 1–10 expert ratings are used anywhere in scoring.

## 2. The dataset

`data/benchmark.csv` — 51 steels, derived from the processed dataset of the [Steel-predictor](https://github.com/Steel-predictor-project/Steel-predictor) project.

| Property | Ground-truth measurement | Units | n steels |
|---|---|---|---|
| Edge retention | CATRA total cards cut (TCC) | mm | 48 |
| Toughness | Charpy unnotched impact energy | ft-lbs | 12 |

Each row carries the composition (C, Cr, V, Mo, W, Co, N, Mn, Si, Nb, Ni), a `powder_metallurgy` flag, and `catra_test_hrc` (hardness at test) where known.

**Attribution.** The ground-truth measurements are not ours:
- **Edge retention (CATRA, 48 steels)** comes from Larrin Thomas, *"Testing the Edge Retention of 48 Knife Steels"* (2020), [KnifeSteelNerds.com](https://knifesteelnerds.com/2020/05/01/testing-the-edge-retention-of-48-knife-steels/). This benchmark's edge-retention ground truth is entirely his published CATRA data — full credit to him.
- **Toughness (Charpy, 12 steels)** comes from [Crucible Industries](https://www.crucible.com/) datasheets.
- **Compositions/hardness** come from manufacturer datasheets and literature.

Every underlying source is enumerated in the Steel-predictor repo's [`DATA_SOURCES.md`](https://github.com/Steel-predictor-project/Steel-predictor/blob/main/DATA_SOURCES.md); the underlying factual measurements remain the property of their original publishers.

## 3. The task given to each model

A fixed system + user prompt (`harness/prompts.py`) supplies the composition and asks for a compact JSON object:

```
{"edge_retention": <1-10>, "toughness": <1-10>}
```

- **Zero-shot.** No examples, no fine-tuning, no retrieval.
- **Temperature 0**, one sample per steel (no self-consistency / averaging).
- Identical prompt for every model, routed through OpenRouter so the harness is provider-agnostic.

## 4. Scoring

Predictions are compared to the measurements in `harness/score.py`.

Because different models calibrate the 1–10 scale differently, the **headline metrics are scale-free rank statistics** — they judge only whether a model *orders steels correctly*, which is the decision-relevant question ("is steel A tougher than steel B?").

- **Spearman ρ** — rank correlation between predicted score and measured value. `1.0` = perfect ordering, `0.0` = random. *(primary)*
- **Kendall τ-b** — a second rank-correlation view. *(reported in `scores.csv`)*
- **Pairwise accuracy** — over all pairs of steels, the fraction ordered the same way as the measurement (ties excluded). Easy to read, and it still means something with only a dozen steels. *(primary)*
- **Normalized MAE** — after min-max scaling the measurement to 1–10, the mean absolute error vs. the predicted score. A calibration sanity check. *(secondary — sensitive to how a model uses the scale)*

## 5. Baselines

- **steel-predictor (reference ML)** — the purpose-built model from the sister repo. **Fairness caveat:** it was *trained on these very CATRA/Charpy measurements*, so its scores here are largely **in-sample** and are shown as an upper-reference bar, **not** a fair head-to-head with the zero-shot LLMs. Its honest out-of-sample number is the LOOCV MAE (0.391) reported in that repo.
- **baseline-constant** — predicts the mean for every steel; a floor (rank metrics are undefined / chance).

## 6. Limitations

- **Small toughness set (n = 12).** Treat toughness ρ as indicative, not definitive.
- **Composition-only.** Models are not given heat-treat protocol or edge geometry (only hardness when recorded). This measures what chemistry *alone* implies — the same constraint the reference model operates under. Real-world knife performance depends heavily on heat treatment and geometry.
- **Single sample, temperature 0.** No multi-sample averaging or prompt-ensembling yet.
- **Rank, not magnitude.** We measure ordering, not physically-calibrated predictions.
- **Contamination is possible but unlikely to dominate.** Some steels are widely discussed online; however, the *specific* CATRA/Charpy values are relatively obscure, and the rank task rewards genuine metallurgical reasoning rather than recall of a leaderboard.

## 7. Reproduce

```bash
export OPENROUTER_API_KEY=...
./run_benchmark.sh                 # run all models + rebuild scores/leaderboard
python harness/generate_charts.py  # regenerate the figures in docs/assets/
```
