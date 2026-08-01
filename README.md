# steel-llm-eval

**How well can large language models predict knife-steel properties from chemical composition alone?**

An open, reproducible benchmark that gives an LLM only a steel's composition (e.g. `C=1.45%, Cr=20%, V=4%, Mo=1%, powder-metallurgy: yes`) and asks it to rate two properties on a 1–10 scale, then scores those ratings against **objective laboratory measurements**:

- **Edge retention** ← CATRA standardized machine-cutting test (total card stock cut, mm) — 48 steels
- **Toughness** ← Charpy impact energy (ft-lbs) — 12 steels

Scoring is **scale-free** (rank correlation + pairwise ranking accuracy), so a model is judged purely on whether it orders steels correctly, not on how it calibrates the 1–10 scale.

[![Code License: Apache 2.0](https://img.shields.io/badge/code-Apache%202.0-blue.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/data%20%26%20model-CC%20BY%204.0-lightgrey.svg)](data/LICENSE)

**📊 Site with charts & analysis:** https://steel-predictor-project.github.io/steel-llm-eval/ · **Docs:** [methodology](docs/methodology.md) · [results & analysis](docs/results-analysis.md)

---

## Leaderboard

Ranked by mean Spearman rank correlation (ρ) vs. the measurements. Higher is better; 1.0 = perfect ordering, 0.0 = random.

| Model | Edge ρ (n) | Edge pairwise | Tough ρ (n) | Tough pairwise | **Mean ρ** |
|---|---|---|---|---|---|
| steel-predictor (reference ML) † | 0.992 (48) | 0.98 | 0.946 (12) | 0.938 | **0.969** |
| anthropic/claude-sonnet-5 | 0.894 (48) | 0.918 | 0.844 (12) | 0.881 | **0.869** |
| google/gemini-3.6-flash | 0.918 (48) | 0.913 | 0.698 (12) | 0.797 | **0.808** |
| openai/gpt-4o | 0.868 (48) | 0.907 | 0.600 (12) | 0.746 | **0.734** |
| meta-llama/llama-3.3-70b-instruct | 0.864 (47) | 0.964 | 0.514 (12) | 0.780 | **0.689** |
| deepseek/deepseek-chat-v3.1 | 0.869 (48) | 0.910 | 0.380 (12) | 0.661 | **0.625** |
| openai/gpt-4o-mini | 0.850 (48) | 0.984 | 0.385 (12) | 0.689 | **0.617** |

_Edge retention n=48 (CATRA), toughness n=12 (Charpy). Zero-shot, temperature 0, one sample per steel._

**† Important fairness caveat:** the reference ML model ([Steel-predictor](https://github.com/Steel-predictor-project/Steel-predictor)) was **trained on these same CATRA/Charpy measurements**, so its scores here are largely *in-sample* and are shown as an upper-reference bar, **not** as a fair head-to-head with the zero-shot LLMs. The model's honest out-of-sample performance is its LOOCV MAE (0.391), reported in that repo. The LLMs, by contrast, have never seen this labeled set.

### What the numbers say
- **LLMs are genuinely good at ranking edge retention** (ρ ≈ 0.85–0.92). Wear resistance is strongly and legibly encoded in composition (carbide-forming elements — C, V, Cr, W, Mo), and frontier models clearly "know" that chemistry.
- **Toughness is where they struggle** (ρ 0.38–0.84). It depends on subtler factors (carbide size/distribution, powder-metallurgy processing, matrix state) that aren't obvious from a composition string, and the spread across models is large.
- **Frontier > small.** Claude Sonnet and Gemini lead; the smaller/cheaper models drop off sharply on toughness while staying competitive on edge retention.

---

## Reproduce

```bash
git clone https://github.com/Steel-predictor-project/steel-llm-eval.git
cd steel-llm-eval

export OPENROUTER_API_KEY=sk-or-...   # one key → OpenAI, Anthropic, Google, Meta, DeepSeek, ...
./run_benchmark.sh                    # runs every model and rebuilds the leaderboard
```

Run a single model, or a quick offline sanity check with no API key:

```bash
python harness/run_eval.py --model anthropic/claude-sonnet-5
python harness/run_eval.py --provider mock     # deterministic heuristic, no key needed
python harness/score.py
```

Raw per-steel responses are written to `results/raw_<model>.csv`; scores to `results/scores.csv` and `results/leaderboard.md`.

---

## How it works

1. **Prompt** (`harness/prompts.py`) — a fixed system + user prompt gives the model the composition, PM flag, and test hardness (when known) and asks for JSON: `{"edge_retention": n, "toughness": n}`. Identical for every model.
2. **Run** (`harness/run_eval.py`) — queries a model for all 51 steels via OpenRouter and parses the JSON.
3. **Score** (`harness/score.py`) — vs. the measurements:
   - **Spearman ρ** and **Kendall τ** rank correlation (headline; scale-free).
   - **Pairwise accuracy** — over all steel pairs, how often the model orders them the same way the measurement does (ties excluded).
   - **Normalized MAE** — a calibration sanity check after min-max scaling the measurement to 1–10 (secondary; see `scores.csv`).
4. **Baselines** — the purpose-built ML model (upper reference, in-sample caveat above) and a constant predictor (floor).

## Methodology notes & limitations
- **Ground truth is objective measurement only** (CATRA, Charpy). No subjective 1–10 expert ratings are used anywhere in scoring.
- **Composition-only.** Models are not told heat-treat protocol or geometry (only hardness where recorded), so this measures what chemistry *alone* implies — the same constraint the reference model operates under.
- **Small toughness set (n=12).** Treat toughness ρ as indicative, not definitive; single sample per steel at temperature 0 (no self-consistency / multi-sample averaging yet).
- **Rank metrics are primary** precisely because different models calibrate the 1–10 scale differently; ranking is what's comparable and decision-relevant.

## Data & provenance
The benchmark (`data/benchmark.csv`) is derived from the processed dataset of the [Steel-predictor](https://github.com/Steel-predictor-project/Steel-predictor) project; every underlying source (manufacturer datasheets, published CATRA/Charpy measurements) is cited in that repo's `DATA_SOURCES.md`. Underlying factual measurements remain the property of their original publishers.

## License
Code: **Apache-2.0** (`LICENSE`). Curated benchmark data + reference model outputs: **CC BY 4.0** (`data/LICENSE`), covering only this project's compilation/derived features. Attribution requested: "Steel Property Predictor Project" with a link to this repo.

## Contributing
PRs welcome to add models (extend the list in `run_benchmark.sh`), prompt variants (few-shot, chain-of-thought, self-consistency), or additional measured steels (with cited public sources). Please don't add subjective-rating datasets as ground truth.
