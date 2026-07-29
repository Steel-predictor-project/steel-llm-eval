# Leaderboard — LLMs predicting knife-steel properties from composition

Ranked by mean Spearman rank correlation vs. objective lab measurements (CATRA edge retention, Charpy toughness). Higher is better; 1.0 = perfect ordering, 0.0 = random.

| Model | Edge ρ (n) | Edge pairwise | Tough ρ (n) | Tough pairwise | **Mean ρ** |
|---|---|---|---|---|---|
| steel-predictor (reference ML) | 0.992 (48) | 0.98 | 0.946 (12) | 0.938 | **0.969** |
| anthropic/claude-sonnet-5 | 0.894 (48) | 0.918 | 0.844 (12) | 0.881 | **0.869** |
| google/gemini-3.6-flash | 0.918 (48) | 0.913 | 0.698 (12) | 0.797 | **0.808** |
| openai/gpt-4o | 0.868 (48) | 0.907 | 0.6 (12) | 0.746 | **0.734** |
| meta-llama/llama-3.3-70b-instruct | 0.864 (47) | 0.964 | 0.514 (12) | 0.78 | **0.689** |
| deepseek/deepseek-chat-v3.1 | 0.869 (48) | 0.91 | 0.38 (12) | 0.661 | **0.625** |
| openai/gpt-4o-mini | 0.85 (48) | 0.984 | 0.385 (12) | 0.689 | **0.617** |
| baseline-constant | - (48) | - | - (12) | - | **-** |

_Edge retention n=48 steels (CATRA), toughness n=12 steels (Charpy)._
