#!/usr/bin/env bash
# Reproduce the steel-property LLM benchmark end to end.
# Requires OPENROUTER_API_KEY for real models (or use --provider mock offline).
set -euo pipefail

python -m pip install -r requirements.txt

MODELS=(
  "openai/gpt-4o"
  "openai/gpt-4o-mini"
  "anthropic/claude-sonnet-5"
  "google/gemini-3.6-flash"
  "meta-llama/llama-3.3-70b-instruct"
  "deepseek/deepseek-chat-v3.1"
)
for m in "${MODELS[@]}"; do
  echo "=== $m ==="
  python harness/run_eval.py --model "$m" || echo "  (skipped: $m unavailable)"
done

python harness/score.py
echo
echo "Done. See results/leaderboard.md"
