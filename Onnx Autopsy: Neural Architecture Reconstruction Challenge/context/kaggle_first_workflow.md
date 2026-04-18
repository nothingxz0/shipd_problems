# Kaggle-First Workflow (Simple)

Goal: iterate fast on Kaggle, avoid slow local experimentation.

## What the current solution is
- `solution.py` is retrieval-based.
- It now includes a lightweight learned candidate reranker on top of retrieval.
- It does NOT run a heavy neural training loop.
- It uses deterministic multi-view features (full/even/edge) but not stochastic training augmentation.

## Loop to follow
1. Tune and test approaches in Kaggle runtime (T4 x2).
2. Measure local validation NED in Kaggle before spending submission credits.
3. When an approach is better, update `solution.py` in workspace.
4. Run `solution.py` once locally only to generate `./working/submission.csv`.
5. Submit both files required by Shipd:
   - `solution.py`
   - generated `submission.csv`

## Performance controls in solution.py
- Fast stack required by default:
   - `SHIPD_REQUIRE_SKLEARN=1` (default)
   - if sklearn/numpy missing, script fails fast instead of running slow fallback
- Runtime profile selector:
   - `SHIPD_PROFILE=fast` for quickest iteration
   - `SHIPD_PROFILE=balanced`
   - `SHIPD_PROFILE=quality`
   - `SHIPD_PROFILE=quality_trim` (default)
- Learned reranker controls:
   - `SHIPD_ENABLE_RERANKER=1` (default)
   - `SHIPD_RERANKER_MAX_QUERIES=5000` (default)
   - `SHIPD_RERANKER_MAX_CANDS=24` (default)
   - `SHIPD_RERANKER_BLEND=0.70` (default)

## Recommended Kaggle tuning order
1. Keep `SHIPD_PROFILE=quality_trim` and `SHIPD_ENABLE_RERANKER=1`.
2. Sweep `SHIPD_RERANKER_MAX_QUERIES` in {3000, 5000, 6500}.
3. Sweep `SHIPD_RERANKER_BLEND` in {0.60, 0.70, 0.80}.
4. Promote only configs that improve weighted proxy and stress NED.

## Submission gate rule
- Submit only when Kaggle validation shows a clear gain vs previous best.
- Keep one credit in reserve for the final attempt.

## Current score checkpoint
- Latest submitted score: 0.8372
- Baseline to beat for payout: 0.8733
- Local baseline (before learned reranker): 0.86839
- Local with learned reranker (`MAX_QUERIES=3000`): 0.88079
- Local with learned reranker (`MAX_QUERIES=5000`): 0.88355
- Gap to baseline closed locally with current defaults.
