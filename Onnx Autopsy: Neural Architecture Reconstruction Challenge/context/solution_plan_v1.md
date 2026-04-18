# ONNX Autopsy - Max-Score Kaggle-First Master Plan

Date: 2026-04-18
Status: approved-for-implementation after user confirmation

## 1) Score Targets
1. Minimum target: beat payout baseline > 0.8733
2. Primary target: push into 0.92-0.95 range with robust validation
3. Stretch target: 0.98-0.99 as an optimization objective, not a guaranteed outcome

## 2) Non-Negotiable Execution Policy
1. All heavy testing and tuning happens on Kaggle runtime (T4 x2)
2. No long local benchmarking loops
3. Local machine is used only for final artifact generation:
4. Run solution.py once
5. Produce ./working/submission.csv
6. Submit solution.py + submission.csv

## 3) Metric Alignment (Exact)
1. Every experiment is ranked by exact mean NED
2. Validation protocol is fixed and versioned:
3. random split NED
4. stress split NED (longest hex-heavy)
5. weighted proxy = 0.3 * random + 0.7 * stress
6. Promotion rule: weighted proxy must improve by >= 0.005

## 4) Phase Roadmap

### Phase A - Kaggle Harness and Reproducibility (Day 1)
1. Build a single Kaggle experiment harness notebook
2. Fix seeds and deterministic split IDs
3. Log every run to an experiments table
4. Record runtime, random NED, stress NED, weighted proxy, config hash
5. Freeze this harness to prevent metric drift

Deliverable:
- repeatable experiment runner in Kaggle

### Phase B - Retrieval Engine Optimization (Days 1-4)
1. Tune retrieval core aggressively (highest ROI first)
2. Sweep char/byte n-gram ranges, feature dimensions, view weights
3. Sweep top_k, batch_size, candidate caps, rerank depth
4. Add family-aware retrieval shards and compare against global index
5. Add reciprocal-rank and similarity-calibrated weighting

Deliverable:
- best retrieval config with consistent weighted proxy gain

### Phase C - Learned Candidate Reranker (Days 4-8)
1. Keep retrieval as candidate generator
2. Train a lightweight reranker on train-only pseudo-label setup
3. For each row, generate candidate sequences and score candidate-vs-truth NED in training folds
4. Train reranker to predict candidate quality using only train-derived features
5. Use reranker at inference to pick final candidate

Candidate features:
1. retrieval similarity stats
2. candidate length delta
3. sequence prior score
4. token-position agreement with neighbor consensus
5. family and length priors

Deliverable:
- retrieval + learned reranker stack

### Phase D - Neural Model from Scratch (Days 8-18)
1. Train a compact byte-to-sequence model in Kaggle
2. Architecture family:
3. byte/chunk encoder (hierarchical)
4. sequence decoder over layer tokens
5. length head for calibration
6. Start with small model for quick iteration, then scale depth/width

Default-safe augmentation ON:
1. token masking
2. token dropout
3. multi-view chunk crops preserving prefix and suffix context

Hard disallow list:
1. target reordering
2. target rewriting
3. destructive truncation that breaks label semantics

Deliverable:
- neural model checkpoint and inference routine with reproducible config

### Phase E - Ensemble and Calibration (Days 18-25)
1. Build candidate pools from retrieval and neural outputs
2. Calibrate with validation-based blending
3. Use confidence-aware fallback rules for hard rows
4. Add strict JSON guards and schema checks
5. Freeze best ensemble variant

Deliverable:
- final ensemble candidate selector ready for submission mode

### Phase F - Submission Playbook (Days 25-30)
1. Use credits only on promoted models
2. Keep final reserve credit
3. For each submission:
4. run final Kaggle validation
5. update solution.py with proven config
6. run local once to generate ./working/submission.csv
7. submit two files

Deliverable:
- final score-maximized submission package

## 5) Experiment Budget Rules
1. Fast loop budget: 10-20 min per config
2. Mid loop budget: 30-60 min per promising config
3. Full loop budget: only for finalists
4. Any run that exceeds budget without clear signal is terminated

## 6) Risk Controls
1. Overfit to random split
2. Mitigation: stress split dominates promotion score
3. Metric mismatch
4. Mitigation: exact NED implementation reused across all phases
5. Invalid output format
6. Mitigation: mandatory JSON + schema validation before every submission build
7. Compute waste
8. Mitigation: Kaggle-first policy and experiment budget caps

## 7) Definition of Done for Coding Start
1. Kaggle harness built and logging
2. Baseline reproduced in harness
3. Phase B sweep matrix defined
4. Promotion thresholds locked

When these are complete, coding proceeds immediately in this order:
1. harness utilities
2. retrieval sweeps
3. reranker training
4. neural model
5. ensemble
