# ONNX Autopsy Plan v2 (Target >= 0.95)

Date: 2026-04-18
Mode: Planning only (no coding in this step)

## Status Update (Implemented)
- Phase A implemented in `solution.py` (multi-view retrieval + stronger reranking defaults).
- Phase B partially implemented in `solution.py` via learned tabular reranker.
- Latest local checkpoint improved from 0.86839 to 0.88355 on the eval split.
- Remaining work is Phase C (neural reranker) and Phase D (ensemble calibration).

## 1) Target and Rules
- Current public score: 0.8372
- Payout baseline to beat: 0.8733
- New target: >= 0.95
- Runtime policy: Kaggle-first experimentation, local only for final `solution.py` run and CSV generation

## 2) Success Metric and Promotion Gates
- Primary metric: exact NED
- Validation protocol is fixed:
  - random split NED
  - stress split NED (longest hex-heavy)
  - weighted proxy = 0.3 * random + 0.7 * stress
- Promotion rule for every approach change:
  - weighted proxy must improve by >= 0.005
  - and stress NED must not regress

## 3) Phase A (Immediate) - Beat Baseline Reliably
Goal: move from 0.8372 to >= 0.88 quickly and safely

Actions:
- Run full retrieval sweep matrix in Kaggle harness
- Extend profile matrix beyond fast/balanced/quality:
  - dynamic top_k by query length bucket
  - more aggressive candidate caps for long samples
  - adjusted view weights (full/even/edge)
- Add robust fallback rules:
  - consensus candidate fallback
  - frequent-sequence fallback for low-confidence rows

Exit gate:
- weighted proxy >= 0.885 and stress NED improvement over current best

## 4) Phase B - Learned Candidate Reranker (High ROI)
Goal: push into 0.90-0.93 zone

Actions:
- Keep retrieval as candidate generator (top 50-80)
- Create training table from folds:
  - features per candidate: similarity stats, rank stats, length delta, prior score, token-position agreement
  - target: candidate NED to ground truth
- Train lightweight reranker in Kaggle (tree-based regression/ranking)
- Inference: pick highest predicted-NED candidate per sample

Exit gate:
- weighted proxy >= 0.92 and stable stress gain

## 5) Phase C - Neural Reranker From Scratch
Goal: approach 0.94-0.95+

Actions:
- Build compact byte-candidate scorer (from scratch, no pretrained weights)
- Train on fold-generated candidates:
  - input A: compressed byte/chunk representation of onnx_hex
  - input B: candidate layer sequence embedding
  - objective: pairwise ranking or direct NED regression
- Safe augmentation ON by default:
  - byte masking
  - token dropout
  - multi-view crops preserving prefix and suffix context

Exit gate:
- weighted proxy >= 0.94 with no stress collapse

## 6) Phase D - Final Ensemble and Calibration
Goal: cross 0.95 with stability

Actions:
- Blend three scores:
  - retrieval score
  - tabular reranker score
  - neural reranker score
- Learn blend weights by length bucket and confidence bucket
- Apply confidence-aware routing:
  - very high confidence -> top blended candidate
  - low confidence -> fallback to stronger retrieval prior
- Final strict output guard:
  - enforce valid JSON list string per row

Exit gate:
- weighted proxy >= 0.95 and best seed-consistent result

## 7) Submission Credit Strategy (6 Credits)
- Credit 1 already used (0.8372)
- Next credits only after gate passes:
  - Credit 2: first > baseline run (>= 0.88 expected)
  - Credit 3: reranker jump run
  - Credit 4: neural reranker run
  - Credit 5: ensemble run
  - Credit 6: final reserved only if offline gain >= 0.005

## 8) Runtime Budget Discipline
- Fast loop config sweep: 10-20 min per batch
- Mid loop reranker: 20-45 min per run
- Neural loop: capped epochs with early stop on stress NED
- Kill any run that exceeds budget without interim gain

## 9) Deliverable Flow (unchanged)
- Experiment and tune on Kaggle
- Port best logic into workspace `solution.py`
- Run local once to generate `./working/submission.csv`
- Submit exactly:
  - `solution.py`
  - generated `submission.csv`

## 10) First Implementation Block After Approval
- Execute Phase A sweep to lock strongest retrieval config
- Immediately start Phase B candidate-table + reranker pipeline
