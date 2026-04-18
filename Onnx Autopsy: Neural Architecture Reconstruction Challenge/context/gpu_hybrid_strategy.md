# ONNX Autopsy: Hybrid CPU+GPU Strategy for Maximum Score and Speed

## 1) Goal and Design Principles

Primary objective:
- Maximize leaderboard score (NED) while minimizing end-to-end runtime.

Practical objective:
- Use local CPU-friendly workflows for fast iteration.
- Exploit A10G GPU for the expensive learning and retrieval parts where it provides real gains.

Key principles:
1. Prefer exact reconstruction when possible (higher score ceiling).
2. Use learned retrieval only when exact parsing is uncertain.
3. Make fallback retrieval GPU-native so scale-up is natural.
4. Use confidence-based routing instead of one-size-fits-all inference.
5. Keep deterministic behavior for reproducibility and easier debugging.

---

## 2) Why the Current Pipeline Leaves GPU Underused

The current `solution.py` approach is strong but CPU-centric:

- Sparse TF-IDF retrieval via sklearn vectorizers.
- Cosine similarity via sklearn `linear_kernel` on sparse matrices.
- Optional learned reranker via sklearn `HistGradientBoostingRegressor`.
- Python fallback retriever for environments without sklearn.

This design has two consequences:
1. It cannot naturally leverage A10G for heavy math.
2. It is sensitive to NumPy/SciPy/sklearn binary compatibility issues.

In short, this is good for portability, but not ideal for fully utilizing challenge GPU resources.

---

## 3) Proposed Hybrid Architecture (Expert System)

Use three experts and a confidence router:

- Expert A: Deterministic ONNX parser (exact-first path).
- Expert B: GPU dense embedding retriever (learned fallback path).
- Expert C: Existing sparse retriever (safe fallback and sanity path).

High-level flow:
1. Try deterministic parse from raw ONNX bytes.
2. If parse confidence is high, return parsed sequence directly.
3. If parse confidence is low, use GPU retriever + reranker.
4. If GPU path fails or confidence is still low, use CPU sparse fallback.
5. Optionally ensemble near ties using confidence-weighted blending.

---

## 4) Expert A: Deterministic ONNX Parser (Score Driver)

Why this matters:
- For this challenge type, exact graph reconstruction often outperforms approximate retrieval.
- A deterministic parser can produce near-perfect outputs on many samples.

Core parser responsibilities:
1. Decode hex to bytes robustly.
2. Parse ONNX protobuf model graph.
3. Extract node operator types in canonical execution order.
4. Normalize aliases and deterministic ordering edge cases.
5. Output target sequence.

Confidence signals for parser routing:
- Successful protobuf parse.
- Graph integrity checks pass.
- Node count in plausible range.
- Operator vocabulary overlap with train distribution.
- Minimal unknown/unsupported ops.

Example confidence score:
- `conf_parse = w1*valid_parse + w2*graph_integrity + w3*vocab_match - w4*unknown_op_ratio`
- If `conf_parse >= threshold_parse`, trust parser output.

Expected impact:
- Largest score jump among all components if parser quality is high.

---

## 5) Expert B: GPU Dense Retriever (Speed + Robust Fallback)

Why not only sparse TF-IDF:
- Sparse sklearn stack is CPU-first.
- Dense embedding retrieval maps naturally to GPU + FAISS.

Recommended stack:
- PyTorch for embedding model training/inference.
- FAISS GPU for nearest neighbor search.
- Optional XGBoost GPU or small MLP for reranking.

Pipeline:
1. Convert hex to byte tokens.
2. Encode bytes with a compact byte-level model (CNN/Transformer-lite).
3. Produce fixed-length embedding vector.
4. Build FAISS GPU index on train embeddings.
5. Retrieve top-k nearest neighbors for each query.
6. Construct candidate sequences from neighbors.
7. Rerank candidates with feature model.

Candidate features to keep from current approach:
- Expected NED proxy.
- Neighbor support/vote count.
- Length prior penalty.
- Sequence prior score.
- Base similarity aggregate.

Expected impact:
- Better scalability and stronger GPU utilization.
- Often faster batch inference once index and embeddings are ready.

---

## 6) Expert C: Existing Sparse Retriever (Safety Net)

Keep current sparse retriever as a robust fallback:
- Useful when GPU dependencies are unavailable.
- Useful for A/B testing and regression checks.
- Helps isolate whether score regressions come from parser or dense model.

Do not delete this path early.
Treat it as a control arm during migration.

---

## 7) Confidence Router and Ensembling

Use confidence-aware selection instead of always selecting one expert globally.

Per-sample routing proposal:
1. Compute `conf_parse`.
2. If `conf_parse >= T_parse_high`, return parser sequence.
3. Else run GPU retriever and compute `conf_gpu`.
4. If `conf_gpu >= T_gpu_high`, return GPU best sequence.
5. Else compare parser and GPU top candidates with a tie-break model.
6. If both low-confidence, call sparse fallback and vote among top options.

Tie-break rule candidates:
- Confidence-weighted score.
- Prior-adjusted agreement with train sequence motifs.
- Sequence length plausibility by ONNX byte length bucket.

This routing usually improves both score and worst-case reliability.

---

## 8) Local vs A10G Profiles

Define two runtime profiles:

Local fast-iterate profile:
- Smaller train subset for quick iteration.
- Lower top-k.
- Fewer reranker candidates.
- CPU-compatible path preferred for developer loop.

A10G full-power profile:
- Full train data.
- Larger embedding model and/or test-time augmentations.
- Larger top-k retrieval and stronger reranker.
- Multi-seed ensembling across checkpoint variants.

Profile switching should be explicit via environment variables so behavior is reproducible.

---

## 9) Benchmark Plan (Score and Speed)

Measure both quality and cost at each phase.

Core metrics:
- Validation NED (mean).
- P50/P95 inference latency per sample.
- End-to-end wall time for eval split.
- GPU memory usage and utilization.

Required ablations:
1. Sparse baseline only.
2. Parser only.
3. GPU retriever only.
4. Parser + GPU router.
5. Parser + GPU + sparse fallback.
6. Final ensemble with confidence blending.

Decision rule:
- Promote only changes that improve NED with neutral or better wall time.
- If NED gains are significant, allow moderate runtime increase only when within challenge limits.

---

## 10) Dependency and Environment Strategy

To avoid binary mismatch issues:
1. Use an isolated virtual environment.
2. Avoid mixing system SciPy with user-site NumPy.
3. Keep CPU sparse stack and GPU stack optional but explicitly pinned.

Suggested package groups:
- Core CPU: numpy, scipy, scikit-learn.
- GPU retrieval: torch (CUDA), faiss-gpu.
- Optional GPU rerank: xgboost with GPU support.

Use locked versions per profile to keep runs reproducible.

---

## 11) Phased Implementation Roadmap

Phase 1: Stabilize baseline
- Fix environment and reproducibility.
- Lock CPU baseline metrics.

Phase 2: Deterministic parser
- Implement parser and confidence scoring.
- Validate parser-only score.

Phase 3: GPU dense retrieval
- Train byte encoder.
- Integrate FAISS GPU search.
- Add candidate builder compatibility layer.

Phase 4: Router + fallback
- Add confidence routing and tie-break logic.
- Keep sparse fallback for low-confidence cases.

Phase 5: Ensembling and tuning
- Multi-seed models, confidence blending, threshold sweeps.
- Finalize A10G profile for submission generation.

---

## 12) Risks and Mitigations

Risk: parser errors on malformed or unusual models.
- Mitigation: strict validation + automatic fallback routing.

Risk: GPU stack complexity and dependency friction.
- Mitigation: optional modules, clear profile flags, lockfile-based installs.

Risk: overfitting confidence thresholds to one local split.
- Mitigation: repeated split validation and threshold averaging.

Risk: speed regressions from excessive ensembling.
- Mitigation: cap ensemble width; use adaptive ensembling only for low-confidence samples.

---

## 13) Practical Recommendation

For maximum score with practical engineering effort:
1. Build deterministic parser first.
2. Add GPU dense retrieval second.
3. Keep sparse fallback path.
4. Deploy confidence router.
5. Tune thresholds and ensemble depth on A10G.

This strategy is the best fit for "score first, speed second, but still scalable" and aligns with the challenge providing an A10G GPU.
