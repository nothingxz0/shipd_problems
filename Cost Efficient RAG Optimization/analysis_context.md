# Cost Efficient RAG Optimization - Analysis Context

## 1) Task Understanding

You must output, for each test query:
- `id`
- `evidence_ids` (comma-separated chunk IDs like `C3,C7`; can be empty)
- `answer` (short text)

Objective is a balanced metric:

- `score = 0.5 * answer_F1 + 0.5 * evidence_efficiency`
- `evidence_efficiency = 0.3 + 0.4 * precision + 0.3 * compression`
- `precision = |selected ∩ gold| / |selected|`
- `compression = 1 - |selected| / total_pool_size`

Key interpretation:
- Answer quality and retrieval efficiency matter equally.
- Over-retrieval is explicitly penalized.
- High compression is rewarded independently of answer quality.

## 2) Dataset Profile

Files in `dataset/public/`:
- `train.csv`: 2766 rows
- `test.csv`: 923 rows
- `sample_submission.csv`: 923 rows

Columns:
- Train: `id, query, context, num_chunks, evidence_ids, answer`
- Test: `id, query, context, num_chunks`
- Sample: `id, evidence_ids, answer`

Sample submission defaults:
- `evidence_ids` empty for all rows
- `answer = unknown` for all rows

Train/test ID overlap:
- None (`id_overlap = 0`)

## 3) Chunk Pool Size and Integrity

`num_chunks` distributions:

Train:
- 9:1, 10:7, 11:17, 12:48, 13:97, 14:192, 15:322, 16:377, 17:426, 18:369, 19:347, 20:245, 21:165, 22:83, 23:55, 24:13, 25:2

Test:
- 11:3, 12:14, 13:49, 14:64, 15:94, 16:140, 17:122, 18:135, 19:104, 20:95, 21:50, 22:30, 23:18, 24:4, 25:1

Integrity checks:
- Test: `num_chunks` matches parsed chunk markers for all rows.
- Train: one real annotation anomaly + one regex false-positive if naive parsing is used.

Important parsing note:
- Count chunk IDs using line-start markers (`^\[C\d+\]`, multiline), not any bracket occurrence in text.
- One train row (`q_637607f20d`) contains out-of-sequence marker `C96` in context, creating a mismatch (`num_chunks=18`, parsed markers=19).
- Another row (`q_770877608c`) has a repeated in-text marker token (duplicate `C4`) that can fool naive regex counting.

## 4) Gold Evidence Label Statistics (Train)

Evidence count distribution (`len(evidence_ids)`):
- 0:555, 1:410, 2:330, 3:301, 4:187, 5:223, 6:189, 7:154, 8:134, 9:96, 10:57, 11:49, 12:29, 13:20, 14:16, 15:10, 16:4, 17:1, 18:1

Observations:
- 555/2766 (~20.1%) rows have empty gold evidence.
- Evidence IDs are valid and clean:
  - all evidence IDs are subset of context chunk IDs
  - no duplicate evidence IDs in a row
  - evidence ID numeric range is `C1..C24`
- 5 rows have `|gold| == num_chunks` (no compression possible if you must keep all gold).

Gold-to-pool ratio:
- min: 0.0
- mean: ~0.2152
- max: 1.0

## 5) Query, Context, and Answer Lengths

Query token length:
- Train: min 3, p25 8, median 11, p75 16, p95 23, max 34, mean 12.20
- Test:  min 3, p25 8, median 11, p75 15, p95 24, max 34, mean 12.45

Context character length:
- Train: min 4552, p25 7597, median 8565, p75 9546, p95 10852, max 12629, mean 8646.95
- Test:  min 5474, p25 7742, median 8569, p75 9543, p95 10768, max 12613, mean 8643.62

Answer token length (train):
- min 0, p25 1, median 2, p75 3, p95 12, max 135, mean 3.49
- One-token answers: 1002/2766 (~36.2%)
- Empty answers: 3

## 6) Answer Format Quirks

Answers are not uniformly simple strings.

Notable label artifacts:
- 1182/2766 (~42.7%) are bracketed list-like serialized values, e.g. `['English Language']`
- 1200/2766 (~43.4%) contain quote characters
- 388/2766 (~14.0%) contain commas
- Numeric-like answers: 244/2766 (~8.8%)
- `unknown` appears 0 times in train answers

Implication:
- Post-processing/normalization matters a lot for answer_F1.
- Naive generation may lose points due to formatting mismatch.

## 7) Grounding and Extractiveness Signals

Using simple normalized substring checks (lowercase, punctuation/article stripped):
- answer appears in full context for ~60.74% (1680/2766)
- answer appears in concatenated gold evidence for ~52.17% (1443/2766)

For list-like answers only:
- all list items found in gold evidence: ~56.18%
- at least one list item found in gold evidence: ~62.35%

For zero-evidence rows:
- 555 rows total
- answer appears in full context for only ~27.21% of them

Implication:
- Task is not purely extractive; some answers require transformation, aggregation, normalization, or handling noisy/misaligned labels.

## 8) Content Modality Mix (Heuristic)

Heuristic chunk-style prevalence is very similar in train and test:

Train total chunks: 47949
- KG-like (`-->` or ` -- `): 12672 (26.43%)
- Table-like (` | ` patterns): 11898 (24.81%)
- URL/wiki-like references: 12606 (26.29%)
- Math-like fragments: 4849 (10.11%)
- Very long lines (>260 chars): 14730 (30.72%)

Test total chunks: 16009
- KG-like: 4161 (25.99%)
- Table-like: 3892 (24.31%)
- URL/wiki-like: 4210 (26.30%)
- Math-like: 1512 (9.44%)
- Very long lines (>260 chars): 4922 (30.75%)

Implication:
- Heterogeneous chunk types are a core challenge; same distribution shift is minimal between train/test.

## 9) Metric Pressure: Efficiency Bounds (Train)

From gold stats:
- If selecting all chunks, evidence_eff mean is low (~0.3861).
- If selecting exactly gold chunks, evidence_eff mean is high (~0.9354).

Fixed-size random-selection thought experiment (assuming precision=0 for empty selection):
- select 0 chunks: avg evidence_eff ~0.6000
- select 1 chunk: avg evidence_eff ~0.6684
- select 3 chunks: avg evidence_eff ~0.6329
- select 5 chunks: avg evidence_eff ~0.5975
- select 10 chunks: avg evidence_eff ~0.5090

Implication:
- Compression reward is strong; selecting fewer chunks has major upside.
- But overall score still depends 50% on answer_F1, so overly aggressive pruning can hurt if answer quality collapses.

## 10) Practical Implications Before Modeling

- Robust chunk parser is mandatory (line-start marker extraction, tolerate malformed IDs).
- Retrieval should be selective and confidence-aware, not "select all".
- Need explicit handling for:
  - empty-evidence-like cases
  - serialized list answers
  - numeric/date normalization
  - heterogeneous chunk modalities (text/table/KG)
- Validation should track both:
  - answer F1
  - evidence precision/compression and combined evidence_eff

## 11) Open Ambiguity To Verify During Implementation

- Precision when `selected_chunks` is empty is not explicitly defined in the prompt formula (division by zero case). Code should mirror organizer evaluation convention exactly once confirmed.

