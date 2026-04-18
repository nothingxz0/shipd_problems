# ONNX Autopsy: Problem and Dataset Analysis

Date: 2026-04-18
Scope: Analysis only (no modeling code changes yet)

## 1) Task Understanding
- Input: onnx_hex (hex-encoded, optimized ONNX graph bytes).
- Output: target_sequence as a JSON array string of PyTorch layer names in forward order.
- Evaluation: Mean NED (normalized Levenshtein over token sequences), higher is better.
- Submission file: ./working/submission.csv with columns id,target_sequence.

## 2) Hard Constraints From Prompt
- No pretrained weights; train from scratch using train.csv only.
- No onnx / onnxruntime at inference time.
- Custom tokenizer required (no pretrained tokenizers).
- No external data.
- Reproducibility requirement includes torch.manual_seed(42).
- Runtime environment requirement: self-contained script, reads from ./dataset/public/, writes to ./working/submission.csv.

## 3) Verified Dataset Profile (from actual CSV files)
- train rows: 8000
- test rows: 2000
- train columns: id, onnx_hex, target_sequence
- test columns: id, onnx_hex
- all ids unique in train and test
- exact train/test onnx_hex overlap: 0 rows
- duplicate onnx_hex inside train: 0 rows

### Hex Length Distribution
- train: min 884, median 4282, p90 18764, max 72352
- test: min 884, median 9881, p90 51485.8, max 93680
- test is substantially longer than train (clear distribution shift to harder samples)
- fraction of test above train quantiles:
  - > train p90: 27.1%
  - > train p95: 14.5%
  - > train p99: 8.1%

### Label Sequence Distribution (train)
- target length: min 3, median 11, p90 20, max 48
- buckets:
  - <=6: 2154
  - 7-12: 2279
  - 13-20: 2903
  - 21+: 664

### Input Format Integrity
- onnx_hex uses only lowercase hex chars: 0-9,a-f
- all rows have even hex length
- no invalid chars detected

## 4) Observed Label Vocabulary vs Prompt Vocabulary
Observed train vocabulary size: 27
Observed tokens:
- AdaptiveAvgPool2d, AvgPool2d, BatchNorm1d, BatchNorm2d, Conv1d, Conv2d, Dropout, ELU, Embedding, Flatten, GELU, GRU, Hardswish, Hardtanh, Identity, LSTM, LayerNorm, LeakyReLU, Linear, MaxPool2d, PReLU, RNN, ReLU, ReLU6, SiLU, Sigmoid, Tanh

Important mismatch notes:
- Identity appears in train labels but is not listed in prompt vocabulary.
- MultiheadAttention is listed in prompt but appears 0 times in train labels.
- Many prompt-listed classes are absent in train labels (for example Conv3d, ConvTranspose*, Dropout2d, AlphaDropout, Softmax, Upsample, PixelShuffle, etc.).

## 5) Label Pattern Signals (train)
- rows containing Linear: 100%
- rows ending with Linear: 7809 / 8000 (~97.6%)
- rows containing Dropout: 72.15%
- rows containing Conv2d: 44.0%
- rows containing RNN/GRU/LSTM: 19.0%
- rows containing Embedding and LayerNorm: both 13.425%
- rows containing Identity: 5.24%

Common transitions:
- Conv2d -> BatchNorm2d
- Dropout -> LayerNorm
- Dropout -> Linear
- AdaptiveAvgPool2d -> Flatten
- LayerNorm -> Linear
- Linear -> Dropout

## 6) Coarse Family Proxy (from target patterns)
Using simple token-presence heuristics on train labels:
- cnn_or_resnet proxy: 3520
- pure_mlp_like proxy: 1885
- rnn_family proxy: 1521
- transformer_like proxy: 1074

Note: These proxies are disjoint in train with this heuristic; no explicit hybrid token overlap was observed in labels themselves (hybrid may still exist in graph-level structure, or appear more in test as prompt claims).

## 7) Sample Submission Anomaly
- sample_submission.csv has 2000 rows and id order matches test.csv exactly.
- Only 1 row is non-empty (the first row has a concrete sequence); all others are [].
- Treat this as a possible illustrative/example row, not as a reliable leakage pattern.

## 8) Difficulty Drivers Confirmed by Data
- Strong train-test shift toward longer onnx_hex in test.
- Target sequence lengths vary widely (3 to 48), making fixed-template prediction brittle.
- NED rewards near-correct ordering/length; malformed JSON yields zero for a row.
- Since inference cannot use ONNX parsers, robust byte-level/hex-level modeling is mandatory.

## 9) Practical Implications for Next Phase (no coding yet)
- Prioritize methods robust to long context and test shift.
- Keep decoder outputs strictly within observed label space plus any prompt-required classes if justified.
- Add strong JSON-format guards in final inference pipeline.
- Build validation split that stresses long-hex examples to mimic hidden test hardness.

## 10) Environment Note During Analysis
- Current local Python env in this workspace did not include pandas; stdlib csv/json worked for analysis.
- This does not change challenge requirements; Kaggle/Shipd runtime is expected to include standard Kaggle stack.
