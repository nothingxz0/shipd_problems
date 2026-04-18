#!/usr/bin/env python3
"""
ONNX Autopsy baseline+ plan implementation.

This script is self-contained and runs with zero CLI args.
It reads train/test from dataset folders and writes ./working/submission.csv.

Core approach:
- Retrieval over raw hex using custom byte-level views.
- Default augmentation is enabled via multi-view retrieval:
  - full hex view
  - even-byte downsampled view
  - edge-focused prefix/suffix view
- Candidate sequences are reranked with length and sequence priors.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SEED = 42
random.seed(SEED)


# Performance tuning knobs.
CPU_COUNT = max(1, (os.cpu_count() or 1))
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_COUNT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_COUNT))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_COUNT))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_COUNT))

PROFILE_PRESETS: Dict[str, Dict[str, int]] = {
	"fast": {
		"top_k": 24,
		"batch_size": 256,
		"metric_neighbors": 14,
		"max_candidates": 32,
	},
	"balanced": {
		"top_k": 32,
		"batch_size": 128,
		"metric_neighbors": 20,
		"max_candidates": 48,
	},
	"quality": {
		"top_k": 40,
		"batch_size": 64,
		"metric_neighbors": 28,
		"max_candidates": 64,
	},
	"quality_trim": {
		"top_k": 36,
		"batch_size": 128,
		"metric_neighbors": 24,
		"max_candidates": 64,
	},
}

ACTIVE_PROFILE_NAME = os.getenv("SHIPD_PROFILE", "quality_trim").strip().lower()
if ACTIVE_PROFILE_NAME not in PROFILE_PRESETS:
	ACTIVE_PROFILE_NAME = "quality_trim"
ACTIVE_PROFILE = PROFILE_PRESETS[ACTIVE_PROFILE_NAME]

# Default to requiring the fast sklearn stack to avoid very slow pure-python fallback runs.
REQUIRE_SKLEARN = os.getenv("SHIPD_REQUIRE_SKLEARN", "1").strip() != "0"

# Learned reranker controls.
RERANKER_ENABLED = os.getenv("SHIPD_ENABLE_RERANKER", "1").strip() != "0"
RERANKER_MAX_QUERIES = int(os.getenv("SHIPD_RERANKER_MAX_QUERIES", "5000"))
RERANKER_MAX_CANDS_PER_QUERY = int(os.getenv("SHIPD_RERANKER_MAX_CANDS", "24"))
RERANKER_BLEND = float(os.getenv("SHIPD_RERANKER_BLEND", "0.70"))


def log(msg: str) -> None:
	print(f"[solution] {msg}", flush=True)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
	with path.open("r", encoding="utf-8", newline="") as f:
		return list(csv.DictReader(f))


def parse_target_sequence(raw: str) -> List[str]:
	try:
		value = json.loads(raw)
	except Exception:
		return []
	if not isinstance(value, list):
		return []
	out: List[str] = []
	for x in value:
		if isinstance(x, str):
			out.append(x)
	return out


def dumps_target_sequence(seq: Sequence[str]) -> str:
	return json.dumps(list(seq), ensure_ascii=True)


def has_required_dataset_files(directory: Path) -> bool:
	return directory.is_dir() and (directory / "train.csv").exists() and (directory / "test.csv").exists()


def resolve_data_dir() -> Path:
	candidates: List[Path] = [
		Path("./dataset/public"),
		Path("./dataset"),
		Path("../dataset/public"),
		Path("../dataset"),
	]

	kaggle_input = Path("/kaggle/input")
	if kaggle_input.exists() and kaggle_input.is_dir():
		for train_file in kaggle_input.rglob("train.csv"):
			parent = train_file.parent
			if (parent / "test.csv").exists():
				candidates.append(parent)

	seen: set = set()
	valid: List[Path] = []
	for path in candidates:
		try:
			key = str(path.resolve()) if path.exists() else str(path)
		except Exception:
			key = str(path)
		if key in seen:
			continue
		seen.add(key)
		if has_required_dataset_files(path):
			valid.append(path.resolve())

	if not valid:
		raise FileNotFoundError(
			"Could not locate dataset directory containing train.csv and test.csv. "
			"Expected paths like ./dataset/public or /kaggle/input/..."
		)

	valid.sort(key=lambda p: (0 if (p / "sample_submission.csv").exists() else 1, len(str(p))))
	return valid[0]


def byte_even_view(hex_str: str) -> str:
	if len(hex_str) <= 4:
		return hex_str
	# Keep every other byte as a deterministic augmentation view.
	return "".join(hex_str[i : i + 2] for i in range(0, len(hex_str), 4))


def byte_odd_view(hex_str: str) -> str:
	if len(hex_str) <= 6:
		return hex_str
	# Complementary interleaved bytes to recover information lost in even view.
	return "".join(hex_str[i : i + 2] for i in range(2, len(hex_str), 4))


def byte_edge_view(hex_str: str, prefix_bytes: int = 2048, suffix_bytes: int = 2048) -> str:
	total_bytes = len(hex_str) // 2
	if total_bytes <= prefix_bytes + suffix_bytes:
		return hex_str
	prefix = hex_str[: prefix_bytes * 2]
	suffix = hex_str[-suffix_bytes * 2 :]
	return prefix + suffix


class SequencePrior:
	def __init__(self, sequences: Sequence[Tuple[str, ...]]):
		self.vocab: set = set()
		self.start_counts: Counter = Counter()
		self.end_counts: Counter = Counter()
		self.bigram_counts: Dict[str, Counter] = defaultdict(Counter)

		for seq in sequences:
			if not seq:
				continue
			self.vocab.update(seq)
			self.start_counts[seq[0]] += 1
			self.end_counts[seq[-1]] += 1
			for a, b in zip(seq, seq[1:]):
				self.bigram_counts[a][b] += 1

		self.vocab_size = max(1, len(self.vocab))
		self.start_total = max(1, sum(self.start_counts.values()))
		self.end_total = max(1, sum(self.end_counts.values()))

	def score(self, seq: Tuple[str, ...]) -> float:
		if not seq:
			return -8.0

		alpha = 0.5
		s = 0.0

		s += math.log((self.start_counts[seq[0]] + alpha) / (self.start_total + alpha * self.vocab_size))
		for a, b in zip(seq, seq[1:]):
			row = self.bigram_counts.get(a)
			row_total = sum(row.values()) if row else 0
			count = row[b] if row else 0
			s += math.log((count + alpha) / (row_total + alpha * self.vocab_size))
		s += math.log((self.end_counts[seq[-1]] + alpha) / (self.end_total + alpha * self.vocab_size))

		return s / max(1, len(seq))


class LengthPrior:
	def __init__(self, train_hex_lens: Sequence[int], train_seq_lens: Sequence[int]):
		bucket_values: Dict[int, List[int]] = defaultdict(list)
		for h, s in zip(train_hex_lens, train_seq_lens):
			bucket_values[self.bucket(h)].append(s)

		self.bucket_means: Dict[int, float] = {}
		global_mean = sum(train_seq_lens) / max(1, len(train_seq_lens))
		for b in range(0, 20):
			values = bucket_values.get(b)
			if values:
				self.bucket_means[b] = sum(values) / len(values)
			else:
				self.bucket_means[b] = global_mean

	@staticmethod
	def bucket(hex_len: int) -> int:
		return min(19, int(math.log2(max(2, hex_len))))

	def expected_length(self, hex_len: int) -> float:
		return self.bucket_means[self.bucket(hex_len)]


@lru_cache(maxsize=400_000)
def sequence_lev_distance(a: Tuple[str, ...], b: Tuple[str, ...]) -> int:
	n, m = len(a), len(b)
	if n == 0:
		return m
	if m == 0:
		return n

	dp = list(range(m + 1))
	for i in range(1, n + 1):
		prev = dp[0]
		dp[0] = i
		ai = a[i - 1]
		for j in range(1, m + 1):
			tmp = dp[j]
			cost = 0 if ai == b[j - 1] else 1
			dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
			prev = tmp
	return dp[m]


def sequence_ned(a: Tuple[str, ...], b: Tuple[str, ...]) -> float:
	if not a and not b:
		return 1.0
	d = sequence_lev_distance(a, b)
	return 1.0 - d / max(len(a), len(b))


def score_candidate_heuristic(rec: Dict[str, object]) -> float:
	return (
		1.00 * float(rec["exp_ned"])
		+ 0.06 * float(rec["support"])
		+ 0.03 * float(rec["base_score"])
		+ 0.02 * float(rec["len_penalty"])
		+ 0.003 * float(rec["prior"])
	)


def candidate_feature_vector(rec: Dict[str, object]) -> List[float]:
	return [
		float(rec["exp_ned"]),
		float(rec["support"]),
		float(rec["base_score"]),
		float(rec["len_penalty"]),
		float(rec["prior"]),
		float(rec["vote_count"]),
		float(rec["cand_len"]),
		float(rec["expected_len"]),
		float(rec["query_log_len"]),
	]


def build_candidate_records(
	query_hex_len: int,
	neighbors: Sequence[Tuple[int, float]],
	train_sequences: Sequence[Tuple[str, ...]],
	prior_scores: Dict[Tuple[str, ...], float],
	default_length: float,
	frequent_sequences: Sequence[Tuple[str, ...]],
	max_metric_neighbors: int,
	max_candidates: int,
) -> List[Dict[str, object]]:
	candidate_scores: Dict[Tuple[str, ...], float] = defaultdict(float)
	neighbor_counts: Counter = Counter()
	weight_sum = 0.0
	length_sum = 0.0
	nn_items: List[Tuple[Tuple[str, ...], float]] = []

	for rank, (idx, sim) in enumerate(neighbors):
		if idx < 0 or idx >= len(train_sequences):
			continue
		seq = train_sequences[idx]
		if not seq:
			continue
		sim = max(0.0, float(sim))
		rank_decay = 1.0 / (1.0 + 0.17 * rank)
		w = (sim + 1e-6) * rank_decay
		candidate_scores[seq] += w
		neighbor_counts[seq] += 1
		weight_sum += w
		length_sum += w * len(seq)
		nn_items.append((seq, w))

	if not candidate_scores:
		return []

	for seq in frequent_sequences[:10]:
		candidate_scores[seq] += 0.0

	if max_candidates > 0 and len(candidate_scores) > max_candidates:
		top_items = heapq.nlargest(max_candidates, candidate_scores.items(), key=lambda x: x[1])
		candidate_scores = defaultdict(float, top_items)

	expected_len = length_sum / weight_sum if weight_sum > 0 else default_length
	if not math.isfinite(expected_len) or expected_len <= 0:
		expected_len = default_length

	if weight_sum > 0:
		nn_items = [(seq, w / weight_sum) for seq, w in nn_items]
	else:
		uniform = 1.0 / max(1, len(nn_items))
		nn_items = [(seq, uniform) for seq, _ in nn_items]

	if max_metric_neighbors > 0:
		nn_items = nn_items[:max_metric_neighbors]

	if nn_items:
		max_len = int(round(expected_len))
		max_len = max(1, min(64, max_len))
		consensus: List[str] = []
		for pos in range(max_len):
			tok_w: Dict[str, float] = defaultdict(float)
			for seq, w in nn_items:
				if pos < len(seq):
					tok_w[seq[pos]] += w
			if not tok_w:
				break
			best_tok = max(tok_w.items(), key=lambda x: x[1])[0]
			consensus.append(best_tok)
		if consensus:
			candidate_scores[tuple(consensus)] += 0.0

	records: List[Dict[str, object]] = []
	query_log_len = math.log1p(max(1.0, query_hex_len / 2.0))
	for seq, base_score in candidate_scores.items():
		lp = prior_scores.get(seq, -8.0)
		len_penalty = -abs(len(seq) - expected_len) / max(1.0, expected_len)
		exp_ned = 0.0
		support = 0.0
		for nseq, w in nn_items:
			exp_ned += w * sequence_ned(seq, nseq)
			if nseq == seq:
				support += w

		rec: Dict[str, object] = {
			"seq": seq,
			"base_score": float(base_score),
			"support": float(support),
			"len_penalty": float(len_penalty),
			"prior": float(lp),
			"exp_ned": float(exp_ned),
			"vote_count": float(neighbor_counts.get(seq, 0)),
			"cand_len": float(len(seq)),
			"expected_len": float(expected_len),
			"query_log_len": float(query_log_len),
		}
		rec["heuristic"] = score_candidate_heuristic(rec)
		records.append(rec)

	return records


def train_candidate_reranker(
	retriever: object,
	train_hex: Sequence[str],
	train_sequences: Sequence[Tuple[str, ...]],
	prior_scores: Dict[Tuple[str, ...], float],
	len_prior: LengthPrior,
	frequent_sequences: Sequence[Tuple[str, ...]],
	top_k: int,
	batch_size: int,
	metric_neighbors: int,
	max_candidates: int,
):
	if not RERANKER_ENABLED:
		return None

	try:
		import numpy as np
		from sklearn.ensemble import HistGradientBoostingRegressor
	except Exception as exc:
		log(f"Skipping learned reranker (sklearn model unavailable: {exc}).")
		return None

	n_train = len(train_hex)
	if n_train == 0:
		return None

	query_indices = list(range(n_train))
	if RERANKER_MAX_QUERIES > 0 and n_train > RERANKER_MAX_QUERIES:
		rng = random.Random(SEED)
		query_indices = sorted(rng.sample(query_indices, RERANKER_MAX_QUERIES))

	query_hex = [train_hex[i] for i in query_indices]
	train_top_k = min(n_train, top_k + 1)
	log(f"Training learned reranker on {len(query_indices)} pseudo-queries (top_k={train_top_k})...")

	if isinstance(retriever, SklearnRetriever):
		neighbor_lists = retriever.query(query_hex, top_k=train_top_k, batch_size=batch_size)
	else:
		neighbor_lists = retriever.query(query_hex, top_k=train_top_k)

	X_rows: List[List[float]] = []
	y_rows: List[float] = []

	for q_idx, nn in zip(query_indices, neighbor_lists):
		filtered = [(idx, sim) for idx, sim in nn if idx != q_idx]
		if not filtered:
			continue

		default_len = len_prior.expected_length(len(train_hex[q_idx]))
		records = build_candidate_records(
			query_hex_len=len(train_hex[q_idx]),
			neighbors=filtered,
			train_sequences=train_sequences,
			prior_scores=prior_scores,
			default_length=default_len,
			frequent_sequences=frequent_sequences,
			max_metric_neighbors=metric_neighbors,
			max_candidates=max_candidates,
		)
		if not records:
			continue

		records.sort(key=lambda r: float(r["heuristic"]), reverse=True)
		if RERANKER_MAX_CANDS_PER_QUERY > 0:
			records = records[:RERANKER_MAX_CANDS_PER_QUERY]

		true_seq = train_sequences[q_idx]
		for rec in records:
			seq = rec["seq"]
			if not isinstance(seq, tuple):
				continue
			X_rows.append(candidate_feature_vector(rec))
			y_rows.append(sequence_ned(seq, true_seq))

	if len(X_rows) < 3000:
		log(f"Skipping learned reranker (insufficient samples: {len(X_rows)}).")
		return None

	X = np.asarray(X_rows, dtype=np.float32)
	y = np.asarray(y_rows, dtype=np.float32)

	model = HistGradientBoostingRegressor(
		loss="squared_error",
		learning_rate=0.05,
		max_iter=260,
		max_depth=6,
		min_samples_leaf=24,
		l2_regularization=0.08,
		random_state=SEED,
	)
	model.fit(X, y)

	pred_preview = model.predict(X[: min(len(X), 20000)])
	mae = float(np.mean(np.abs(pred_preview - y[: len(pred_preview)])))
	log(f"Learned reranker ready (samples={len(X_rows)}, feature_dim={X.shape[1]}, preview_mae={mae:.5f}).")
	return model


def choose_sequence(
	query_hex_len: int,
	neighbors: Sequence[Tuple[int, float]],
	train_sequences: Sequence[Tuple[str, ...]],
	prior_scores: Dict[Tuple[str, ...], float],
	default_length: float,
	frequent_sequences: Sequence[Tuple[str, ...]],
	max_metric_neighbors: int,
	max_candidates: int,
	reranker_model=None,
) -> Tuple[str, ...]:
	fallback_seq = frequent_sequences[0] if frequent_sequences else ("Linear", "ReLU", "Linear")
	if not neighbors:
		return fallback_seq

	records = build_candidate_records(
		query_hex_len=query_hex_len,
		neighbors=neighbors,
		train_sequences=train_sequences,
		prior_scores=prior_scores,
		default_length=default_length,
		frequent_sequences=frequent_sequences,
		max_metric_neighbors=max_metric_neighbors,
		max_candidates=max_candidates,
	)
	if not records:
		return fallback_seq

	if reranker_model is not None:
		try:
			import numpy as np

			X = np.asarray([candidate_feature_vector(r) for r in records], dtype=np.float32)
			preds = reranker_model.predict(X)
			best_idx = 0
			best_score = -1e18
			for i, (rec, pred) in enumerate(zip(records, preds)):
				blended = RERANKER_BLEND * float(pred) + (1.0 - RERANKER_BLEND) * float(rec["heuristic"])
				if blended > best_score:
					best_score = blended
					best_idx = i
			best_seq = records[best_idx].get("seq")
			if isinstance(best_seq, tuple) and best_seq:
				return best_seq
		except Exception:
			pass

	best = max(records, key=lambda r: float(r["heuristic"]))
	seq = best.get("seq")
	if isinstance(seq, tuple) and seq:
		return seq
	return fallback_seq


class FallbackRetriever:
	"""Pure-Python sparse TF-IDF retriever for environments without numpy/sklearn."""

	def __init__(self, train_hex: Sequence[str]):
		self.train_hex = list(train_hex)
		self.idf: Dict[int, float] = {}
		self.postings: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

		self._build_index()

	@staticmethod
	def _hex_to_bytes(hex_str: str) -> List[int]:
		n = len(hex_str)
		out: List[int] = []
		# Convert as many valid byte pairs as possible.
		end = n - (n % 2)
		for i in range(0, end, 2):
			try:
				out.append(int(hex_str[i : i + 2], 16))
			except Exception:
				continue
		return out

	@staticmethod
	def _extract_features(hex_str: str) -> Dict[int, float]:
		"""
		Deterministic byte-level features:
		- byte unigrams
		- byte bigrams
		- hashed trigrams (sampled)
		- edge bigrams (prefix/suffix emphasis)
		- even-index bigrams as augmentation view
		"""
		b = FallbackRetriever._hex_to_bytes(hex_str)
		if not b:
			return {}

		feats: Dict[int, float] = defaultdict(float)

		# Offsets keep feature namespaces separate.
		O_UNI = 0
		O_BIGRAM = 1_000
		O_TRI = 80_000
		O_EDGE = 150_000
		O_EVEN = 230_000
		TRI_BUCKET = 65_536

		# Unigrams.
		for x in b:
			feats[O_UNI + x] += 1.0

		# Full bigrams.
		for i in range(len(b) - 1):
			idx = O_BIGRAM + b[i] * 256 + b[i + 1]
			feats[idx] += 1.0

		# Sampled trigrams for extra discrimination with bounded cost.
		for i in range(0, len(b) - 2, 2):
			a, c, d = b[i], b[i + 1], b[i + 2]
			h = ((a * 1315423911) ^ (c * 2654435761) ^ (d * 2246822519)) & (TRI_BUCKET - 1)
			feats[O_TRI + h] += 1.0

		# Edge bigrams (strong architecture cues often appear in serialized headers/tails).
		edge = 512
		left = b[:edge]
		right = b[-edge:] if len(b) > edge else []
		for arr in (left, right):
			for i in range(len(arr) - 1):
				idx = O_EDGE + arr[i] * 256 + arr[i + 1]
				feats[idx] += 1.4

		# Even-index bigrams as deterministic augmentation view.
		even = b[::2]
		for i in range(len(even) - 1):
			idx = O_EVEN + even[i] * 256 + even[i + 1]
			feats[idx] += 0.8

		return feats

	def _build_index(self) -> None:
		n_docs = len(self.train_hex)
		doc_features: List[Dict[int, float]] = []
		df: Counter = Counter()

		for hex_str in self.train_hex:
			feats = self._extract_features(hex_str)
			doc_features.append(feats)
			for k in feats.keys():
				df[k] += 1

		# IDF with smoothing.
		self.idf = {k: math.log((n_docs + 1.0) / (v + 1.0)) + 1.0 for k, v in df.items()}

		# Build normalized postings list.
		for doc_id, feats in enumerate(doc_features):
			if not feats:
				continue
			weighted: Dict[int, float] = {}
			norm_sq = 0.0
			for k, tf in feats.items():
				w = (1.0 + math.log(tf)) * self.idf.get(k, 1.0)
				weighted[k] = w
				norm_sq += w * w
			norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
			inv_norm = 1.0 / norm
			for k, w in weighted.items():
				self.postings[k].append((doc_id, w * inv_norm))

	def _query_features(self, hex_str: str) -> Dict[int, float]:
		feats = self._extract_features(hex_str)
		if not feats:
			return {}
		weighted: Dict[int, float] = {}
		norm_sq = 0.0
		for k, tf in feats.items():
			idf = self.idf.get(k)
			if idf is None:
				continue
			w = (1.0 + math.log(tf)) * idf
			weighted[k] = w
			norm_sq += w * w
		norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
		inv_norm = 1.0 / norm
		for k in list(weighted.keys()):
			weighted[k] *= inv_norm
		return weighted

	def query(self, test_hex: Sequence[str], top_k: int = 24) -> List[List[Tuple[int, float]]]:
		out: List[List[Tuple[int, float]]] = []
		for q in test_hex:
			qv = self._query_features(q)
			if not qv:
				out.append([])
				continue

			scores: Dict[int, float] = defaultdict(float)
			for k, wq in qv.items():
				plist = self.postings.get(k)
				if not plist:
					continue
				for doc_id, wd in plist:
					scores[doc_id] += wq * wd

			if not scores:
				out.append([])
				continue

			top = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
			top = [(doc_id, float(score)) for doc_id, score in top]
			out.append(top)
		return out


class SklearnRetriever:
	def __init__(self):
		import numpy as np
		from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

		self.np = np
		self.HashingVectorizer = HashingVectorizer
		self.TfidfTransformer = TfidfTransformer

		self.vec_full = HashingVectorizer(
			analyzer="char",
			ngram_range=(3, 6),
			n_features=1 << 20,
			lowercase=False,
			alternate_sign=False,
			binary=False,
			norm=None,
			dtype=np.float32,
		)
		self.vec_even = HashingVectorizer(
			analyzer="char",
			ngram_range=(3, 5),
			n_features=1 << 19,
			lowercase=False,
			alternate_sign=False,
			binary=False,
			norm=None,
			dtype=np.float32,
		)
		self.vec_odd = HashingVectorizer(
			analyzer="char",
			ngram_range=(3, 5),
			n_features=1 << 19,
			lowercase=False,
			alternate_sign=False,
			binary=False,
			norm=None,
			dtype=np.float32,
		)
		self.vec_edge = HashingVectorizer(
			analyzer="char",
			ngram_range=(3, 6),
			n_features=1 << 19,
			lowercase=False,
			alternate_sign=False,
			binary=False,
			norm=None,
			dtype=np.float32,
		)

		self.tfidf_full = TfidfTransformer(norm="l2", sublinear_tf=True)
		self.tfidf_even = TfidfTransformer(norm="l2", sublinear_tf=True)
		self.tfidf_odd = TfidfTransformer(norm="l2", sublinear_tf=True)
		self.tfidf_edge = TfidfTransformer(norm="l2", sublinear_tf=True)

		self.X_train_full = None
		self.X_train_even = None
		self.X_train_odd = None
		self.X_train_edge = None

	def fit(self, train_hex: Sequence[str]) -> None:
		train_full = list(train_hex)
		train_even = [byte_even_view(x) for x in train_hex]
		train_odd = [byte_odd_view(x) for x in train_hex]
		train_edge = [byte_edge_view(x) for x in train_hex]

		Xf = self.vec_full.transform(train_full)
		Xe = self.vec_even.transform(train_even)
		Xo = self.vec_odd.transform(train_odd)
		Xg = self.vec_edge.transform(train_edge)

		self.X_train_full = self.tfidf_full.fit_transform(Xf)
		self.X_train_even = self.tfidf_even.fit_transform(Xe)
		self.X_train_odd = self.tfidf_odd.fit_transform(Xo)
		self.X_train_edge = self.tfidf_edge.fit_transform(Xg)

	def query(self, test_hex: Sequence[str], top_k: int = 24, batch_size: int = 64) -> List[List[Tuple[int, float]]]:
		from sklearn.metrics.pairwise import linear_kernel

		np = self.np
		test_full = list(test_hex)
		test_even = [byte_even_view(x) for x in test_hex]
		test_odd = [byte_odd_view(x) for x in test_hex]
		test_edge = [byte_edge_view(x) for x in test_hex]

		Xt_full = self.tfidf_full.transform(self.vec_full.transform(test_full))
		Xt_even = self.tfidf_even.transform(self.vec_even.transform(test_even))
		Xt_odd = self.tfidf_odd.transform(self.vec_odd.transform(test_odd))
		Xt_edge = self.tfidf_edge.transform(self.vec_edge.transform(test_edge))

		n_test = Xt_full.shape[0]
		n_train = self.X_train_full.shape[0]
		k = min(top_k, n_train)
		out: List[List[Tuple[int, float]]] = []

		for start in range(0, n_test, batch_size):
			end = min(n_test, start + batch_size)

			sim_full = linear_kernel(Xt_full[start:end], self.X_train_full)
			sim_even = linear_kernel(Xt_even[start:end], self.X_train_even)
			sim_odd = linear_kernel(Xt_odd[start:end], self.X_train_odd)
			sim_edge = linear_kernel(Xt_edge[start:end], self.X_train_edge)

			for local_i, (row_full, row_even, row_odd, row_edge) in enumerate(
				zip(sim_full, sim_even, sim_odd, sim_edge)
			):
				q_hex_len = len(test_full[start + local_i])
				if q_hex_len < 45_000:
					w_full, w_even, w_odd, w_edge = 0.48, 0.24, 0.18, 0.10
				elif q_hex_len < 120_000:
					w_full, w_even, w_odd, w_edge = 0.54, 0.22, 0.16, 0.08
				else:
					w_full, w_even, w_odd, w_edge = 0.60, 0.20, 0.14, 0.06

				row = w_full * row_full + w_even * row_even + w_odd * row_odd + w_edge * row_edge
				if k <= 0:
					out.append([])
					continue
				idx = np.argpartition(row, -k)[-k:]
				idx = idx[np.argsort(row[idx])[::-1]]
				out.append([(int(i), float(row[i])) for i in idx])

		return out


def build_predictions(train_rows: Sequence[Dict[str, str]], test_rows: Sequence[Dict[str, str]]) -> List[Tuple[str, ...]]:
	train_hex = [r["onnx_hex"] for r in train_rows]
	test_hex = [r["onnx_hex"] for r in test_rows]
	train_targets: List[Tuple[str, ...]] = [tuple(parse_target_sequence(r.get("target_sequence", "[]"))) for r in train_rows]

	train_hex_lens = [len(x) for x in train_hex]
	train_seq_lens = [len(x) for x in train_targets]

	seq_counter = Counter(train_targets)
	frequent_sequences = [seq for seq, _ in seq_counter.most_common(120) if seq]
	if not frequent_sequences:
		frequent_sequences = [("Linear", "ReLU", "Linear")]

	prior = SequencePrior(train_targets)
	unique_sequences = list(seq_counter.keys())
	prior_scores = {seq: prior.score(seq) for seq in unique_sequences}
	len_prior = LengthPrior(train_hex_lens, train_seq_lens)

	use_sklearn = False
	retriever = None
	try:
		# noqa: F401 - import check only
		import numpy  # type: ignore
		import sklearn  # type: ignore

		use_sklearn = True
	except Exception as exc:
		log(f"sklearn stack unavailable, using pure-python fallback retriever ({exc}).")
		if REQUIRE_SKLEARN:
			raise RuntimeError(
				"Required fast stack unavailable (numpy/sklearn). "
				"Install dependencies or set SHIPD_REQUIRE_SKLEARN=0 to allow fallback."
			)

	top_k = ACTIVE_PROFILE["top_k"]
	batch_size = ACTIVE_PROFILE["batch_size"]
	metric_neighbors = ACTIVE_PROFILE["metric_neighbors"]
	max_candidates = ACTIVE_PROFILE["max_candidates"]

	log(
		f"Profile={ACTIVE_PROFILE_NAME} top_k={top_k} batch_size={batch_size} "
		f"metric_neighbors={metric_neighbors} max_candidates={max_candidates} "
		f"threads={CPU_COUNT}"
	)

	if use_sklearn:
		log("Building sklearn multi-view retrieval index...")
		retriever = SklearnRetriever()
		retriever.fit(train_hex)
		neighbors = retriever.query(test_hex, top_k=top_k, batch_size=batch_size)
	else:
		log("Building fallback retrieval index...")
		retriever = FallbackRetriever(train_hex)
		neighbors = retriever.query(test_hex, top_k=top_k)

	reranker_model = None
	if use_sklearn and RERANKER_ENABLED:
		t_rr = time.time()
		reranker_model = train_candidate_reranker(
			retriever=retriever,
			train_hex=train_hex,
			train_sequences=train_targets,
			prior_scores=prior_scores,
			len_prior=len_prior,
			frequent_sequences=frequent_sequences,
			top_k=top_k,
			batch_size=batch_size,
			metric_neighbors=metric_neighbors,
			max_candidates=max_candidates,
		)
		if reranker_model is None:
			log("Using heuristic reranking (learned reranker unavailable).")
		else:
			log(f"Learned reranker training took {time.time() - t_rr:.2f}s")
	elif RERANKER_ENABLED and not use_sklearn:
		log("Skipping learned reranker because fast sklearn stack is unavailable.")

	predictions: List[Tuple[str, ...]] = []
	for q_hex, nn in zip(test_hex, neighbors):
		default_len = len_prior.expected_length(len(q_hex))
		pred = choose_sequence(
			query_hex_len=len(q_hex),
			neighbors=nn,
			train_sequences=train_targets,
			prior_scores=prior_scores,
			default_length=default_len,
			frequent_sequences=frequent_sequences,
			max_metric_neighbors=metric_neighbors,
			max_candidates=max_candidates,
			reranker_model=reranker_model,
		)
		predictions.append(pred)

	return predictions


def write_submission(test_rows: Sequence[Dict[str, str]], predictions: Sequence[Tuple[str, ...]], out_path: Path) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["id", "target_sequence"])
		for row, seq in zip(test_rows, predictions):
			writer.writerow([row["id"], dumps_target_sequence(seq)])


def validate_submission(out_path: Path, expected_rows: int) -> None:
	rows = read_csv_rows(out_path)
	if len(rows) != expected_rows:
		raise ValueError(f"submission row count mismatch: expected {expected_rows}, got {len(rows)}")
	for i, row in enumerate(rows):
		if "id" not in row or "target_sequence" not in row:
			raise ValueError("submission missing required columns")
		try:
			val = json.loads(row["target_sequence"])
			if not isinstance(val, list):
				raise ValueError("target_sequence is not a list")
		except Exception as exc:
			raise ValueError(f"invalid JSON at row {i + 2}: {exc}") from exc


def main() -> None:
	t0 = time.time()

	data_dir = resolve_data_dir()
	log(f"Using data directory: {data_dir}")

	train_path = data_dir / "train.csv"
	test_path = data_dir / "test.csv"
	out_path = Path("./working/submission.csv")

	train_rows = read_csv_rows(train_path)
	test_rows = read_csv_rows(test_path)

	if not train_rows or not test_rows:
		raise RuntimeError("train/test files are empty or unreadable")

	log(f"Loaded train rows: {len(train_rows)}")
	log(f"Loaded test rows: {len(test_rows)}")

	predictions = build_predictions(train_rows, test_rows)
	write_submission(test_rows, predictions, out_path)
	validate_submission(out_path, expected_rows=len(test_rows))

	dt = time.time() - t0
	log(f"Wrote submission: {out_path.resolve()}")
	log(f"Completed in {dt:.2f}s")


if __name__ == "__main__":
	main()

