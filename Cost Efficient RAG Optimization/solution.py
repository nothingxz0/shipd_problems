#!/usr/bin/env python3
"""
Single-file baseline for Cost Efficient RAG Optimization.

Outputs:
- working/submission.csv (default)

Design goals:
- Strong lexical retrieval + learned reranking + compact evidence selection
- QA-assisted answer extraction with heuristic fallback
- Train-time parameter tuning using challenge metric
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import statistics
import subprocess
import time
import importlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
except Exception:
    np = None
    LogisticRegression = None

try:
    XGBModule = importlib.import_module("xgboost")
    XGBClassifier = XGBModule.XGBClassifier
    XGBDMatrix = XGBModule.DMatrix
except Exception:
    XGBModule = None
    XGBClassifier = None
    XGBDMatrix = None

try:
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
except Exception:
    torch = None
    AutoModelForQuestionAnswering = None
    AutoTokenizer = None

# -------------------------
# Text + parsing utilities
# -------------------------

WORD_RE = re.compile(r"[a-z0-9]+")
CHUNK_MARKER_RE = re.compile(r"^\[(C\d+)\]\s*", re.MULTILINE)
TRIPLE_RE = re.compile(r"^\s*(.+?)\s*--\s*(.+?)\s*-->\s*(.+?)\s*$")
HTML_TAG_RE = re.compile(r"<[^>]{1,200}>")

NUMBER_RE = re.compile(
    r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|thousand|%))?\b",
    flags=re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|2100)\b")
MONTH_DATE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)[\s,]+\d{1,2}(?:[\s,]+\d{4})?\b",
    flags=re.IGNORECASE,
)

STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "to",
    "for",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "be",
    "by",
    "with",
    "at",
    "from",
    "that",
    "this",
    "it",
    "as",
    "what",
    "which",
    "who",
    "when",
    "where",
    "how",
    "many",
    "much",
    "does",
    "do",
    "did",
    "about",
    "into",
    "than",
    "largest",
    "largest",
}

ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}


@dataclass
class Chunk:
    cid: str
    text: str
    tokens: List[str]
    token_set: set
    norm_text: str


@dataclass
class RowData:
    rid: str
    query: str
    q_tokens: List[str]
    q_token_set: set
    q_type: str
    num_chunks: int
    chunks: List[Chunk]
    gold_evidence: List[str]
    gold_answer: str


@dataclass
class RankedChunk:
    chunk: Chunk
    score: float
    signals: Dict[str, float] = field(default_factory=dict)


@dataclass
class Params:
    keep_ratio: float
    no_chunk_threshold: float
    k_default: int
    k_numeric: int
    k_date: int
    k_person: int
    k_location: int
    redundancy_jaccard: float


@dataclass
class LearnedReranker:
    model: Any
    feature_names: List[str]
    blend_weight: float


class QAReader:
    def __init__(
        self,
        model_name: str,
        max_length: int = 384,
        stride: int = 128,
        max_answer_tokens: int = 24,
    ) -> None:
        if AutoTokenizer is None or AutoModelForQuestionAnswering is None or torch is None:
            raise RuntimeError("transformers/torch not available for QA reader")

        self.model_name = model_name
        self.max_length = max_length
        self.stride = stride
        self.max_answer_tokens = max_answer_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.device.type == "cuda":
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name, dtype=torch.float16)
        else:
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def best_span(self, question: str, context: str) -> Tuple[str, float]:
        if not context.strip():
            return "", -1e9

        enc = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )

        if not enc.get("input_ids"):
            return "", -1e9

        feat = self.tokenizer.pad(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
            },
            return_tensors="pt",
        )

        input_ids = feat["input_ids"].to(self.device)
        attention_mask = feat["attention_mask"].to(self.device)

        with torch.inference_mode():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)

        best_text = ""
        best_score = -1e9

        for i in range(feat["input_ids"].shape[0]):
            offsets = enc["offset_mapping"][i]
            seq_ids = enc.sequence_ids(i)
            valid = [j for j, sid in enumerate(seq_ids) if sid == 1 and offsets[j] != (0, 0)]
            if not valid:
                continue

            start_logits = out.start_logits[i]
            end_logits = out.end_logits[i]
            top_starts = torch.topk(start_logits, k=min(10, start_logits.shape[0])).indices.tolist()
            top_ends = torch.topk(end_logits, k=min(10, end_logits.shape[0])).indices.tolist()
            valid_set = set(valid)

            for st in top_starts:
                if st not in valid_set:
                    continue
                for en in top_ends:
                    if en not in valid_set:
                        continue
                    if en < st:
                        continue
                    if en - st + 1 > self.max_answer_tokens:
                        continue

                    score = float(start_logits[st] + end_logits[en])
                    if score <= best_score:
                        continue

                    start_char, _ = offsets[st]
                    _, end_char = offsets[en]
                    if end_char <= start_char:
                        continue

                    text = context[start_char:end_char].strip()
                    if not text:
                        continue

                    best_score = score
                    best_text = text

        return best_text, best_score


GLOBAL_QA_READER: Optional[QAReader] = None


def gpu_visible_via_nvidia_smi() -> bool:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip()
        return bool(out)
    except Exception:
        return False


def normalize_eval(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def answer_f1(pred: str, gold: str) -> float:
    p = normalize_eval(pred).split()
    g = normalize_eval(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    p_count = Counter(p)
    g_count = Counter(g)
    common = sum((p_count & g_count).values())
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


def evidence_efficiency(selected: Sequence[str], gold: Sequence[str], total_pool: int) -> float:
    selected_set = set(selected)
    gold_set = set(gold)
    if total_pool <= 0:
        return 0.3

    # Undefined in prompt when selection is empty; use precision=0.0 as conservative choice.
    precision = (len(selected_set & gold_set) / len(selected_set)) if selected_set else 0.0
    compression = 1.0 - (len(selected_set) / total_pool)
    return 0.3 + 0.4 * precision + 0.3 * compression


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    toks = WORD_RE.findall(text.lower())
    if not remove_stopwords:
        return toks
    return [t for t in toks if t not in STOPWORDS]


def safe_norm(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower()))


def char_ngrams(text: str, n: int = 3) -> set:
    s = safe_norm(text)
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def parse_context_chunks(context: str) -> List[Tuple[str, str]]:
    matches = list(CHUNK_MARKER_RE.finditer(context))
    if not matches:
        return []

    out: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        cid = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(context)
        text = context[start:end].strip()
        out.append((cid, text))
    return out


def parse_evidence_ids(evidence_ids: str) -> List[str]:
    evidence_ids = (evidence_ids or "").strip()
    if not evidence_ids:
        return []
    return [x.strip() for x in evidence_ids.split(",") if x.strip()]


def detect_question_type(query: str) -> str:
    q = query.lower().strip()

    if "how many" in q or "number of" in q or "population" in q or "capacity" in q or "how much" in q:
        return "numeric"
    if q.startswith("when") or "what year" in q or "birthdate" in q or "date" in q:
        return "date"
    if q.startswith("who") or "which actor" in q or "which person" in q:
        return "person"
    if q.startswith("where") or "what country" in q or "which country" in q or "what city" in q or "town" in q:
        return "location"
    return "generic"


def extract_ordinal(query: str) -> int:
    q = query.lower()
    for word, value in ORDINAL_WORDS.items():
        if re.search(r"\b" + re.escape(word) + r"\b", q):
            return value
    m = re.search(r"\b(\d+)(?:st|nd|rd|th)\b", q)
    if m:
        return int(m.group(1))
    return -1


# -------------------------
# Retrieval ranking
# -------------------------


def rank_positions(values: Sequence[float]) -> List[int]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    pos = [0] * len(values)
    for r, i in enumerate(order, start=1):
        pos[i] = r
    return pos


def minmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def cosine_from_weight_maps(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    for k, v in a.items():
        dot += v * b.get(k, 0.0)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def build_tfidf_map(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = Counter(tokens)
    return {t: c * idf.get(t, 1.0) for t, c in tf.items()}


def bm25_score(query_tokens: List[str], doc_tokens: List[str], df: Dict[str, int], n_docs: int, avgdl: float) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    freq = Counter(doc_tokens)
    dl = len(doc_tokens)
    k1 = 1.5
    b = 0.75
    score = 0.0
    for t in set(query_tokens):
        f = freq.get(t, 0)
        if f == 0:
            continue
        n_q = df.get(t, 0)
        idf = math.log((n_docs - n_q + 0.5) / (n_q + 0.5) + 1.0)
        denom = f + k1 * (1.0 - b + b * (dl / max(avgdl, 1e-6)))
        score += idf * (f * (k1 + 1.0) / denom)
    return score


def rank_chunks(row: RowData) -> List[RankedChunk]:
    chunks = row.chunks
    if not chunks:
        return []

    n = len(chunks)
    doc_tokens = [c.tokens for c in chunks]
    avgdl = statistics.mean(max(len(t), 1) for t in doc_tokens)

    df = Counter()
    for toks in doc_tokens:
        for t in set(toks):
            df[t] += 1

    idf = {t: math.log((n + 1.0) / (df_t + 1.0)) + 1.0 for t, df_t in df.items()}
    q_vec = build_tfidf_map(row.q_tokens, idf)

    bm25_vals: List[float] = []
    tfidf_vals: List[float] = []
    overlap_vals: List[float] = []
    char_vals: List[float] = []

    q_char = char_ngrams(row.query, n=3)

    for c in chunks:
        bm25_vals.append(bm25_score(row.q_tokens, c.tokens, df, n, avgdl))

        d_vec = build_tfidf_map(c.tokens, idf)
        tfidf_vals.append(cosine_from_weight_maps(q_vec, d_vec))

        overlap_vals.append(len(row.q_token_set & c.token_set) / max(1, len(row.q_token_set)))

        c_char = char_ngrams(c.text, n=3)
        char_vals.append(jaccard(q_char, c_char))

    bm25_n = minmax(bm25_vals)
    tfidf_n = minmax(tfidf_vals)
    overlap_n = minmax(overlap_vals)
    char_n = minmax(char_vals)

    ranks_bm25 = rank_positions(bm25_vals)
    ranks_tfidf = rank_positions(tfidf_vals)
    ranks_overlap = rank_positions(overlap_vals)
    ranks_char = rank_positions(char_vals)

    rrf_vals = []
    rrf_k = 60.0
    for i in range(n):
        rrf = (
            1.0 / (rrf_k + ranks_bm25[i])
            + 1.0 / (rrf_k + ranks_tfidf[i])
            + 1.0 / (rrf_k + ranks_overlap[i])
            + 1.0 / (rrf_k + ranks_char[i])
        )
        rrf_vals.append(rrf)

    rrf_n = minmax(rrf_vals)

    scored: List[RankedChunk] = []
    for i, c in enumerate(chunks):
        lexical = 0.45 * bm25_n[i] + 0.25 * tfidf_n[i] + 0.20 * overlap_n[i] + 0.10 * char_n[i]
        final_score = 0.60 * rrf_n[i] + 0.40 * lexical

        # Small boost for KG-style triples when query is entity/relation-heavy.
        if "-->" in c.text:
            final_score += 0.015

        scored.append(
            RankedChunk(
                chunk=c,
                score=final_score,
                signals={
                    "bm25_raw": bm25_vals[i],
                    "bm25": bm25_n[i],
                    "tfidf": tfidf_n[i],
                    "overlap": overlap_n[i],
                    "char": char_n[i],
                    "rrf": rrf_n[i],
                    "lexical": lexical,
                    "is_kg": 1.0 if "-->" in c.text else 0.0,
                    "is_table": 1.0 if "|" in c.text else 0.0,
                },
            )
        )

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored


def chunk_feature_names() -> List[str]:
    return [
        "base_score",
        "rrf",
        "lexical",
        "bm25",
        "tfidf",
        "overlap",
        "char",
        "rank_inv",
        "rank_ratio",
        "chunk_len_norm",
        "is_kg",
        "is_table",
        "q_numeric",
        "q_date",
        "q_person",
        "q_location",
        "q_generic",
        "has_num_in_chunk",
        "has_date_in_chunk",
        "query_chunk_token_overlap",
    ]


def chunk_feature_vector(row: RowData, rc: RankedChunk, rank_idx: int, total: int) -> List[float]:
    sig = rc.signals
    q_tokens = row.q_token_set
    c_tokens = rc.chunk.token_set

    overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
    rank_ratio = rank_idx / max(1, total - 1)
    chunk_len_norm = min(len(rc.chunk.tokens), 140) / 140.0

    return [
        float(rc.score),
        float(sig.get("rrf", 0.0)),
        float(sig.get("lexical", 0.0)),
        float(sig.get("bm25", 0.0)),
        float(sig.get("tfidf", 0.0)),
        float(sig.get("overlap", 0.0)),
        float(sig.get("char", 0.0)),
        1.0 / (rank_idx + 1.0),
        float(rank_ratio),
        float(chunk_len_norm),
        float(sig.get("is_kg", 1.0 if "-->" in rc.chunk.text else 0.0)),
        float(sig.get("is_table", 1.0 if "|" in rc.chunk.text else 0.0)),
        1.0 if row.q_type == "numeric" else 0.0,
        1.0 if row.q_type == "date" else 0.0,
        1.0 if row.q_type == "person" else 0.0,
        1.0 if row.q_type == "location" else 0.0,
        1.0 if row.q_type == "generic" else 0.0,
        1.0 if find_numbers(rc.chunk.text) else 0.0,
        1.0 if find_dates(rc.chunk.text) else 0.0,
        float(overlap),
    ]


def train_chunk_reranker(
    train_rows: Sequence[RowData],
    ranked_map: Dict[str, List[RankedChunk]],
    force_gpu: bool = False,
    seed: int = 42,
) -> Optional[LearnedReranker]:
    if np is None:
        print("[reranker] numpy unavailable, skipping learned reranker")
        return None

    feature_names = chunk_feature_names()
    X: List[List[float]] = []
    y: List[int] = []
    aug_added = 0

    for row in train_rows:
        ranked = ranked_map[row.rid]
        gold = set(row.gold_evidence)
        total = max(1, len(ranked))

        for idx, rc in enumerate(ranked):
            feat = chunk_feature_vector(row, rc, idx, total)
            label = 1 if rc.chunk.cid in gold else 0
            X.append(feat)
            y.append(label)

            # Train-time augmentation via duplication:
            # - upweight positives so the classifier keeps recall
            # - upweight hard negatives from top ranks with high lexical overlap
            if label == 1:
                X.append(feat)
                y.append(1)
                aug_added += 1
            else:
                overlap = len(row.q_token_set & rc.chunk.token_set) / max(1, len(row.q_token_set))
                hard_negative = idx < 6 and overlap >= 0.35
                if hard_negative:
                    X.append(feat)
                    y.append(0)
                    aug_added += 1

    if not X or not y or len(set(y)) < 2:
        print("[reranker] insufficient labels, skipping learned reranker")
        return None

    X_np = np.asarray(X, dtype=float)
    y_np = np.asarray(y, dtype=int)

    # GPU path: use XGBoost CUDA for compatibility with Kaggle P100 and A10G.
    if force_gpu:
        if XGBClassifier is None:
            print("[reranker] force_gpu requested but xgboost unavailable, falling back to sklearn")
            force_gpu = False

    if force_gpu:
        try:
            model = XGBClassifier(
                n_estimators=180,
                max_depth=5,
                learning_rate=0.06,
                subsample=0.9,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cuda",
                random_state=seed,
                n_jobs=4,
            )
            model.fit(X_np, y_np)
            pos_rate = float(np.mean(y_np))
            print(
                f"[reranker] trained xgboost-cuda reranker | samples={len(y_np)} aug={aug_added} pos_rate={pos_rate:.4f}"
            )
            return LearnedReranker(model={"type": "xgb", "model": model}, feature_names=feature_names, blend_weight=0.56)
        except Exception as e:
            print(f"[reranker] xgboost-cuda training failed, fallback to sklearn: {e}")

    if LogisticRegression is None:
        print("[reranker] sklearn unavailable and torch/cuda path not used")
        return None

    model = LogisticRegression(
        C=1.6,
        class_weight="balanced",
        solver="liblinear",
        max_iter=500,
        random_state=seed,
    )
    model.fit(X_np, y_np)

    pos_rate = float(np.mean(y_np))
    print(f"[reranker] trained sklearn reranker | samples={len(y_np)} aug={aug_added} pos_rate={pos_rate:.4f}")
    return LearnedReranker(model=model, feature_names=feature_names, blend_weight=0.48)


def apply_chunk_reranker(
    rows: Sequence[RowData],
    ranked_map: Dict[str, List[RankedChunk]],
    reranker: Optional[LearnedReranker],
) -> Dict[str, List[RankedChunk]]:
    if reranker is None or np is None:
        return ranked_map

    out: Dict[str, List[RankedChunk]] = {}
    w = reranker.blend_weight

    for row in rows:
        ranked = ranked_map[row.rid]
        if not ranked:
            out[row.rid] = []
            continue

        total = max(1, len(ranked))
        feats = [chunk_feature_vector(row, rc, i, total) for i, rc in enumerate(ranked)]
        X_np = np.asarray(feats, dtype=float)

        probs = predict_reranker_probs(reranker, X_np)

        rescored: List[RankedChunk] = []
        for i, rc in enumerate(ranked):
            prob = float(probs[i])
            blended = (1.0 - w) * rc.score + w * prob
            signals = dict(rc.signals)
            signals["prob"] = prob
            rescored.append(RankedChunk(chunk=rc.chunk, score=blended, signals=signals))

        rescored.sort(key=lambda x: x.score, reverse=True)
        out[row.rid] = rescored

    return out


def predict_reranker_probs(reranker: LearnedReranker, X_np: Any) -> Any:
    if isinstance(reranker.model, dict) and reranker.model.get("type") == "xgb":
        model = reranker.model["model"]

        # Use DMatrix predict to avoid repeated mismatched-device warnings seen with sklearn wrapper
        # when booster is on cuda and numpy inputs are on cpu.
        if XGBDMatrix is not None:
            try:
                booster = model.get_booster()
                booster.set_param({"device": "cuda"})
                dmat = XGBDMatrix(X_np)
                probs = booster.predict(dmat)
                return np.asarray(probs, dtype=float)
            except Exception:
                pass

        probs = model.predict_proba(X_np)[:, 1]
        return np.asarray(probs, dtype=float)

    probs = reranker.model.predict_proba(X_np)[:, 1]
    return np.asarray(probs, dtype=float)


# -------------------------
# Evidence selection
# -------------------------


def k_cap_for_question(q_type: str, p: Params) -> int:
    if q_type == "numeric":
        return p.k_numeric
    if q_type == "date":
        return p.k_date
    if q_type == "person":
        return p.k_person
    if q_type == "location":
        return p.k_location
    return p.k_default


def select_evidence(row: RowData, ranked: List[RankedChunk], p: Params) -> List[RankedChunk]:
    if not ranked:
        return []

    top = ranked[0]
    top_prob = float(top.signals.get("prob", top.score))
    if top.score < p.no_chunk_threshold and top_prob < 0.18:
        return []

    selected: List[RankedChunk] = [top]
    cap = max(1, k_cap_for_question(row.q_type, p))
    top_score = top.score

    second_prob = float(ranked[1].signals.get("prob", ranked[1].score)) if len(ranked) > 1 else 0.0
    prob_gap = top_prob - second_prob

    # High confidence + clear margin: keep evidence very compact.
    if top_prob >= 0.72 and prob_gap >= 0.20:
        cap = 1

    # Ordinal/table or relation-heavy patterns often need extra supporting chunk(s).
    if extract_ordinal(row.query) > 0 or any("-->" in x.chunk.text for x in ranked[:3]):
        cap = min(6, max(cap, 3))

    # For likely direct factoid answers, avoid context stuffing.
    if row.q_type in {"numeric", "date", "person", "location"}:
        cap = min(cap, 3)

    # For low-confidence factoid queries, allow one extra candidate to improve recall.
    low_conf_factoid = row.q_type in {"numeric", "date"} and top_prob < 0.52
    if low_conf_factoid:
        cap = max(cap, 4)

    for cand in ranked[1:]:
        if len(selected) >= cap:
            break

        cand_prob = float(cand.signals.get("prob", cand.score))

        # Keep only strong candidates by either blended score ratio or reranker probability.
        keep_by_score = cand.score >= top_score * p.keep_ratio
        keep_by_prob = cand_prob >= max(0.16, top_prob - 0.24)
        if not keep_by_score and not keep_by_prob:
            if low_conf_factoid and len(selected) < 2:
                continue
            break

        redundant = False
        for s in selected:
            if jaccard(cand.chunk.token_set, s.chunk.token_set) >= p.redundancy_jaccard:
                redundant = True
                break
        if redundant:
            continue

        selected.append(cand)

    # Keep ids in numeric chunk order for stable output.
    selected_sorted = sorted(
        selected,
        key=lambda x: int(x.chunk.cid[1:]) if x.chunk.cid[1:].isdigit() else 999,
    )
    return selected_sorted


# -------------------------
# Answer extraction
# -------------------------


def clean_candidate(text: str, q_type: str) -> str:
    text = text.strip()
    text = HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\[[^\]]+\]\([^\)]+\)", "", text)
    text = re.sub(r"\[[0-9]+\]", "", text)
    text = re.sub(r"\b(?:q|a)\s*:\s*", " ", text, flags=re.IGNORECASE)
    text = text.strip(" \t\n\r\"'`[]{}")
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    # Avoid URL-heavy fragments that usually hurt answer F1.
    if "http://" in text.lower() or "https://" in text.lower() or "www." in text.lower():
        return ""

    # Keep answers compact.
    words = text.split()
    limit = 9 if q_type == "generic" else 7
    if len(words) > limit:
        text = " ".join(words[:limit])

    return text.strip(" ,;:.\"'")


def find_numbers(text: str) -> List[str]:
    return [m.group(0).strip() for m in NUMBER_RE.finditer(text)]


def find_dates(text: str) -> List[str]:
    out: List[str] = []
    out.extend(m.group(0).strip() for m in MONTH_DATE_RE.finditer(text))
    out.extend(m.group(0).strip() for m in YEAR_RE.finditer(text))
    return out


def extract_triples(lines: Iterable[str]) -> List[Tuple[str, str, str, str]]:
    out: List[Tuple[str, str, str, str]] = []
    for line in lines:
        m = TRIPLE_RE.match(line.strip())
        if not m:
            continue
        subj, rel, obj = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        out.append((subj, rel, obj, line))
    return out


def split_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def line_overlap_score(line: str, q_tokens: Sequence[str], q_token_set: set) -> float:
    lt = tokenize(line)
    if not lt:
        return 0.0
    overlap = len(set(lt) & q_token_set) / max(1, len(q_token_set))
    bonus = 0.10 if "-->" in line else 0.0
    return overlap + bonus


def is_valid_candidate(text: str, q_type: str, q_token_set: set) -> bool:
    c = clean_candidate(text, q_type)
    if not c:
        return False

    if "<" in c or ">" in c:
        return False

    toks = WORD_RE.findall(c.lower())
    if not toks:
        return False

    non_stop = [t for t in toks if t not in STOPWORDS]
    if not non_stop:
        return False

    if q_type == "numeric":
        return bool(find_numbers(c))

    if q_type == "date":
        return bool(find_dates(c))

    if set(non_stop).issubset(q_token_set):
        return False

    banned_singletons = {
        "the",
        "in",
        "what",
        "this",
        "other",
        "file",
        "list",
        "jump",
        "net",
        "name",
        "total",
        "pool",
        "dog",
        "we",
        "it",
    }
    if len(non_stop) == 1 and non_stop[0] in banned_singletons:
        return False

    if q_type in {"person", "location"}:
        has_cap = bool(re.search(r"[A-Z]", c))
        if not has_cap and len(non_stop) < 2:
            return False

    return True


def candidate_quality_adjustment(candidate: str, q_type: str, q_token_set: set) -> float:
    toks = tokenize(candidate)
    if not toks:
        return -2.0

    uniq = set(toks)
    overlap = len(uniq & q_token_set) / max(1, len(uniq))
    adj = 0.0

    # Penalize near copies of the query body.
    if overlap >= 0.85 and len(uniq) >= 2:
        adj -= 0.65

    # Penalize noisy/verbose fragments.
    if "?" in candidate:
        adj -= 0.55
    if len(toks) > (8 if q_type == "generic" else 6):
        adj -= 0.30
    if any(x in candidate.lower() for x in ["cookie", "privacy", "terms", "copyright"]):
        adj -= 0.50

    if q_type == "numeric":
        if find_numbers(candidate):
            adj += 0.28
        if len(toks) > 4:
            adj -= 0.35

    if q_type == "date":
        if find_dates(candidate):
            adj += 0.25
        if len(toks) > 5:
            adj -= 0.30

    if q_type in {"person", "location"} and not re.search(r"[A-Z]", candidate):
        adj -= 0.35

    return adj


def normalize_qa_span_for_type(span: str, q_type: str) -> str:
    span = clean_candidate(span, q_type)
    if not span:
        return ""

    if q_type == "numeric":
        nums = find_numbers(span)
        if nums:
            return nums[0]

    if q_type == "date":
        ds = find_dates(span)
        if ds:
            return ds[0]

    return span


def qa_candidate_adjustment(
    candidate: str,
    q_type: str,
    q_token_set: set,
    source: Sequence[RankedChunk],
) -> float:
    # Start with global candidate quality then add QA-specific grounding checks.
    adj = candidate_quality_adjustment(candidate, q_type, q_token_set)

    cand_norm = normalize_eval(candidate)
    if cand_norm:
        support_hits = 0
        for rc in source[:4]:
            if cand_norm in normalize_eval(rc.chunk.text):
                support_hits += 1
        if support_hits > 0:
            adj += 0.30 + 0.08 * min(support_hits - 1, 2)
        else:
            adj -= 0.45

    cand_low = candidate.lower()
    if any(tok in cand_low for tok in ["http://", "https://", "www.", "/wiki/"]):
        adj -= 1.20
    if any(ch in candidate for ch in ["|", "{", "}", "<", ">"]):
        adj -= 0.60
    if candidate.count(":") >= 2:
        adj -= 0.35
    if len(candidate.split()) > 9:
        adj -= 0.35

    if q_type in {"person", "location"}:
        if re.fullmatch(r"[A-Z][A-Za-z0-9'\-]+(?:\s+[A-Z][A-Za-z0-9'\-]+){0,4}", candidate):
            adj += 0.20

    return adj


def choose_best(candidates: List[Tuple[float, str]], q_type: str, q_token_set: set) -> str:
    if not candidates:
        return ""

    dedup: Dict[str, Tuple[float, str]] = {}
    for base_score, text in candidates:
        if not is_valid_candidate(text, q_type, q_token_set):
            continue
        c = clean_candidate(text, q_type)
        if not c:
            continue

        score = base_score + candidate_quality_adjustment(c, q_type, q_token_set)
        key = normalize_eval(c)

        prev = dedup.get(key)
        if prev is None or score > prev[0]:
            dedup[key] = (score, c)

    if not dedup:
        return ""

    ranked = sorted(dedup.values(), key=lambda x: x[0], reverse=True)
    for _, c in ranked:
        if c:
            return c
    return ""


def parse_table_blocks(text: str) -> List[List[List[str]]]:
    lines = split_lines(text)
    blocks: List[List[List[str]]] = []
    current: List[List[str]] = []

    def flush() -> None:
        nonlocal current
        if len(current) >= 2:
            blocks.append(current)
        current = []

    for ln in lines:
        if "|" not in ln:
            flush()
            continue
        cells = [c.strip() for c in ln.split("|")]
        cells = [c for c in cells if c != ""]
        if not cells:
            flush()
            continue
        # Skip markdown separator lines.
        if all(set(c) <= set("-: ") for c in cells):
            continue
        current.append(cells)
    flush()
    return blocks


def answer_from_context(
    row: RowData,
    selected: List[RankedChunk],
    ranked: List[RankedChunk],
) -> str:
    # Use selected evidence plus top-ranked context for answer recall.
    # Evidence IDs remain compact; answer extraction can still benefit from nearby high-ranked chunks.
    source: List[RankedChunk] = []
    q_type = row.q_type
    max_source_chunks = 10 if q_type == "generic" else 8
    seen = set()
    for rc in list(selected) + list(ranked):
        key = rc.chunk.cid
        if key in seen:
            continue
        seen.add(key)
        source.append(rc)
        if len(source) >= max_source_chunks:
            break

    if not source:
        return "unknown"

    all_lines: List[str] = []
    for rc in source:
        all_lines.extend(split_lines(rc.chunk.text))

    q_tokens = row.q_tokens
    q_token_set = row.q_token_set

    candidates: List[Tuple[float, str]] = []

    # 0) Optional QA-reader candidates over merged and per-chunk contexts.
    if GLOBAL_QA_READER is not None:
        qa_sources = source[:4]
        merged = "\n\n".join(rc.chunk.text for rc in qa_sources)
        span, qa_score = GLOBAL_QA_READER.best_span(row.query, merged)
        span = normalize_qa_span_for_type(span, q_type)
        if span and is_valid_candidate(span, q_type, q_token_set):
            score_norm = max(-20.0, min(20.0, qa_score)) / 10.0
            qa_adj = qa_candidate_adjustment(span, q_type, q_token_set, qa_sources)
            if qa_score >= 6.0 or qa_adj >= 0.15:
                candidates.append((2.7 + score_norm + qa_adj, span))

        # Fallback to top chunk QA only when merged span confidence looks weak.
        if qa_score < 7.0 and qa_sources:
            span2, qa_score2 = GLOBAL_QA_READER.best_span(row.query, qa_sources[0].chunk.text)
            span2 = normalize_qa_span_for_type(span2, q_type)
            if span2 and is_valid_candidate(span2, q_type, q_token_set):
                score_norm2 = max(-20.0, min(20.0, qa_score2)) / 11.0
                qa_adj2 = qa_candidate_adjustment(span2, q_type, q_token_set, qa_sources[:1])
                if qa_score2 >= 5.5 or qa_adj2 >= 0.25:
                    candidates.append((2.45 + score_norm2 + qa_adj2, span2))

    # 1) Triple candidates: very useful for this dataset.
    triples = extract_triples(all_lines)
    for subj, rel, obj, line in triples:
        rel_tokens = set(tokenize(rel))
        subj_tokens = set(tokenize(subj))
        rel_hit = len(rel_tokens & q_token_set)
        subj_hit = len(subj_tokens & q_token_set)

        if rel_hit == 0 and subj_hit == 0 and q_type in {"person", "location", "generic"}:
            continue

        base = line_overlap_score(line, q_tokens, q_token_set)
        score = base + 2.0 * rel_hit + 0.6 * subj_hit

        if q_type == "numeric":
            nums = find_numbers(obj)
            if nums:
                candidates.append((score + 1.2, nums[0]))
            continue

        if q_type == "date":
            ds = find_dates(obj)
            if ds:
                candidates.append((score + 1.2, ds[0]))
            continue

        # Generic/person/location defaults to object side.
        candidates.append((score, obj))

    # 2) Table + ordinal pattern for multi-hop style questions.
    ordinal = extract_ordinal(row.query)
    if ordinal > 0:
        relation_keywords = [
            "population",
            "capital",
            "currency",
            "language",
            "birth",
            "date",
            "country",
            "city",
            "town",
        ]

        table_entity = ""
        for rc in source:
            blocks = parse_table_blocks(rc.chunk.text)
            for block in blocks:
                header = block[0]
                rows = block[1:]
                rank_col = 0
                if header and any("rank" in h.lower() for h in header):
                    rank_col = next((i for i, h in enumerate(header) if "rank" in h.lower()), 0)

                for row_cells in rows:
                    if rank_col >= len(row_cells):
                        continue
                    rank_cell = row_cells[rank_col]
                    rank_num_match = re.match(r"\D*(\d+)", rank_cell)
                    if not rank_num_match:
                        continue
                    if int(rank_num_match.group(1)) != ordinal:
                        continue

                    # Try to read country/city-like column first.
                    target_col = -1
                    for i, h in enumerate(header):
                        hl = h.lower()
                        if "country" in hl or "city" in hl or "town" in hl or "nation" in hl:
                            target_col = i
                            break
                    if target_col == -1:
                        # Fallback: pick a non-rank textual cell.
                        for i, cell in enumerate(row_cells):
                            if i == rank_col:
                                continue
                            if not find_numbers(cell):
                                target_col = i
                                break

                    if 0 <= target_col < len(row_cells):
                        table_entity = row_cells[target_col]
                        candidates.append((2.2 + line_overlap_score(" ".join(row_cells), q_tokens, q_token_set), table_entity))

        # Two-hop lookup in top ranked chunks for specific relation if needed.
        if table_entity:
            entity_norm = normalize_eval(table_entity)
            for rc in ranked[:8]:
                for subj, rel, obj, line in extract_triples(split_lines(rc.chunk.text)):
                    subj_norm = normalize_eval(subj)
                    if not entity_norm or entity_norm not in subj_norm:
                        continue

                    rel_l = rel.lower()
                    rel_hit = any(k in rel_l for k in relation_keywords if k in row.query.lower())
                    base = 1.8 + (1.2 if rel_hit else 0.0) + line_overlap_score(line, q_tokens, q_token_set)

                    if q_type == "numeric":
                        nums = find_numbers(obj)
                        if nums:
                            candidates.append((base + 1.1, nums[0]))
                    elif q_type == "date":
                        ds = find_dates(obj)
                        if ds:
                            candidates.append((base + 1.1, ds[0]))
                    else:
                        candidates.append((base, obj))

    # 3) High-overlap line-based fallback.
    scored_lines = sorted(
        ((line_overlap_score(ln, q_tokens, q_token_set), ln) for ln in all_lines),
        key=lambda x: x[0],
        reverse=True,
    )

    for base, ln in scored_lines[:18]:
        if base < 0.06:
            continue

        if q_type == "numeric":
            m_num_focus = re.search(
                r"(?:total|number|count|population|capacity|rank|amount|sum)[^\d]{0,25}(\d{1,3}(?:,\d{3})*(?:\.\d+)?)",
                ln,
                flags=re.IGNORECASE,
            )
            if m_num_focus:
                candidates.append((base + 1.1, m_num_focus.group(1)))
            nums = find_numbers(ln)
            if nums:
                candidates.append((base + 0.9, nums[0]))
                continue

        if q_type == "date":
            ds = find_dates(ln)
            if ds:
                candidates.append((base + 0.9, ds[0]))
                continue

        if "-->" in ln:
            m = TRIPLE_RE.match(ln)
            if m:
                candidates.append((base + 0.5, m.group(3)))
                continue

        # Definition-style fallback: answer often follows a copula.
        m_tail = re.search(r"\b(?:is|are|was|were)\b\s+(.+)$", ln, flags=re.IGNORECASE)
        if m_tail:
            tail = m_tail.group(1)
            tail = re.split(r"[.;]\s", tail)[0]
            tail = tail.strip()
            if tail:
                candidates.append((base + 0.35, tail))

        # Reverse-definition pattern: "X is ..." where X is often the desired answer.
        m_head = re.match(r"([A-Z][A-Za-z0-9'\-]*(?:\s+[A-Z][A-Za-z0-9'\-]*){0,5})\s+(?:is|was|are|were)\b", ln)
        if m_head:
            head = m_head.group(1).strip()
            head_toks = [t for t in tokenize(head) if t not in STOPWORDS]
            if head_toks and not set(head_toks).issubset(q_token_set):
                candidates.append((base + 0.42, head))

        # Capitalized phrase fallback for person/location/generic.
        cap_phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}\b", ln)
        if cap_phrases:
            for ph in cap_phrases[:2]:
                ph_toks = [t for t in WORD_RE.findall(ph.lower()) if t not in STOPWORDS]
                if ph_toks and set(ph_toks).issubset(q_token_set):
                    continue
                candidates.append((base + 0.3, ph))
            continue

        # Quoted snippet fallback for descriptive questions.
        m2 = re.search(r"['\"]([^'\"]{3,60})['\"]", ln)
        if m2:
            candidates.append((base + 0.2, m2.group(1)))

    ans = choose_best(candidates, q_type, q_token_set)
    if ans:
        return ans

    return "unknown"


def maybe_init_qa_reader(model_name: str) -> Optional[QAReader]:
    if not model_name or model_name.lower() in {"none", "off", "disabled"}:
        return None
    if AutoTokenizer is None or AutoModelForQuestionAnswering is None or torch is None:
        print("[qa] transformers/torch unavailable, skipping QA reader")
        return None

    try:
        print(f"[qa] loading model: {model_name}")
        reader = QAReader(model_name=model_name)
        print(f"[qa] model loaded on {reader.device.type}")
        return reader
    except Exception as e:
        print(f"[qa] failed to load model, fallback to heuristic answerer: {e}")
        return None


# -------------------------
# Data loading + prediction
# -------------------------


def parse_row_dict(r: Dict[str, str], has_labels: bool) -> RowData:
    rid = r["id"]
    query = r["query"]
    q_tokens = tokenize(query)
    q_token_set = set(q_tokens)
    q_type = detect_question_type(query)

    parsed = parse_context_chunks(r["context"])
    chunks: List[Chunk] = []
    for cid, txt in parsed:
        toks = tokenize(txt)
        chunks.append(
            Chunk(
                cid=cid,
                text=txt,
                tokens=toks,
                token_set=set(toks),
                norm_text=safe_norm(txt),
            )
        )

    num_chunks = int(r.get("num_chunks", len(chunks)) or len(chunks))
    gold_evidence = parse_evidence_ids(r.get("evidence_ids", "")) if has_labels else []
    gold_answer = r.get("answer", "") if has_labels else ""

    return RowData(
        rid=rid,
        query=query,
        q_tokens=q_tokens,
        q_token_set=q_token_set,
        q_type=q_type,
        num_chunks=num_chunks,
        chunks=chunks,
        gold_evidence=gold_evidence,
        gold_answer=gold_answer,
    )


def iter_rows(csv_path: Path, has_labels: bool) -> Iterable[RowData]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            yield parse_row_dict(r, has_labels=has_labels)


def load_rows(csv_path: Path, has_labels: bool) -> List[RowData]:
    return list(iter_rows(csv_path, has_labels=has_labels))


def count_csv_rows(csv_path: Path) -> int:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def load_rows_sampled(csv_path: Path, has_labels: bool, sample_size: int, seed: int = 42) -> List[RowData]:
    if sample_size <= 0:
        return load_rows(csv_path, has_labels=has_labels)

    rng = random.Random(seed)
    sample: List[RowData] = []
    seen = 0

    for row in iter_rows(csv_path, has_labels=has_labels):
        seen += 1
        if len(sample) < sample_size:
            sample.append(row)
            continue

        j = rng.randint(1, seen)
        if j <= sample_size:
            sample[j - 1] = row

    return sample


def evaluate_on_train(rows: Sequence[RowData], ranked_map: Dict[str, List[RankedChunk]], p: Params) -> Dict[str, float]:
    total_score = 0.0
    total_f1 = 0.0
    total_eff = 0.0

    for i, row in enumerate(rows, start=1):
        ranked = ranked_map[row.rid]
        selected = select_evidence(row, ranked, p)
        selected_ids = [x.chunk.cid for x in selected]
        pred_answer = answer_from_context(row, selected, ranked)

        f1 = answer_f1(pred_answer, row.gold_answer)
        eff = evidence_efficiency(selected_ids, row.gold_evidence, max(row.num_chunks, 1))
        score = 0.5 * f1 + 0.5 * eff

        total_f1 += f1
        total_eff += eff
        total_score += score

        if GLOBAL_QA_READER is not None and i % 100 == 0:
            print(f"[train-eval] processed {i}/{len(rows)}")

    n = max(len(rows), 1)
    return {
        "score": total_score / n,
        "answer_f1": total_f1 / n,
        "evidence_eff": total_eff / n,
    }


def tune_params(train_rows: Sequence[RowData], ranked_map: Dict[str, List[RankedChunk]], seed: int = 42) -> Params:
    random.seed(seed)

    # Lean tuning grid for runtime safety in raw no-arg execution.
    keep_values = [0.62, 0.70]
    no_chunk_values = [0.00, 0.03]
    k_defaults = [2, 3, 4]
    redundancy_values = [0.78, 0.84]

    best = None
    best_metrics = {"score": -1.0, "answer_f1": 0.0, "evidence_eff": 0.0}

    for keep in keep_values:
        for no_chunk in no_chunk_values:
            for k_def in k_defaults:
                for red in redundancy_values:
                    p = Params(
                        keep_ratio=keep,
                        no_chunk_threshold=no_chunk,
                        k_default=k_def,
                        k_numeric=min(6, k_def + 1),
                        k_date=max(2, k_def),
                        k_person=max(2, k_def - 1),
                        k_location=max(2, k_def - 1),
                        redundancy_jaccard=red,
                    )
                    metrics = evaluate_on_train(train_rows, ranked_map, p)

                    if metrics["score"] > best_metrics["score"]:
                        best_metrics = metrics
                        best = p

    assert best is not None
    print(
        "[tune] best params:",
        best,
        "| train score=",
        f"{best_metrics['score']:.6f}",
        "f1=",
        f"{best_metrics['answer_f1']:.6f}",
        "eff=",
        f"{best_metrics['evidence_eff']:.6f}",
    )
    return best


def predict_rows(rows: Sequence[RowData], ranked_map: Dict[str, List[RankedChunk]], p: Params) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for i, row in enumerate(rows, start=1):
        ranked = ranked_map[row.rid]
        selected = select_evidence(row, ranked, p)
        selected_ids = [x.chunk.cid for x in selected]

        # Validate ids are actually in context chunk list.
        valid_ids = {c.cid for c in row.chunks}
        selected_ids = [cid for cid in selected_ids if cid in valid_ids]

        # Canonical order by numeric id.
        selected_ids = sorted(
            list(dict.fromkeys(selected_ids)),
            key=lambda cid: int(cid[1:]) if cid[1:].isdigit() else 999,
        )

        answer = answer_from_context(row, selected, ranked)
        if not answer:
            answer = "unknown"

        out.append((row.rid, ",".join(selected_ids), answer))

        if GLOBAL_QA_READER is not None and i % 100 == 0:
            print(f"[predict] processed {i}/{len(rows)}")

    return out


def write_submission(rows: Sequence[Tuple[str, str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "evidence_ids", "answer"])
        for rid, evid, ans in rows:
            writer.writerow([rid, evid, ans])


def build_rankings(
    rows: Sequence[RowData],
    progress_prefix: str = "rank",
    progress_every: int = 5000,
) -> Dict[str, List[RankedChunk]]:
    ranked_map: Dict[str, List[RankedChunk]] = {}
    total = len(rows)
    start = time.time()
    for i, row in enumerate(rows, start=1):
        ranked_map[row.rid] = rank_chunks(row)

        if progress_every > 0 and (i % progress_every == 0 or i == total):
            elapsed = time.time() - start
            rate = i / max(elapsed, 1e-6)
            print(f"[{progress_prefix}] {i}/{total} ({rate:.1f} rows/s)")

    return ranked_map


def default_params() -> Params:
    return Params(
        keep_ratio=0.62,
        no_chunk_threshold=0.03,
        k_default=3,
        k_numeric=4,
        k_date=3,
        k_person=2,
        k_location=2,
        redundancy_jaccard=0.84,
    )


def has_required_dataset_files(path: Path) -> bool:
    return (path / "train.csv").exists() and (path / "test.csv").exists()


def discover_data_dir(preferred: Optional[Path] = None) -> Path:
    candidates: List[Path] = []

    def add(p: Path) -> None:
        p = p.resolve()
        if p not in candidates:
            candidates.append(p)

    if preferred is not None:
        add(preferred)

    for env_var in ["SHIPD_DATA_DIR", "DATA_DIR"]:
        v = os.environ.get(env_var, "").strip()
        if v:
            add(Path(v))

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent

    add(cwd / "dataset" / "public")
    add(cwd / "data")
    add(cwd / "dataset")
    add(cwd)

    add(script_dir / "dataset" / "public")
    add(script_dir / "data")
    add(script_dir / "dataset")
    add(script_dir)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for p in kaggle_input.rglob("train.csv"):
            parent = p.parent
            if (parent / "test.csv").exists():
                add(parent)

    valid = [p for p in candidates if has_required_dataset_files(p)]
    if not valid:
        raise FileNotFoundError("Could not find directory containing train.csv and test.csv")

    valid.sort(key=lambda p: ((p / "sample_submission.csv").exists(), -len(p.parts), str(p)), reverse=True)
    return valid[0]


def default_output_path() -> Path:
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists() and kaggle_working.is_dir():
        return kaggle_working / "submission.csv"
    script_dir = Path(__file__).resolve().parent
    return script_dir / "working" / "submission.csv"


def rerank_single_row(row: RowData, ranked: List[RankedChunk], reranker: Optional[LearnedReranker]) -> List[RankedChunk]:
    if reranker is None or np is None or not ranked:
        return ranked

    total = max(1, len(ranked))
    feats = [chunk_feature_vector(row, rc, i, total) for i, rc in enumerate(ranked)]
    X_np = np.asarray(feats, dtype=float)
    probs = predict_reranker_probs(reranker, X_np)

    rescored: List[RankedChunk] = []
    w = reranker.blend_weight
    for i, rc in enumerate(ranked):
        prob = float(probs[i])
        blended = (1.0 - w) * rc.score + w * prob
        signals = dict(rc.signals)
        signals["prob"] = prob
        rescored.append(RankedChunk(chunk=rc.chunk, score=blended, signals=signals))

    rescored.sort(key=lambda x: x.score, reverse=True)
    return rescored


def predict_csv_stream(
    test_csv_path: Path,
    output_path: Path,
    p: Params,
    reranker: Optional[LearnedReranker],
    test_limit: int = 0,
    expected_total: Optional[int] = None,
    progress_every: int = 2000,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    start = time.time()
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "evidence_ids", "answer"])

        for row in iter_rows(test_csv_path, has_labels=False):
            if test_limit > 0 and written >= test_limit:
                break

            ranked = rank_chunks(row)
            ranked = rerank_single_row(row, ranked, reranker)
            selected = select_evidence(row, ranked, p)
            selected_ids = [x.chunk.cid for x in selected]

            # Validate ids are actually in context chunk list.
            valid_ids = {c.cid for c in row.chunks}
            selected_ids = [cid for cid in selected_ids if cid in valid_ids]

            selected_ids = sorted(
                list(dict.fromkeys(selected_ids)),
                key=lambda cid: int(cid[1:]) if cid[1:].isdigit() else 999,
            )

            answer = answer_from_context(row, selected, ranked) or "unknown"
            writer.writerow([row.rid, ",".join(selected_ids), answer])

            written += 1
            if progress_every > 0 and written % progress_every == 0:
                elapsed = time.time() - start
                rate = written / max(elapsed, 1e-6)
                if expected_total is not None and expected_total > 0:
                    print(f"[predict] {written}/{expected_total} ({rate:.1f} rows/s)")
                else:
                    print(f"[predict] {written} rows ({rate:.1f} rows/s)")

    return written


def main() -> None:
    global GLOBAL_QA_READER

    parser = argparse.ArgumentParser(description="Single-file solution for Cost Efficient RAG Optimization")
    parser.add_argument("--data-dir", default="", help="Path containing train.csv and test.csv (auto-detected when empty)")
    parser.add_argument("--output", default="", help="Submission CSV output path (auto-selected when empty)")
    parser.add_argument("--no-tune", action="store_true", help="Disable train-time parameter tuning")
    parser.add_argument("--disable-reranker", action="store_true", help="Disable learned logistic chunk reranker")
    parser.add_argument("--skip-train", action="store_true", help="Force fast mode: no train load, no tuning, no reranker")
    parser.add_argument("--train-sample", type=int, default=1200, help="Train rows used for tuning/reranker (0 = all rows)")
    parser.add_argument("--test-limit", type=int, default=0, help="Limit test rows for debugging (0 = all rows)")
    parser.add_argument(
        "--stream-test",
        action="store_true",
        help="Stream test prediction directly to submission to reduce memory usage",
    )
    parser.add_argument("--rank-progress-every", type=int, default=5000, help="Progress print frequency for ranking")
    parser.add_argument(
        "--qa-model",
        default="",
        help="Hugging Face extractive QA model for final prediction (empty = auto)",
    )
    parser.add_argument(
        "--force-gpu",
        action="store_true",
        help="Force GPU-backed reranker path (enabled by default).",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution. By default, a CUDA GPU is required.",
    )
    parser.add_argument(
        "--qa-train-eval",
        action="store_true",
        help="Run one post-tuning train evaluation with QA model enabled",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(stream_test=True, force_gpu=True)
    args = parser.parse_args()

    gpu_ok = gpu_visible_via_nvidia_smi()
    if args.force_gpu and not gpu_ok and not args.allow_cpu:
        raise RuntimeError("GPU is required by default. Enable Kaggle GPU or run with --allow-cpu.")

    auto_qa_model = "deepset/minilm-uncased-squad2"
    qa_model_arg = (args.qa_model or "").strip()
    if qa_model_arg.lower() in {"", "auto", "default"}:
        if gpu_ok:
            args.qa_model = auto_qa_model
            print(f"[info] qa_model=auto -> {args.qa_model}")
        else:
            args.qa_model = "none"
            print("[info] qa_model=auto -> none (cpu fallback)")
    else:
        args.qa_model = qa_model_arg
        print(f"[info] qa_model={args.qa_model}")

    if gpu_ok:
        try:
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                text=True,
            ).strip()
            print(f"[info] gpu={gpu_info}")
        except Exception:
            print("[info] gpu=visible")
    else:
        print("[warn] running on CPU fallback")

    preferred_data_dir = Path(args.data_dir) if args.data_dir else None
    data_dir = discover_data_dir(preferred_data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    print(f"[info] data_dir={data_dir}")

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Expected train/test CSV files under {data_dir}")

    if args.skip_train:
        args.no_tune = True
        args.disable_reranker = True
        args.qa_train_eval = False
        print("[info] --skip-train enabled; disabling tune/reranker/train-eval")

    need_train = (not args.no_tune) or (not args.disable_reranker) or args.qa_train_eval

    train_rows: List[RowData] = []
    train_ranked: Dict[str, List[RankedChunk]] = {}
    reranker: Optional[LearnedReranker] = None

    if need_train:
        total_train = count_csv_rows(train_path)
        if args.train_sample > 0:
            print(f"[info] loading sampled train rows (sample={args.train_sample}, total={total_train})...")
            train_rows = load_rows_sampled(train_path, has_labels=True, sample_size=args.train_sample, seed=args.seed)
        else:
            print(f"[info] loading full train rows (total={total_train})...")
            train_rows = load_rows(train_path, has_labels=True)

        print(f"[info] loaded train rows for fitting/tuning: {len(train_rows)}")
        print("[info] ranking train chunks...")
        train_ranked = build_rankings(
            train_rows,
            progress_prefix="train-rank",
            progress_every=args.rank_progress_every,
        )

        if not args.disable_reranker:
            reranker = train_chunk_reranker(
                train_rows,
                train_ranked,
                force_gpu=args.force_gpu and not args.allow_cpu,
                seed=args.seed,
            )
            if reranker is not None:
                print("[info] applying learned reranker to train rankings...")
                train_ranked = apply_chunk_reranker(train_rows, train_ranked, reranker)
        else:
            print("[info] learned reranker disabled")
    else:
        print("[info] no train-time steps requested; skipping train load")

    if args.no_tune:
        params = default_params()
        print("[info] using default params:", params)
    else:
        if not train_rows:
            print("[warn] tuning requested but no train rows available; using default params")
            params = default_params()
        else:
            print("[info] tuning params on train sample...")
            GLOBAL_QA_READER = None
            params = tune_params(train_rows, train_ranked, seed=args.seed)

    # Optional one-shot train evaluation with QA enabled (after tuning only).
    if args.qa_train_eval and train_rows:
        GLOBAL_QA_READER = maybe_init_qa_reader(args.qa_model)
        train_metrics = evaluate_on_train(train_rows, train_ranked, params)
        print(
            "[train] score=",
            f"{train_metrics['score']:.6f}",
            "f1=",
            f"{train_metrics['answer_f1']:.6f}",
            "eff=",
            f"{train_metrics['evidence_eff']:.6f}",
        )
        GLOBAL_QA_READER = None

    output_path = Path(args.output) if args.output else default_output_path()
    print(f"[info] output={output_path}")

    if args.stream_test:
        total_test = count_csv_rows(test_path)
        expected_total = min(total_test, args.test_limit) if args.test_limit > 0 else total_test
        print(f"[info] streaming test prediction for {expected_total} rows...")
        GLOBAL_QA_READER = maybe_init_qa_reader(args.qa_model)
        written = predict_csv_stream(
            test_csv_path=test_path,
            output_path=output_path,
            p=params,
            reranker=reranker,
            test_limit=args.test_limit,
            expected_total=expected_total,
            progress_every=max(1000, args.rank_progress_every // 2),
        )
        GLOBAL_QA_READER = None
        print(f"[done] wrote {written} rows to {output_path}")
        return

    print("[info] loading test rows...")
    test_rows = load_rows(test_path, has_labels=False)
    if args.test_limit > 0 and len(test_rows) > args.test_limit:
        test_rows = test_rows[: args.test_limit]
        print(f"[info] using test subset: {len(test_rows)} rows")

    print("[info] ranking test chunks...")
    test_ranked = build_rankings(
        test_rows,
        progress_prefix="test-rank",
        progress_every=args.rank_progress_every,
    )

    if reranker is not None:
        print("[info] applying learned reranker to test rankings...")
        test_ranked = apply_chunk_reranker(test_rows, test_ranked, reranker)

    print("[info] predicting test...")
    GLOBAL_QA_READER = maybe_init_qa_reader(args.qa_model)
    submission_rows = predict_rows(test_rows, test_ranked, params)
    GLOBAL_QA_READER = None

    write_submission(submission_rows, output_path)
    print(f"[done] wrote {len(submission_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
