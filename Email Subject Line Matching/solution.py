import re
import importlib
from collections import Counter
from itertools import permutations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
	_sentence_transformers = importlib.import_module("sentence_transformers")
	SentenceTransformer = _sentence_transformers.SentenceTransformer
except Exception:
	SentenceTransformer = None


# Tuned on repeated synthetic block validation.
NEG_PER_POS = 3
SMOOTH_ALPHA = 5.0
MIN_TOKEN_FREQ = 8
MIN_LOG_RATIO = 0.02

ENTITY_WEIGHT = 1.2
ASSOC_WEIGHT = 1.1
OVERLAP_WEIGHT = 0.45
JACCARD_WEIGHT = 0.8

W_ASSOC = 1.2
W_TFIDF = 0.2
W_BI = 1.2

BI_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RANDOM_SEED = 42


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
ENTITY_RE = re.compile(r"\b(Project|Program|Initiative|Stream)\s+[A-Z][A-Za-z0-9-]*\b")

STOPWORDS = {
	"the",
	"a",
	"an",
	"and",
	"or",
	"to",
	"for",
	"of",
	"on",
	"in",
	"at",
	"with",
	"from",
	"by",
	"is",
	"are",
	"be",
	"this",
	"that",
	"it",
	"as",
	"we",
	"you",
	"your",
	"our",
	"please",
	"regards",
	"thanks",
	"hi",
	"hello",
	"re",
	"fw",
}


def token_set(text: str) -> set:
	return {
		t.lower()
		for t in TOKEN_RE.findall(str(text))
		if len(t) >= 3 and t.lower() not in STOPWORDS
	}


def entity_set(text: str) -> set:
	return {m.group(0) for m in ENTITY_RE.finditer(str(text))}


def build_association_scores(train_df: pd.DataFrame) -> dict:
	rng = np.random.default_rng(RANDOM_SEED)

	pos_counts = Counter()
	neg_counts = Counter()
	body_token_freq = Counter()
	subj_token_freq = Counter()

	grouped = {
		category: group.reset_index(drop=True)
		for category, group in train_df.groupby("category")
	}

	for _, row in train_df.iterrows():
		body_tokens = list(token_set(row["body"]))
		subj_tokens = list(token_set(row["subject"]))
		body_token_freq.update(body_tokens)
		subj_token_freq.update(subj_tokens)

		for bt in body_tokens:
			for st in subj_tokens:
				pos_counts[(bt, st)] += 1

	for _, group in grouped.items():
		n = len(group)
		for i in range(n):
			body_tokens = list(token_set(group.loc[i, "body"]))

			for _ in range(NEG_PER_POS):
				j = int(rng.integers(0, n - 1))
				if j >= i:
					j += 1
				subj_tokens = list(token_set(group.loc[j, "subject"]))

				for bt in body_tokens:
					for st in subj_tokens:
						neg_counts[(bt, st)] += 1

	keep_body_tokens = {tok for tok, c in body_token_freq.items() if c >= MIN_TOKEN_FREQ}
	keep_subj_tokens = {tok for tok, c in subj_token_freq.items() if c >= MIN_TOKEN_FREQ}

	scores = {}
	for (bt, st), pos_val in pos_counts.items():
		if bt not in keep_body_tokens or st not in keep_subj_tokens:
			continue

		neg_val = neg_counts[(bt, st)]
		log_ratio = np.log((pos_val + SMOOTH_ALPHA) / (neg_val + SMOOTH_ALPHA))

		if log_ratio > MIN_LOG_RATIO:
			scores[(bt, st)] = float(log_ratio)

	return scores


def pair_score(body: str, subject: str, assoc_scores: dict) -> float:
	body_tokens = token_set(body)
	subj_tokens = token_set(subject)
	body_entities = entity_set(body)
	subj_entities = entity_set(subject)

	assoc_sum = 0.0
	for bt in body_tokens:
		best = 0.0
		for st in subj_tokens:
			score = assoc_scores.get((bt, st), 0.0)
			if score > best:
				best = score
		assoc_sum += best

	token_overlap = len(body_tokens & subj_tokens)
	union = len(body_tokens | subj_tokens)
	jaccard = (token_overlap / union) if union else 0.0
	entity_overlap = len(body_entities & subj_entities)

	return (
		ENTITY_WEIGHT * entity_overlap
		+ ASSOC_WEIGHT * assoc_sum
		+ OVERLAP_WEIGHT * token_overlap
		+ JACCARD_WEIGHT * jaccard
	)


def fit_semantic_vectorizers(train_df: pd.DataFrame) -> tuple:
	corpus = pd.concat([train_df["body"], train_df["subject"]], axis=0).astype(str).tolist()

	word_vec = TfidfVectorizer(
		analyzer="word",
		ngram_range=(1, 2),
		min_df=2,
		max_df=0.98,
		sublinear_tf=True,
		norm="l2",
		max_features=120000,
	)
	char_vec = TfidfVectorizer(
		analyzer="char_wb",
		ngram_range=(3, 5),
		min_df=2,
		sublinear_tf=True,
		norm="l2",
		max_features=180000,
	)

	word_vec.fit(corpus)
	char_vec.fit(corpus)
	return word_vec, char_vec


def semantic_matrix(body_texts: list, subject_texts: list, word_vec, char_vec) -> np.ndarray:
	bw = word_vec.transform(body_texts)
	sw = word_vec.transform(subject_texts)
	bc = char_vec.transform(body_texts)
	sc = char_vec.transform(subject_texts)

	word_sim = (bw @ sw.T).toarray()
	char_sim = (bc @ sc.T).toarray()
	return 0.6 * word_sim + 0.4 * char_sim


def zscore_matrix(mat: np.ndarray) -> np.ndarray:
	m = mat.astype(np.float32)
	mu = float(m.mean())
	sigma = float(m.std())
	if sigma < 1e-6:
		return m - mu
	return (m - mu) / sigma


def build_bi_lookup(test_df: pd.DataFrame, test_subjects_df: pd.DataFrame) -> dict | None:
	if SentenceTransformer is None:
		print("Warning: sentence-transformers is unavailable; falling back to lexical+tfidf only.")
		return None

	try:
		model = SentenceTransformer(BI_MODEL_NAME, device="cpu")
	except Exception as exc:
		print(f"Warning: could not load {BI_MODEL_NAME}; fallback active. Error: {exc}")
		return None

	texts = sorted(
		set(test_df["body"].astype(str).tolist())
		| set(test_subjects_df["subject"].astype(str).tolist())
	)

	emb = model.encode(
		texts,
		batch_size=128,
		show_progress_bar=False,
		convert_to_numpy=True,
		normalize_embeddings=True,
	)

	return {t: emb[i].astype(np.float32) for i, t in enumerate(texts)}


def bi_matrix(body_texts: list, subject_texts: list, lookup: dict | None) -> np.ndarray:
	if lookup is None:
		return np.zeros((4, 4), dtype=np.float32)

	b = np.vstack([lookup[str(t)] for t in body_texts]).astype(np.float32)
	s = np.vstack([lookup[str(t)] for t in subject_texts]).astype(np.float32)

	b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
	s = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-12)
	return b @ s.T


def best_assignment(score_matrix: np.ndarray) -> np.ndarray:
	best_perm = None
	best_score = -1e18

	for perm in permutations(range(4)):
		score = 0.0
		for i in range(4):
			score += score_matrix[i, perm[i]]
		if score > best_score:
			best_score = score
			best_perm = perm

	return np.asarray(best_perm, dtype=np.int64)


def predict_submission(train_df: pd.DataFrame, test_df: pd.DataFrame, test_subjects_df: pd.DataFrame) -> pd.DataFrame:
	assoc_scores = build_association_scores(train_df)
	word_vec, char_vec = fit_semantic_vectorizers(train_df)
	bi_lookup = build_bi_lookup(test_df, test_subjects_df)

	out_rows = []

	for block_id, body_group in test_df.groupby("block_id"):
		bodies = body_group.sort_values("body_index").reset_index(drop=True)
		subjects = (
			test_subjects_df[test_subjects_df["block_id"] == block_id]
			.sort_values("subject_letter")
			.reset_index(drop=True)
		)

		body_texts = bodies["body"].tolist()
		subject_texts = subjects["subject"].tolist()
		subject_letters = subjects["subject_letter"].tolist()

		assoc_matrix = np.zeros((4, 4), dtype=np.float32)
		for i in range(4):
			for j in range(4):
				assoc_matrix[i, j] = pair_score(body_texts[i], subject_texts[j], assoc_scores)

		assoc_z = zscore_matrix(assoc_matrix)
		tfidf_z = zscore_matrix(semantic_matrix(body_texts, subject_texts, word_vec, char_vec))
		bi_z = zscore_matrix(bi_matrix(body_texts, subject_texts, bi_lookup))

		score_matrix = W_ASSOC * assoc_z + W_TFIDF * tfidf_z + W_BI * bi_z

		assignment = best_assignment(score_matrix)

		for i, subj_col in enumerate(assignment):
			out_rows.append(
				{
					"block_id": int(block_id),
					"body_index": int(bodies.loc[i, "body_index"]),
					"assigned_subject": subject_letters[subj_col],
				}
			)

	submission = pd.DataFrame(out_rows).sort_values(["block_id", "body_index"]).reset_index(drop=True)
	return submission


def main() -> None:
	train = pd.read_csv("./dataset/public/train.csv")
	test = pd.read_csv("./dataset/public/test.csv")
	test_subjects = pd.read_csv("./dataset/public/test_subjects.csv")

	submission = predict_submission(train, test, test_subjects)
	submission.to_csv("./working/submission.csv", index=False)

	print("Saved submission to ./working/submission.csv")
	print(submission.head(8).to_string(index=False))


if __name__ == "__main__":
	main()