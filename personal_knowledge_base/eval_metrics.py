"""Retrieval and generation metrics ported from Tencent/WeKnora.

Source: internal/application/service/metric at commit 7a3ca49 (precision.go,
recall.go, mrr.go, map.go, ndcg.go, bleu.go, rouge.go, rouge_score.go,
common.go). Behaviour, including duplicate handling and edge cases, follows the
Go implementation; the ported Go unit tests live in test_eval_metrics.py.

Retrieval metrics take ``gt`` (a list of ground-truth id groups) and ``ids``
(retrieved ids in rank order). Generation metrics compare a generated answer
with a reference answer after WeKnora's sentence split and jieba tokenisation.
"""

import math

import jieba
import regex

_WORD_TOKEN = regex.compile(r"([\p{Han}]+)|([a-zA-Z0-9_.,!?]+)|(\p{P})")

BLEU1 = (1.0, 0.0, 0.0, 0.0)
BLEU2 = (0.5, 0.5, 0.0, 0.0)
BLEU4 = (0.25, 0.25, 0.25, 0.25)


# ── common.go ────────────────────────────────────────────────────────


def split_sentences(text: str) -> list[str]:
    parts = regex.split(r"[。.]", text or "")
    return [part.strip() for part in parts if part.strip()]


def split_into_words(sentences: list[str]) -> list[str]:
    tokens: list[str] = []
    for text in sentences or []:
        for match in _WORD_TOKEN.finditer(text):
            chinese, english, punctuation = match.groups()
            if chinese:
                tokens.extend(jieba.lcut(chinese, HMM=True))
            elif english:
                tokens.extend(english.split())
            elif punctuation:
                tokens.append(punctuation)
    return tokens


def _hit(ids, gt_set) -> int:
    return len({item for item in ids if item in gt_set})


# ── retrieval metrics ────────────────────────────────────────────────


def precision(gt, ids) -> float:
    if not gt or not ids:
        return 0.0
    return sum(_hit(ids, set(group)) / len(ids) for group in gt) / len(gt)


def recall(gt, ids) -> float:
    if not gt or not ids:
        return 0.0
    total = 0.0
    for group in gt:
        group_set = set(group)
        if group_set:
            total += _hit(ids, group_set) / len(group_set)
    return total / len(gt)


def mrr(gt, ids) -> float:
    if not gt:
        return 0.0
    total = 0.0
    for group in gt:
        group_set = set(group)
        for rank, item in enumerate(ids or []):
            if item in group_set:
                total += 1.0 / (rank + 1)
                break
    return total / len(gt)


def mean_average_precision(gt, ids) -> float:
    if not gt:
        return 0.0
    ap_sum = 0.0
    for group in gt:
        remaining = set(group)
        relevant_count = len(remaining)
        hit_count = 0
        ap = 0.0
        for rank, item in enumerate(ids or []):
            if item in remaining:
                remaining.discard(item)
                hit_count += 1
                ap += hit_count / (rank + 1)
        if hit_count:
            ap /= relevant_count
        ap_sum += ap
    return ap_sum / len(gt)


def ndcg(gt, ids, k: int) -> float:
    top = list(ids or [])[: max(k, 0)]
    relevant = {item for group in gt or [] for item in group}
    dcg = 0.0
    counted = set()
    for rank, item in enumerate(top):
        if item not in relevant or item in counted:
            continue
        counted.add(item)
        dcg += 1.0 / math.log2(rank + 2)
    ideal_len = min(len(relevant), max(k, 0))
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_len))
    return dcg / idcg if idcg else 0.0


# ── bleu.go ──────────────────────────────────────────────────────────


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _counts(phrases: list[tuple]) -> dict:
    counts: dict = {}
    for phrase in phrases:
        counts[phrase] = counts.get(phrase, 0) + 1
    return counts


def _modified_precision(candidate, reference, n: int, smoothing: bool) -> float:
    phrases = _ngrams(candidate, n)
    if not phrases:
        return 0.0
    counts = _counts(phrases)
    reference_counts = _counts(_ngrams(reference, n))
    clipped = sum(min(count, reference_counts.get(gram, 0)) for gram, count in counts.items())
    factor = 1.0 if smoothing else 0.0
    return (clipped + factor) / (sum(counts.values()) + factor)


def bleu(generated: str, reference: str, weights=BLEU4, smoothing: bool = False) -> float:
    candidate = [token.lower() for token in split_into_words(split_sentences(generated))]
    ref = [token.lower() for token in split_into_words(split_sentences(reference))]
    precisions = [_modified_precision(candidate, ref, i + 1, smoothing) for i in range(len(weights))]
    s = 0.0
    overlap = 0
    for weight, pn in zip(weights, precisions):
        if not smoothing and weight > 0 and pn == 0:
            return 0.0
        if pn > 0:
            overlap += 1
            s += weight * math.log(pn)
    if overlap == 0 or not candidate:
        return 0.0
    c, r = len(candidate), len(ref)
    brevity = 1.0 if c > r else math.exp(1 - r / c)
    return brevity * math.exp(s)


# ── rouge.go / rouge_score.go ────────────────────────────────────────
# WeKnora's Ngrams type is compared only through Len/Intersection/Union over
# its keys, so distinct-item sets reproduce it exactly.


def _f_score(p: float, r: float) -> float:
    return 2.0 * ((p * r) / (p + r + 1e-8))


def _rouge_n(hyp_sentences, ref_sentences, n: int) -> dict:
    hyp = {" ".join(g) for g in _ngrams(split_into_words(hyp_sentences), n)}
    ref = {" ".join(g) for g in _ngrams(split_into_words(ref_sentences), n)}
    overlap = len(hyp & ref)
    p = overlap / len(hyp) if hyp else 0.0
    r = overlap / len(ref) if ref else 0.0
    return {"p": p, "r": r, "f": _f_score(p, r)}


def _lcs_words(x: list[str], y: list[str]) -> set[str]:
    n, m = len(x), len(y)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if x[i - 1] == y[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    words = set()
    i, j = n, m
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1]:
            words.add(x[i - 1])
            i, j = i - 1, j - 1
        elif table[i - 1][j] > table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return words


def _rouge_l(hyp_sentences, ref_sentences) -> dict:
    m = len(set(split_into_words(ref_sentences)))
    n = len(set(split_into_words(hyp_sentences)))
    union: set[str] = set()
    for ref_sentence in ref_sentences:
        ref_words = split_into_words([ref_sentence])
        for hyp_sentence in hyp_sentences:
            union |= _lcs_words(ref_words, split_into_words([hyp_sentence]))
    llcs = len(union)
    r = llcs / m if m else 0.0
    p = llcs / n if n else 0.0
    return {"f": _f_score(p, r), "p": p, "r": r}


def rouge(generated: str, reference: str, variant: str = "rouge-1", stat: str = "f") -> float:
    hyp = split_sentences(generated)
    ref = split_sentences(reference)
    if variant == "rouge-l":
        scores = _rouge_l(hyp, ref)
    else:
        scores = _rouge_n(hyp, ref, int(variant.split("-")[1]))
    return scores[stat]


# ── aggregate, matching WeKnora's RetrievalMetrics / GenerationMetrics ─


def retrieval_metrics(gt, ids) -> dict:
    return {
        "precision": precision(gt, ids),
        "recall": recall(gt, ids),
        "ndcg3": ndcg(gt, ids, 3),
        "ndcg10": ndcg(gt, ids, 10),
        "mrr": mrr(gt, ids),
        "map": mean_average_precision(gt, ids),
    }


def generation_metrics(generated: str, reference: str) -> dict:
    return {
        "bleu1": bleu(generated, reference, BLEU1),
        "bleu2": bleu(generated, reference, BLEU2),
        "bleu4": bleu(generated, reference, BLEU4),
        "rouge1": rouge(generated, reference, "rouge-1", "f"),
        "rouge2": rouge(generated, reference, "rouge-2", "f"),
        "rougel": rouge(generated, reference, "rouge-l", "f"),
    }
