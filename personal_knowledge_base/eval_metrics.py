"""检索评测指标。纯函数，不依赖 Django，输入是每个排名位置命中了哪些标准答案。"""
import math

DEFAULT_KS = (1, 3, 5, 10)


def gold_hits(row: dict, gold: dict) -> set[str]:
    """一条检索结果命中了哪些标准答案。gold 可以按来源 ID、文档标题或原文片段标注。"""
    hits = set()
    source = str(row.get("knowledge_source") or "")
    if source in (gold.get("expected_sources") or []):
        hits.add(f"source:{source}")
    title = str(row.get("knowledge_title") or "")
    if title in (gold.get("expected_titles") or []):
        hits.add(f"title:{title}")
    content = "".join(str(row.get("content") or "").split())
    for evidence in gold.get("evidence") or []:
        if "".join(evidence.split()) in content:
            hits.add(f"evidence:{evidence}")
    return hits


def gold_size(gold: dict) -> int:
    return sum(len(set(gold.get(key) or [])) for key in ("expected_sources", "expected_titles", "evidence"))


def ranking_metrics(hits_by_rank: list[set[str]], total_gold: int, ks=DEFAULT_KS) -> dict:
    """
    hits_by_rank[i] 是第 i+1 名命中的标准答案集合。
    Recall 和 nDCG 按“找到了多少个不同的标准答案”计算，同一个答案被多条结果命中只算一次。
    """
    total_gold = max(total_gold, 1)
    first_rank = next((rank for rank, hits in enumerate(hits_by_rank, 1) if hits), None)
    metrics = {"reciprocal_rank": 1.0 / first_rank if first_rank else 0.0}
    for k in ks:
        found: set[str] = set()
        dcg = 0.0
        for rank, hits in enumerate(hits_by_rank[:k], 1):
            new = hits - found
            if new:
                dcg += len(new) / math.log2(rank + 1)
                found |= new
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(total_gold, k) + 1))
        metrics[f"hit_at_{k}"] = 1.0 if found else 0.0
        metrics[f"recall_at_{k}"] = len(found) / total_gold
        metrics[f"ndcg_at_{k}"] = dcg / ideal if ideal else 0.0
    return metrics


def summarize(details: list[dict], ks=DEFAULT_KS) -> dict:
    count = max(len(details), 1)
    summary = {"questions": len(details), "mrr": round(sum(row["reciprocal_rank"] for row in details) / count, 4)}
    for k in ks:
        for name in ("hit", "recall", "ndcg"):
            key = f"{name}_at_{k}"
            summary[key] = round(sum(row[key] for row in details) / count, 4)
    summary["avg_latency_ms"] = round(sum(row["latency_ms"] for row in details) / count, 2)
    return summary
