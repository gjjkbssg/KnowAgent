"""Answer scoring for the Agent evaluation set (interview_materials/agent_eval_tasks.json).

A task is answered correctly when every group in ``must`` has at least one
alternative that appears in the normalised answer. Unanswerable tasks are
correct when the answer says the knowledge base has no such information.
"""

import unicodedata

REFUSAL_MARKERS = (
    "没有", "未提及", "未找到", "找不到", "无法确定", "不包含", "未包含", "未记录", "无相关",
    "无法回答", "不清楚", "暂无", "无法提供", "无法找到", "未查到", "不存在",
)


def normalize(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text or "").lower().split())


def score_answer(task: dict, answer: str) -> bool:
    text = normalize(answer)
    if task.get("unanswerable"):
        return any(marker in text for marker in REFUSAL_MARKERS)
    return all(any(normalize(option) in text for option in group) for group in task["must"])


def doc_key(title: str, keys: list[str]) -> str | None:
    """Map a knowledge title to the corpus file stem it came from."""
    for key in keys:
        if key in (title or ""):
            return key
    return None
