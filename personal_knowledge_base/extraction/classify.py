import json
import re
import time

from personal_knowledge_base import model_providers
from personal_knowledge_base.extraction.tender import _model_usage_totals, _usage_delta


LABELS = ("gkzb", "zbgg", "cjgg", "gzgg")
LABEL_NAMES = {
    "gkzb": "公开招标公告",
    "zbgg": "中标公告",
    "cjgg": "成交公告",
    "gzgg": "更正公告",
}
MAX_TEXT_CHARS = 3000


def strip_title(text: str, title: str) -> str:
    normalized_title = re.sub(r"\s+", "", title or "")
    if not normalized_title:
        return text or ""
    return "".join(
        line
        for line in (text or "").splitlines(keepends=True)
        if re.sub(r"\s+", "", line) != normalized_title
    )


_CORRECTION_RE = re.compile(
    r"原公告(?:的(?:采购项目编号|采购项目名称|内容)?|公告)?|"
    r"(?:更正|变更)(?:公告|事项|信息|内容|为|如下)|补充公告"
)
_TENDER_RE = re.compile(
    r"招标项目的潜在(?:投标人|供应商).{0,120}(?:获取招标文件|递交投标文件|提交投标文件)|"
    r"(?:投标截止时间|提交投标文件截止时间|投标文件递交截止时间)"
)
_METHOD_RE = re.compile(r"采购方式\s*[：:]\s*[^\r\n]{1,60}")
_METHOD_VALUE_RE = re.compile(r"采购方式\s*[：:]\s*(?P<method>[^\r\n]{1,60})")
_RESULT_RE = re.compile(r"中标|成交|结果公告|结果信息")
_BID_METHOD_RE = re.compile(r"公开招标|邀请招标")
_DEAL_METHOD_RE = re.compile(r"竞争性磋商|竞争性谈判|竞争性询价|询价|单一来源")
_BID_WORDING_RE = re.compile(r"中标(?:公告|结果|供应商)(?![（(])")
_DEAL_WORDING_RE = re.compile(r"成交(?:公告|结果|供应商)(?![）)])")
_ATTACHMENT_RE = re.compile(r"\S+\.(?:docx|pdf|xlsx|zip)\b", re.IGNORECASE)

SYSTEM_PROMPT = """你是政府采购公告类型分类器。只根据公告正文判断其在平台发布的公告类别。
类别代码：gkzb=公开招标公告，zbgg=中标公告，cjgg=成交公告，gzgg=更正公告。
对于中标/成交结果，优先依据正文中的“采购方式”：公开招标或邀请招标属于中标公告；竞争性磋商、竞争性谈判、询价、单一来源属于成交公告。
标题不作为输入。证据必须从正文原样复制且不超过80个字符；只提交一个最能支持类别的短句。"""

TOOL = {
    "type": "function",
    "function": {
        "name": "submit_notice_type",
        "description": "提交公告平台类型代码及正文证据。",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": list(LABELS)},
                "evidence": {"type": "string", "maxLength": 80},
            },
            "required": ["label", "evidence"],
            "additionalProperties": False,
        },
    },
}


def classify_rules(text: str) -> tuple[str | None, str | None]:
    text = (text or "")[:MAX_TEXT_CHARS]
    match = _CORRECTION_RE.search(text)
    if match:
        return "gzgg", match.group(0)

    match = _TENDER_RE.search(text)
    if match:
        return "gkzb", match.group(0)

    method_match = _METHOD_VALUE_RE.search(text)
    if method_match and _RESULT_RE.search(text):
        method = method_match.group("method")
        if _BID_METHOD_RE.search(method):
            return "zbgg", method_match.group(0)
        if _DEAL_METHOD_RE.search(method):
            return "cjgg", method_match.group(0)

    result_text = _ATTACHMENT_RE.sub(
        "",
        re.sub(r"中标[（(]成交[）)]|中标/成交|中标、成交", "", text),
    )
    match = _DEAL_WORDING_RE.search(result_text)
    if match:
        return "cjgg", match.group(0)
    match = _BID_WORDING_RE.search(result_text)
    if match:
        return "zbgg", match.group(0)
    return None, None


def _sklearn_components(analyzer: str):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    if analyzer == "char":
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=2,
        )
    elif analyzer == "word":
        import jieba

        vectorizer = TfidfVectorizer(
            analyzer="word",
            tokenizer=jieba.lcut,
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
    else:
        raise ValueError("analyzer must be 'char' or 'word'")
    return Pipeline(
        [
            ("tfidf", vectorizer),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=10,
                ),
            ),
        ]
    )


def train_tfidf(texts, labels, analyzer: str = "char"):
    if len(texts) != len(labels) or not texts:
        raise ValueError("texts and labels must have the same nonzero length")
    model = _sklearn_components(analyzer)
    model.fit([(text or "")[:MAX_TEXT_CHARS] for text in texts], labels)
    return model


def predict_tfidf(model, texts) -> list[str]:
    return list(model.predict([(text or "")[:MAX_TEXT_CHARS] for text in texts]))


def top_features(model, k: int = 10) -> dict[str, list[tuple[str, float]]]:
    if k < 1:
        raise ValueError("k must be positive")
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    names = vectorizer.get_feature_names_out()
    return {
        label: [
            (str(names[index]), float(classifier.coef_[class_index, index]))
            for index in classifier.coef_[class_index].argsort()[-k:][::-1]
        ]
        for class_index, label in enumerate(classifier.classes_)
    }


def _parse_content_json(content: str) -> dict:
    content = (content or "").strip()
    if content.startswith("```"):
        content = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            content,
            flags=re.IGNORECASE,
        )
    try:
        value = json.loads(content)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        start = content.find("{")
        if start < 0:
            return {}
        try:
            value, _ = json.JSONDecoder().raw_decode(content[start:])
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}


def _response_payload(response: dict) -> dict:
    for call in response.get("tool_calls") or []:
        function = call.get("function") or {}
        if function.get("name") != "submit_notice_type":
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        if isinstance(arguments, dict):
            return arguments
    return _parse_content_json(response.get("content") or "")


def classify_llm(
    text: str,
    tenant,
    *,
    response: dict | None = None,
    usage_override: dict | None = None,
    latency_ms: float | None = None,
    include_response: bool = False,
):
    text = (text or "")[:MAX_TEXT_CHARS]
    if response is None:
        before = _model_usage_totals(tenant)
        started = time.perf_counter()
        response = model_providers.chat_completion_raw(
            tenant,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "公告正文：\n" + text},
            ],
            tools=[TOOL],
            temperature=0,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        usage = _usage_delta(before, _model_usage_totals(tenant))
    else:
        usage = usage_override or {}
        latency_ms = float(latency_ms or 0)

    payload = _response_payload(response)
    label = payload.get("label")
    if label not in LABELS:
        label = None
    evidence = payload.get("evidence")
    if not isinstance(evidence, str):
        evidence = None
    whitespace_free_text = re.sub(r"\s+", "", text)
    whitespace_free_evidence = re.sub(r"\s+", "", evidence or "")
    usage = {
        **{
            key: int(usage.get(key, 0) or 0)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "latency_ms": float(latency_ms or 0),
        "evidence_ok": bool(
            label
            and evidence
            and len(evidence) <= 80
            and whitespace_free_evidence in whitespace_free_text
        ),
    }
    result = (label, evidence, usage)
    return (*result, response) if include_response else result
