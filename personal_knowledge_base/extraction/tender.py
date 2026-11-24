import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from personal_knowledge_base import model_providers
from personal_knowledge_base.extraction.normalize import (
    DATE_RE,
    _to_halfwidth,
    normalize_amount,
    normalize_code,
    normalize_datetime,
    normalize_name,
)


FIELDS = (
    "project_name",
    "project_code",
    "purchaser",
    "agency",
    "budget_yuan",
    "bid_deadline",
)
STRING_FIELDS = {"project_name", "project_code", "purchaser", "agency"}
SYSTEM_PROMPT = """你是招标公告信息抽取器。只根据用户给出的公告正文抽取字段，并调用 submit_tender_fields 提交结果。
规则：
1. 每个字段同时给出 value 和 evidence。除 budget_yuan 外，evidence 是从正文中原样复制的一段连续文字（不超过 80 字），且必须包含 value 本身。
2. 正文中找不到的字段，value 和 evidence 都填 null，不要猜测。
3. project_name：项目名称，不含“公开招标公告”等公告类型字样。
4. project_code：项目编号；有多个编号时，取标注为“项目编号”的那个。
5. purchaser：采购人（招标人）单位全称。agency：采购代理机构单位全称，没有代理机构时填 null。
6. budget_yuan：项目预算总金额，换算成元，只输出数字。evidence 是字符串数组：有项目总预算时，给出写有总预算的 1 段原文；只有各包预算时，给出每个包预算的原文各 1 段，value 为各包之和。
7. bid_deadline：提交投标文件截止时间，格式 YYYY-MM-DD HH:MM；不要误取获取招标文件的时间或投标文件递交的开始时间。"""


@dataclass
class FieldValue:
    value: Any
    evidence: str | list[str] | None
    source: str


_LINE_LABEL = r"(?m)^[ \t]*(?:\d+\s*[、.．]\s*)?"
_PROJECT_NAME_RE = re.compile(
    _LINE_LABEL + r"(?:采购项目名称|项目名称)\s*[：:]\s*(?P<value>[^\r\n]*)"
)
_PROJECT_CODE_PATTERNS = (
    re.compile(_LINE_LABEL + r"(?:项目编号|项目号)\s*[：:]\s*(?P<value>[^\r\n]*)"),
    re.compile(_LINE_LABEL + r"采购项目编号\s*[：:]\s*(?P<value>[^\r\n]*)"),
)
_CONTACT_HEADING_RE = {
    "purchaser": re.compile(r"采购人信息"),
    "agency": re.compile(r"(?:采购代理机构|代理机构)信息"),
}
_CONTACT_END_RE = re.compile(
    r"(?m)^[ \t]*\d+\s*[、.．]\s*(?:采购代理机构信息|代理机构信息|"
    r"采购人信息|政府采购监督部门信息|项目联系方式|其他信息)"
)
_CONTACT_VALUE_RE = re.compile(
    r"(?:采购人|招标人|代理机构|采购代理机构|名\s*称|名称)\s*[：:]\s*"
    r"(?P<value>[^\r\n]+)"
)
_CONTACT_TRAILING_RE = re.compile(
    r"\s+(?:联系人|联系电话|联系方式|电话|联系地址|地址|邮编|电子邮箱)\s*[：:].*$"
)
_NEXT_FIELD_RE = re.compile(
    r"\s+(?:采购执行编号|执行编号|政府采购计划编号|项目编号|项目号|项目名称|"
    r"预算金额|最高限价|采购人|招标人|代理机构|采购代理机构|提交投标文件截止时间)"
    r"\s*[：:]"
)
_BUDGET_LINE_RE = re.compile(
    r"(?m)^[ \t]*(?:\d+\s*[、.．]\s*)?"
    r"(?P<label>(?:(?:合同包|采购包|包组|标项)\s*)?"
    r"(?:(?:本项目|项目)\s*)?预算金额(?:\s*[（(]\s*元\s*[）)])?)"
    r"\s*[：:]\s*(?P<value>[^\r\n]*)"
)
_TABLE_BUDGET_HEADER_RE = re.compile(
    r"(?m)^[^\r\n]*(?:本包预算总金额|包预算总金额)\s*[（(]\s*元\s*[）)]"
)
_PACKAGE_MARKER_RE = re.compile(
    r"(?m)^[ \t]*(?:包组编号|标项名称|合同包(?:编号|名称)|"
    r"采购包(?:编号|名称)|包名称)\s*[：:]"
)
_BUDGET_TOKEN_RE = re.compile(
    r"(?<![\d第])(?:[￥¥]\s*)?\d[\d,，]*(?:\.\d+)?\s*"
    r"(?:[（(]\s*(?:亿元|万元|元)\s*[）)]|亿元|万元|元)?"
)
_EVIDENCE_BUDGET_UNIT_RE = re.compile(
    r"[^：:\r\n]{0,20}预算(?:总)?金额\s*[（(]\s*(?P<unit>亿元|万元|元)\s*[）)]"
    r"\s*[：:]\s*"
)
_EXPLICIT_DEADLINE_RE = re.compile(
    r"(?m)^[ \t]*(?:\d+\s*[、.．]\s*)?"
    r"(?:提交投标文件截止时间|投标文件递交截止时间|"
    r"投标文件提交截止时间|投标截止时间|投标文件截止时间)"
    r"\s*[：:]\s*(?P<value>[^\r\n]+)"
)
_DEADLINE_HEADING_RE = re.compile(
    r"(?m)^[ \t]*(?:[一二三四五六七八九十]+\s*[、.．]\s*)?"
    r"(?:提交投标文件截止时间|投标文件递交截止时间|"
    r"投标文件提交截止时间|投标文件截止时间|投标文件递交)"
)
_NEXT_SECTION_RE = re.compile(r"(?m)^[ \t]*[一二三四五六七八九十]+\s*[、.．]\s*\S")
_SECTION_DEADLINE_RE = re.compile(
    r"(?m)^[ \t]*(?:\d+\s*[、.．]\s*)?"
    r"(?:提交投标文件|投标文件(?:递交|提交)?)?截止时间\s*[：:]\s*(?P<value>[^\r\n]+)"
)
_GENERIC_TIME_RE = re.compile(r"(?m)^[ \t]*时间\s*[：:]\s*(?P<value>[^\r\n]+)")
_AMOUNT_VALUE_TYPES = {
    "project_name": ("string", "null"),
    "project_code": ("string", "null"),
    "purchaser": ("string", "null"),
    "agency": ("string", "null"),
    "budget_yuan": ("number", "null"),
    "bid_deadline": ("string", "null"),
}
TOOL = {
    "type": "function",
    "function": {
        "name": "submit_tender_fields",
        "description": "提交从招标公告正文中抽取的结构化字段及其证据。",
        "parameters": {
            "type": "object",
            "properties": {
                field: {
                    "type": "object",
                    "properties": {
                        "value": {"type": list(_AMOUNT_VALUE_TYPES[field])},
                        "evidence": (
                            {
                                "type": ["array", "null"],
                                "items": {"type": "string"},
                            }
                            if field == "budget_yuan"
                            else {"type": ["string", "null"]}
                        ),
                    },
                    "required": ["value", "evidence"],
                    "additionalProperties": False,
                }
                for field in FIELDS
            },
            "required": list(FIELDS),
            "additionalProperties": False,
        },
    },
}


def _normalize_value(field: str, value: Any) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if field == "project_code":
        return normalize_code(str(value))
    if field in {"project_name", "purchaser", "agency"}:
        return normalize_name(str(value))
    if field == "budget_yuan":
        if isinstance(value, (int, float)):
            return normalize_amount(str(value), default_unit="元")
        return normalize_amount(str(value))
    if field == "bid_deadline":
        return normalize_datetime(str(value))
    return value


def _field_value(field: str, value: str, evidence: str, source: str = "rule") -> FieldValue:
    value = value.strip()
    next_field = _NEXT_FIELD_RE.search(value)
    if next_field:
        value = value[: next_field.start()]
    trailing = _CONTACT_TRAILING_RE.search(value)
    if trailing:
        value = value[: trailing.start()]
    normalized = _normalize_value(field, value)
    return FieldValue(normalized, evidence.strip() or None, source)


def _first_line_value(pattern: re.Pattern, text: str, field: str) -> FieldValue | None:
    match = pattern.search(text)
    if not match:
        return None
    raw = match.group("value").strip()
    if not raw:
        return None
    return _field_value(field, raw, match.group(0))


def _contact_value(text: str, field: str) -> FieldValue | None:
    heading = _CONTACT_HEADING_RE[field].search(text)
    if heading:
        section = text[heading.end() :]
        end = _CONTACT_END_RE.search(section)
        if end:
            section = section[: end.start()]
        else:
            section = section[:3000]
        match = _CONTACT_VALUE_RE.search(section)
        if match:
            raw = match.group("value").strip()
            if raw:
                return _field_value(field, raw, match.group(0))

    labels = (
        ("采购人", "招标人", "名\\s*称", "名称")
        if field == "purchaser"
        else ("代理机构", "采购代理机构", "名\\s*称", "名称")
    )
    fallback = re.compile(
        r"(?m)^[ \t]*(?:" + "|".join(labels) + r")\s*[：:]\s*(?P<value>[^\r\n]+)"
    )
    return _first_line_value(fallback, text, field)


def _budget_values(value_text: str, label: str) -> list[float]:
    default_unit = "元" if re.search(r"[（(]\s*元\s*[）)]", label) else None
    tokens = [match.group(0).strip() for match in _BUDGET_TOKEN_RE.finditer(value_text)]
    values = [
        amount
        for token in tokens
        if (amount := normalize_amount(token, default_unit=default_unit)) is not None
    ]
    if values:
        return values
    amount = normalize_amount(value_text, default_unit=default_unit)
    return [amount] if amount is not None else []


def _evidence_amounts(quote: str) -> list[float]:
    budget_lines = list(_BUDGET_LINE_RE.finditer(quote))
    if budget_lines:
        amounts = [
            amount
            for match in budget_lines
            for amount in _budget_values(match.group("value"), match.group("label"))
        ]
        if amounts:
            return amounts

    unit_label = _EVIDENCE_BUDGET_UNIT_RE.search(quote)
    if unit_label:
        value_text = quote[unit_label.end() :]
        tokens = [match.group(0).strip() for match in _BUDGET_TOKEN_RE.finditer(value_text)]
        amounts = [
            amount
            for token in tokens
            if (amount := normalize_amount(token, default_unit=unit_label.group("unit")))
            is not None
        ]
        if amounts:
            return amounts
        amount = normalize_amount(value_text, default_unit=unit_label.group("unit"))
        if amount is not None:
            return [amount]

    amounts = _budget_values(quote, "")
    if amounts:
        return amounts
    tokens = [match.group(0).strip() for match in _BUDGET_TOKEN_RE.finditer(quote)]
    return [
        amount
        for token in tokens
        if (amount := normalize_amount(token, default_unit="元")) is not None
    ]


def _budget_value(text: str) -> FieldValue | None:
    matches = list(_BUDGET_LINE_RE.finditer(text))
    if not matches:
        table_header = _TABLE_BUDGET_HEADER_RE.search(text)
        if not table_header:
            return None
        section = text[table_header.end() :]
        next_section = _NEXT_SECTION_RE.search(section)
        if next_section:
            section = section[: next_section.start()]
        rows = [line.strip() for line in section.splitlines() if line.strip()]
        amounts = []
        evidence = None
        for row in rows:
            tokens = [
                match.group(0).strip()
                for match in _BUDGET_TOKEN_RE.finditer(row)
            ]
            if not tokens:
                continue
            amount = normalize_amount(tokens[-1], default_unit="元")
            if amount is not None:
                amounts.append(amount)
                evidence = evidence or row
        if not amounts:
            return None
        return FieldValue(round(sum(amounts), 2), evidence, "rule")
    first_package = _PACKAGE_MARKER_RE.search(text)
    package_start = first_package.start() if first_package else len(text)
    project_lines = [
        match
        for match in matches
        if match.start() < package_start
        and not re.match(r"(?:合同包|采购包|包组|标项)", match.group("label"))
    ]
    chosen = project_lines[:1] if project_lines else matches
    amounts = []
    for match in chosen:
        amounts.extend(_budget_values(match.group("value"), match.group("label")))
    if not amounts:
        return None
    evidence = chosen[0].group(0)
    return FieldValue(round(sum(amounts), 2), evidence.strip(), "rule")


def _deadline_value(text: str) -> FieldValue | None:
    explicit = _EXPLICIT_DEADLINE_RE.search(text)
    if explicit:
        normalized = normalize_datetime(explicit.group("value"))
        if normalized:
            return FieldValue(normalized, explicit.group(0).strip(), "rule")

    heading = _DEADLINE_HEADING_RE.search(text)
    if not heading:
        return None
    section = text[heading.end() :]
    next_section = _NEXT_SECTION_RE.search(section)
    if next_section:
        section = section[: next_section.start()]
    for pattern in (_SECTION_DEADLINE_RE, _GENERIC_TIME_RE):
        match = pattern.search(section)
        if match:
            normalized = normalize_datetime(match.group("value"))
            if normalized:
                return FieldValue(normalized, match.group(0).strip(), "rule")
    for line in section.splitlines():
        if "开始时间" in line or "获取招标" in line or "获取文件" in line:
            continue
        normalized = normalize_datetime(line)
        if normalized and len(normalized) > 5:
            return FieldValue(normalized, line.strip(), "rule")
    return None


def extract_rules(text: str) -> dict[str, FieldValue]:
    fields = {}
    project_name = _first_line_value(_PROJECT_NAME_RE, text, "project_name")
    if project_name:
        fields["project_name"] = project_name

    project_code = next(
        (
            value
            for pattern in _PROJECT_CODE_PATTERNS
            if (value := _first_line_value(pattern, text, "project_code"))
        ),
        None,
    )
    if project_code:
        fields["project_code"] = project_code

    for field in ("purchaser", "agency"):
        value = _contact_value(text, field)
        if value:
            fields[field] = value

    budget = _budget_value(text)
    if budget:
        fields["budget_yuan"] = budget

    deadline = _deadline_value(text)
    if deadline:
        fields["bid_deadline"] = deadline
    return fields


def _model_usage_totals(tenant) -> dict[str, int]:
    if tenant is None:
        return {}
    from django.db.models import Sum

    from personal_knowledge_base.models import ModelUsage

    totals = ModelUsage.objects.filter(
        tenant=tenant,
        deleted_at__isnull=True,
    ).aggregate(
        prompt_tokens=Sum("prompt_tokens"),
        completion_tokens=Sum("completion_tokens"),
        total_tokens=Sum("total_tokens"),
    )
    return {key: int(value or 0) for key, value in totals.items()}


def _usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        key: max(after.get(key, 0) - before.get(key, 0), 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _parse_content_json(content: str) -> dict:
    content = (content or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
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


def _response_fields(response: dict) -> dict:
    arguments = None
    for call in response.get("tool_calls") or []:
        function = call.get("function") or {}
        if function.get("name") == "submit_tender_fields":
            arguments = function.get("arguments")
            break
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = None
    if not isinstance(arguments, dict):
        arguments = _parse_content_json(response.get("content") or "")
    if "fields" in arguments and isinstance(arguments["fields"], dict):
        arguments = arguments["fields"]
    return arguments


def extract_llm(text: str, tenant) -> tuple[dict[str, FieldValue], dict[str, int]]:
    before = _model_usage_totals(tenant)
    response = model_providers.chat_completion_raw(
        tenant,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "公告正文：\n" + text},
        ],
        tools=[TOOL],
        temperature=0,
    )
    after = _model_usage_totals(tenant)
    usage = response.get("usage") or _usage_delta(before, after)
    fields = {}
    values = _response_fields(response)
    for field in FIELDS:
        entry = values.get(field)
        if not isinstance(entry, dict):
            entry = {}
        raw_value = entry.get("value")
        normalized = _normalize_value(field, raw_value)
        evidence = entry.get("evidence")
        if field == "budget_yuan":
            if isinstance(evidence, str):
                evidence = [evidence]
            elif not (
                isinstance(evidence, list)
                and all(isinstance(quote, str) for quote in evidence)
            ):
                evidence = None
        elif not isinstance(evidence, str):
            evidence = None
        fields[field] = FieldValue(normalized, evidence, "llm")
    return fields, {
        key: int(usage.get(key, 0) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def verify_evidence(
    fields: dict[str, FieldValue], text: str
) -> tuple[dict[str, FieldValue], list[str]]:
    whitespace_free_text = re.sub(r"\s+", "", text)
    kept = {}
    dropped = []
    for field, field_value in fields.items():
        if field_value.value is None:
            kept[field] = field_value
            continue
        evidence = field_value.evidence
        if field == "budget_yuan":
            quotes = [evidence] if isinstance(evidence, str) else evidence
            valid_quotes = isinstance(quotes, list) and all(
                isinstance(quote, str) for quote in quotes
            )
        else:
            quotes = [evidence] if isinstance(evidence, str) else []
            valid_quotes = bool(quotes)
        quotes = quotes if valid_quotes else []
        whitespace_free_quotes = [
            re.sub(r"\s+", "", quote) for quote in quotes
        ]
        valid = bool(whitespace_free_quotes) and all(
            quote and quote in whitespace_free_text for quote in whitespace_free_quotes
        )
        if valid and field in STRING_FIELDS:
            normalizer = normalize_code if field == "project_code" else normalize_name
            value = normalizer(str(field_value.value))
            normalized_evidence = normalizer(quotes[0])
            valid = bool(value) and value in normalized_evidence
            if valid and field == "project_code":
                bounded_value = re.compile(
                    rf"(?<![A-Za-z0-9\-_/]){re.escape(value)}(?![A-Za-z0-9\-_/])"
                )
                valid = bounded_value.search(normalized_evidence) is not None
        if valid and field == "bid_deadline":
            expected = normalize_datetime(str(field_value.value))
            evidence_datetimes = [
                normalized
                for quote in quotes
                for match in DATE_RE.finditer(_to_halfwidth(quote))
                if (normalized := normalize_datetime(match.group(0))) is not None
            ]
            valid = expected is not None and expected in evidence_datetimes
        if valid and field == "budget_yuan":
            expected = _normalize_value(field, field_value.value)
            quote_amounts = [_evidence_amounts(quote) for quote in quotes]
            amounts = [amount for values in quote_amounts for amount in values]
            valid = expected is not None and any(
                abs(expected - amount) < 0.01 for amount in amounts
            )
            if (
                not valid
                and expected is not None
                and len(quote_amounts) >= 2
                and all(quote_amounts)
            ):
                valid = abs(expected - sum(values[0] for values in quote_amounts)) < 0.01
        if valid:
            kept[field] = field_value
        else:
            kept[field] = FieldValue(None, field_value.evidence, field_value.source)
            dropped.append(field)
    return kept, dropped


def validate(fields: dict[str, Any]) -> list[str]:
    issues = []
    budget = fields.get("budget_yuan")
    try:
        valid_budget = budget is not None and 0 < float(budget) < 1e11
    except (TypeError, ValueError):
        valid_budget = False
    if not valid_budget:
        issues.append("budget_yuan: must be in (0, 100000000000)")

    deadline = fields.get("bid_deadline")
    try:
        parsed = datetime.strptime(str(deadline), "%Y-%m-%d %H:%M")
        valid_deadline = deadline is not None and 2020 <= parsed.year <= 2035
    except (TypeError, ValueError):
        valid_deadline = False
    if not valid_deadline:
        issues.append("bid_deadline: must parse to a date in 2020–2035")

    for field in ("purchaser", "project_name"):
        if not fields.get(field) or not str(fields[field]).strip():
            issues.append(f"{field}: must be non-empty")
    return issues


def _values_and_sources(fields: dict[str, FieldValue]) -> tuple[dict, dict]:
    values = {field: fields.get(field).value if fields.get(field) else None for field in FIELDS}
    sources = {field: fields[field].source for field in FIELDS if field in fields}
    return values, sources


def _result(
    selected: dict[str, FieldValue],
    usage: dict[str, int],
    evidence_dropped: list[str] | None = None,
    conflict_fields: list[str] | None = None,
) -> dict:
    values, sources = _values_and_sources(selected)
    evidence = {
        field: selected[field].evidence if field in selected else None
        for field in FIELDS
    }
    issues = validate(values)
    needs_review = set(conflict_fields or [])
    needs_review.update(issue.split(":", 1)[0] for issue in issues)
    needs_review.update(field for field, value in values.items() if value is None)
    return {
        "fields": values,
        "evidence": evidence,
        "sources": sources,
        "needs_review": [field for field in FIELDS if field in needs_review],
        "issues": issues,
        "evidence_dropped": evidence_dropped or [],
        "usage": usage,
    }


def build_extraction_result(
    method: str,
    rule_fields: dict[str, FieldValue] | None = None,
    llm_fields: dict[str, FieldValue] | None = None,
    usage: dict[str, int] | None = None,
    evidence_dropped: list[str] | None = None,
) -> dict:
    if method not in {"rules", "llm", "hybrid"}:
        raise ValueError("method must be one of: rules, llm, hybrid")
    rule_fields = rule_fields or {}
    llm_fields = llm_fields or {}
    usage = usage or {}
    evidence_dropped = evidence_dropped or []
    if method == "rules":
        selected = rule_fields
    elif method == "llm":
        selected = llm_fields
    else:
        selected = {}
        conflicts = []
        for field in FIELDS:
            rule_value = rule_fields.get(field)
            llm_value = llm_fields.get(field)
            if rule_value and rule_value.value is not None:
                selected[field] = rule_value
                if llm_value and llm_value.value is not None:
                    if _normalize_value(field, rule_value.value) != _normalize_value(
                        field, llm_value.value
                    ):
                        conflicts.append(field)
            elif llm_value:
                selected[field] = llm_value
        return _result(selected, usage, evidence_dropped, conflicts)

    return _result(selected, usage, evidence_dropped)


def extract(text: str, tenant, method: str) -> dict:
    if method not in {"rules", "llm", "hybrid"}:
        raise ValueError("method must be one of: rules, llm, hybrid")

    rule_fields = extract_rules(text) if method in {"rules", "hybrid"} else {}
    llm_fields = {}
    usage = {}
    dropped = []
    if method in {"llm", "hybrid"}:
        llm_fields, usage = extract_llm(text, tenant)
        llm_fields, dropped = verify_evidence(llm_fields, text)

    return build_extraction_result(
        method,
        rule_fields=rule_fields,
        llm_fields=llm_fields,
        usage=usage,
        evidence_dropped=dropped,
    )
