import re
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


DIGITS = {
    "零": 0,
    "〇": 0,
    "壹": 1,
    "一": 1,
    "贰": 2,
    "二": 2,
    "两": 2,
    "叁": 3,
    "三": 3,
    "肆": 4,
    "四": 4,
    "伍": 5,
    "五": 5,
    "陆": 6,
    "六": 6,
    "柒": 7,
    "七": 7,
    "捌": 8,
    "八": 8,
    "玖": 9,
    "九": 9,
}
UNITS = {"拾": 10, "十": 10, "佰": 100, "百": 100, "仟": 1000, "千": 1000}
BIG = {"万": 10**4, "亿": 10**8}
AMOUNT_RE = re.compile(
    r"(?P<num>\d[\d,，]*(?:\.\d+)?)\s*(?:[（(]\s*(?P<paren_unit>亿元|万元|元)\s*[）)]|(?P<unit>亿元|万元|元))"
)
BARE_AMOUNT_RE = re.compile(r"(?P<num>\d[\d,，]*(?:\.\d+)?)")
DATE_RE = re.compile(
    r"(?P<year>\d{4})\s*[年\-/.]\s*(?P<month>\d{1,2})\s*[月\-/.]\s*"
    r"(?P<day>\d{1,2})\s*日?\s*"
    r"(?:(?P<ampm>上午|下午)?\s*(?P<hour>\d{1,2})\s*(?:[:：]|时|点)\s*"
    r"(?P<minute>\d{1,2})?\s*分?(?:\s*\d{1,2}\s*秒)?)?"
)
TIME_ONLY_RE = re.compile(r"(?<!\d)(?P<hour>\d{1,2})\s*点\s*(?P<minute>\d{1,2})\s*分")
CHINESE_AMOUNT_RE = re.compile(r"[零〇壹一贰二两叁三肆四伍五陆六柒七捌八玖九拾十佰百仟千萬万亿]+[元圆]")
CHINESE_FRACTION_RE = re.compile(r"[零〇壹一贰二两叁三肆四伍五陆六柒七捌八玖九角分整正]*")
UNIT_MULTIPLIERS = {"元": Decimal(1), "万元": Decimal(10_000), "亿元": Decimal(100_000_000)}


def _to_halfwidth(text: str) -> str:
    converted = "".join(
        chr(ord(char) - 0xFEE0) if 0xFF01 <= ord(char) <= 0xFF5E else char
        for char in text
    )
    return converted.replace("\u3000", " ")


def chinese_amount_to_yuan(text: str) -> float:
    """Convert financial Chinese numerals such as 壹佰贰拾万零伍佰元整 to yuan."""
    text = text.replace("圆", "元").replace("整", "").replace("正", "")
    yuan_part, _, fraction = text.partition("元")
    total = section = number = 0
    for char in yuan_part:
        if char in DIGITS:
            number = DIGITS[char]
        elif char in UNITS:
            section += (number or 1) * UNITS[char]
            number = 0
        elif char in BIG:
            total = (
                (total + section + number) * BIG[char]
                if BIG[char] > 10**4
                else total + (section + number) * BIG[char]
            )
            section = number = 0
    total += section + number
    jiao = re.search(r"([零〇壹一贰二两叁三肆四伍五陆六柒七捌八玖九])角", fraction)
    fen = re.search(r"([零〇壹一贰二两叁三肆四伍五陆六柒七捌玖九])分", fraction)
    amount = (
        Decimal(total)
        + (Decimal(DIGITS[jiao.group(1)]) / 10 if jiao else Decimal(0))
        + (Decimal(DIGITS[fen.group(1)]) / 100 if fen else Decimal(0))
    )
    return float(amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def normalize_amount(text: str, default_unit: str | None = None) -> float | None:
    """Normalize a tender amount to yuan, rounding to two decimal places."""
    text = _to_halfwidth(str(text))
    match = AMOUNT_RE.search(text)
    if match:
        unit = match.group("paren_unit") or match.group("unit")
        number = match.group("num")
        multiplier = UNIT_MULTIPLIERS[unit]
    elif default_unit in UNIT_MULTIPLIERS:
        match = BARE_AMOUNT_RE.search(text)
        if not match:
            match = None
            number = ""
        else:
            number = match.group("num")
        if not number:
            chinese_match = CHINESE_AMOUNT_RE.search(text)
            if not chinese_match:
                return None
            return chinese_amount_to_yuan(chinese_match.group() + _chinese_fraction_suffix(text, chinese_match.end()))
        multiplier = UNIT_MULTIPLIERS[default_unit]
    else:
        chinese_match = CHINESE_AMOUNT_RE.search(text)
        if not chinese_match:
            return None
        return chinese_amount_to_yuan(chinese_match.group() + _chinese_fraction_suffix(text, chinese_match.end()))

    try:
        amount = Decimal(number.replace(",", "").replace("，", "")) * multiplier
    except InvalidOperation:
        return None
    return float(amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _chinese_fraction_suffix(text: str, start: int) -> str:
    match = CHINESE_FRACTION_RE.match(text, start)
    return match.group() if match else ""


def normalize_datetime(text: str) -> str | None:
    """Normalize Chinese or numeric tender deadlines to YYYY-MM-DD HH:MM."""
    text = _to_halfwidth(str(text))
    match = DATE_RE.search(text)
    if not match:
        time_match = TIME_ONLY_RE.search(text)
        if not time_match:
            return None
        return f"{int(time_match.group('hour')):02d}:{int(time_match.group('minute')):02d}"

    year = int(match.group("year"))
    month = int(match.group("month"))
    day = int(match.group("day"))
    hour = int(match.group("hour") or 0)
    minute = int(match.group("minute") or 0)
    if match.group("ampm") == "下午" and hour < 12:
        hour += 12
    try:
        return datetime(year, month, day, hour, minute).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return None


def normalize_company(name: str) -> str:
    """Normalize widths, brackets, and whitespace in organization names."""
    name = _to_halfwidth(str(name)).replace("(", "（").replace(")", "）")
    return re.sub(r"\s+", "", name)


def normalize_name(name: str) -> str:
    return normalize_company(name)


def normalize_code(value: str) -> str:
    return re.sub(r"\s+", "", _to_halfwidth(str(value)))


if __name__ == "__main__":
    assert chinese_amount_to_yuan("壹佰贰拾万零伍佰元整") == 1200500
    assert chinese_amount_to_yuan("叁亿肆仟万元") == 340000000
    assert chinese_amount_to_yuan("伍万陆仟柒佰捌拾玖元伍角陆分") == 56789.56
    assert normalize_amount("预算金额：1,234.5万元") == 12345000
    assert normalize_amount("最高限价：人民币壹佰万元整") == 1000000
    assert normalize_datetime("投标截止时间：2026年3月5日下午2时30分") == "2026-03-05 14:30"
    assert normalize_datetime("开标时间 2026-03-05 09:30") == "2026-03-05 09:30"
    assert normalize_company("安徽省招标集团股份有限公司 (合肥)") == "安徽省招标集团股份有限公司（合肥）"
    assert normalize_company("ＡＢＣ科技有限公司") == "ABC科技有限公司"
    print("ok")
