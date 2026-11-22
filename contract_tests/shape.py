import json
import os
from pathlib import Path


SNAPSHOTS = Path(__file__).parent / "snapshots"


def shape_of(value):
    if isinstance(value, dict):
        return {key: shape_of(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [shape_of(value[0])] if value else []
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "str"
    if value is None:
        return "null"
    raise TypeError(f"Unsupported contract value: {type(value).__name__}")


def assert_shape(testcase, name, body):
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOTS / f"{name}.json"
    actual = shape_of(body)
    if os.environ.get("CONTRACT_UPDATE") == "1":
        path.write_text(
            json.dumps(actual, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    try:
        expected = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        testcase.fail(f"Missing API contract snapshot: {path}")
    testcase.assertEqual(actual, expected, f"API shape differs from {path}")
