import json
import os
from pathlib import Path


SNAPSHOTS = Path(__file__).parent / "snapshots"


def capture(testcase, label, response, expected_status=None):
    if expected_status is not None:
        testcase.assertEqual(
            response.status_code,
            expected_status,
            f"{label} returned {response.status_code}: {response.text[:500]}",
        )
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        body = {
            "content_type": response.headers.get("content-type", ""),
            "text": response.text,
        }
    if response.status_code < 400:
        testcase.assertIsInstance(body, dict, label)
        testcase.assertIn("success", body, label)
        testcase.assertIs(body["success"], True, label)
    elif isinstance(body, dict) and "success" in body:
        testcase.assertIs(body["success"], False, label)
        testcase.assertIsInstance(body.get("error"), dict, label)
        testcase.assertIn("code", body["error"], label)
        testcase.assertIn("message", body["error"], label)
    return {"status": response.status_code, "body": body}


def assert_snapshot(testcase, name, responses):
    from contract_tests.shape import assert_shape

    assert_shape(testcase, name, responses)


def assert_value_snapshot(testcase, name, value):
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOTS / f"{name}.json"
    if os.environ.get("CONTRACT_UPDATE") == "1":
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    try:
        expected = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        testcase.fail(f"Missing API contract snapshot: {path}")
    testcase.assertEqual(value, expected, f"API values differ from {path}")
