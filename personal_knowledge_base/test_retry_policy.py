from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import Mock

import requests
from django.test import SimpleTestCase

from .model_providers import ModelAccessDeniedError, ModelHTTPError, _raise_for_model_status
from .retry_policy import RETRYABLE_STATUS, is_retryable, retry_delay


class RetryPolicyTests(SimpleTestCase):
    def test_retryable_model_statuses(self):
        for status_code in RETRYABLE_STATUS:
            with self.subTest(status_code=status_code):
                self.assertTrue(is_retryable(ModelHTTPError(status_code, None, "failed")))

    def test_retryable_transport_errors(self):
        self.assertTrue(is_retryable(requests.Timeout()))
        self.assertTrue(is_retryable(requests.ConnectionError()))

    def test_non_retryable_errors(self):
        for status_code in (400, 401, 404):
            with self.subTest(status_code=status_code):
                self.assertFalse(is_retryable(ModelHTTPError(status_code, None, "failed")))
        self.assertFalse(is_retryable(ModelAccessDeniedError(403, "denied", "forbidden")))
        self.assertFalse(is_retryable(ValueError("invalid response")))
        self.assertFalse(is_retryable(requests.HTTPError("bad request")))

    def test_retry_after_delay_is_honored_and_capped(self):
        self.assertEqual(retry_delay(ModelHTTPError(429, 7.0, "limited"), 2), 7.0)
        self.assertEqual(retry_delay(ModelHTTPError(429, 45.0, "limited"), 2), 30.0)

    def test_full_jitter_uses_exponential_attempt_window(self):
        error = requests.Timeout()
        self.assertEqual(retry_delay(error, 2, rand=lambda: 0), 0)
        delay = retry_delay(error, 2, rand=lambda: 0.999)
        self.assertAlmostEqual(delay, 3.996)
        self.assertLess(delay, 4.0)

    def test_retry_after_seconds_and_http_date_are_parsed(self):
        seconds = Mock(status_code=429, headers={"Retry-After": "7"}, text="busy")
        with self.assertRaises(ModelHTTPError) as raised:
            _raise_for_model_status(seconds)
        self.assertEqual(raised.exception.retry_after, 7.0)
        self.assertIsInstance(raised.exception, requests.HTTPError)

        target = datetime.now(timezone.utc) + timedelta(seconds=20)
        http_date = Mock(status_code=429, headers={"Retry-After": format_datetime(target, usegmt=True)}, text="busy")
        with self.assertRaises(ModelHTTPError) as raised:
            _raise_for_model_status(http_date)
        self.assertGreaterEqual(raised.exception.retry_after, 18.0)
        self.assertLessEqual(raised.exception.retry_after, 20.0)

    def test_retry_after_missing_or_invalid_is_none(self):
        for header in (None, "not-a-date"):
            with self.subTest(header=header):
                response = Mock(status_code=503, headers={"Retry-After": header}, text="unavailable")
                with self.assertRaises(ModelHTTPError) as raised:
                    _raise_for_model_status(response)
                self.assertIsNone(raised.exception.retry_after)

    def test_model_http_error_truncates_response_body(self):
        response = Mock(status_code=503, headers={}, text="x" * 500)
        with self.assertRaises(ModelHTTPError) as raised:
            _raise_for_model_status(response)
        self.assertEqual(len(str(raised.exception).split(": ", 1)[1]), 300)

    def test_model_http_error_preserves_access_denied_behavior(self):
        response = Mock(status_code=401, headers={})
        response.json.return_value = {"error": {"code": "invalid_key", "message": "Unauthorized"}}
        with self.assertRaises(ModelAccessDeniedError):
            _raise_for_model_status(response)
