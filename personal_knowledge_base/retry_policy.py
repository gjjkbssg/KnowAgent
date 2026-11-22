import random

import requests

from .model_providers import ModelHTTPError


RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def is_retryable(exc: Exception) -> bool:
    if isinstance(exc, ModelHTTPError):
        return exc.status_code in RETRYABLE_STATUS
    return isinstance(exc, (requests.Timeout, requests.ConnectionError))


def retry_delay(
    exc: Exception,
    attempt: int,
    base: float = 1.0,
    cap: float = 30.0,
    rand=random.random,
) -> float:
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        return max(0.0, min(float(retry_after), cap))
    return rand() * min(cap, base * 2**attempt)
