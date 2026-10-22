"""Interpret provider retry guidance without depending on an SDK."""

from email.utils import parsedate_to_datetime


def retry_after_seconds(error: BaseException, now: float) -> float:
    if retry_at := getattr(error, "retry_at", 0):
        return max(0, retry_at - now)
    value = getattr(error, "retry_after", None)
    if value is None:
        value = next(
            (
                header
                for headers in (
                    getattr(error, "litellm_response_headers", None),
                    getattr(error, "headers", None),
                    getattr(getattr(error, "response", None), "headers", None),
                )
                if headers
                for name, header in headers.items()
                if str(name).casefold() == "retry-after"
            ),
            None,
        )
    if value is None:
        return 0
    try:
        return max(0, float(value))
    except (TypeError, ValueError):
        try:
            return max(0, parsedate_to_datetime(str(value)).timestamp() - now)
        except (TypeError, ValueError, OverflowError):
            return 0
