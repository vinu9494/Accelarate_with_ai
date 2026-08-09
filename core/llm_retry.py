"""Retry wrapper for LangGraph agent invocations against Anthropic.

Anthropic's API enforces rate limits per model tier. A 429 mid-run is handled
by retrying with the wait time from the Retry-After response header, or a
default backoff if no header is present.
"""

import time

import anthropic

DEFAULT_WAIT_SECONDS = 10.0
WAIT_BUFFER_SECONDS = 3.0


def invoke_with_retry(agent, messages: dict, max_retries: int = 6):
    """Call agent.invoke(messages), retrying on Anthropic RateLimitError (429).

    Reads the Retry-After header from the response if available; otherwise
    waits DEFAULT_WAIT_SECONDS plus a small buffer before retrying.
    """
    for attempt in range(max_retries):
        try:
            return agent.invoke(messages)
        except anthropic.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait_s = DEFAULT_WAIT_SECONDS
            try:
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    wait_s = float(retry_after) + WAIT_BUFFER_SECONDS
            except Exception:
                pass
            print(
                f"[LLM] Rate limited by Anthropic, waiting {wait_s:.1f}s "
                f"before retry {attempt + 1}/{max_retries}"
            )
            time.sleep(wait_s)
