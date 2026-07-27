"""LangFuse callback configuration."""

from __future__ import annotations

import os
import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_langfuse_handler() -> Any | None:
    """Return a configured handler when credentials are available."""
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        return None
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def graph_config(thread_id: str, handler: Any | None = None) -> dict[str, Any]:
    """Config passed to every graph invocation, including checkpoint identity."""
    callbacks = [handler] if handler is not None else []
    return {
        "configurable": {"thread_id": thread_id},
        "callbacks": callbacks,
        "metadata": {"application": "customer-service-booking-agent"},
        "run_name": "customer-service-booking",
    }


def flush_langfuse(handler: Any | None) -> None:
    """Upload queued telemetry without making observability application-critical."""
    if handler is None:
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        logger.exception("Failed to flush LangFuse telemetry")
