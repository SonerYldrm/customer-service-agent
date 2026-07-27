"""AI customer-service and booking agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from customer_service_agent.models import AgentState, BookingDetails

__all__ = ["AgentState", "BookingDetails", "build_graph"]


def __getattr__(name: str) -> Any:
    """Load public objects lazily so importing the package has no side effects."""
    if name == "build_graph":
        from customer_service_agent.graph import build_graph

        return build_graph
    if name in {"AgentState", "BookingDetails"}:
        from customer_service_agent.models import AgentState, BookingDetails

        return {"AgentState": AgentState, "BookingDetails": BookingDetails}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
