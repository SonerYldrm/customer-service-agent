"""Streamlit interface for the customer-service booking demo."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from customer_service_agent.graph import build_graph
from customer_service_agent.models import AgentState, BookingDetails, TimeOption
from customer_service_agent.observability import (
    create_langfuse_handler,
    flush_langfuse,
    graph_config,
)
from customer_service_agent.persistence import PersistenceBundle, create_persistence


INITIAL_STATE: AgentState = {
    "messages": [],
    "booking_details": BookingDetails(),
    "calculated_price": None,
    "time_options": [],
    "selected_slot": None,
    "status": "gathering_info",
}


@st.cache_resource
def _shared_runtime() -> tuple[Any, PersistenceBundle]:
    """One shared graph + Postgres/memory persistence for the Streamlit process."""
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    persistence = create_persistence()
    graph = build_graph(
        llm,
        repository=persistence.repository,
        checkpointer=persistence.checkpointer,
    )
    return graph, persistence


def _initialize_session() -> None:
    """Create per-browser thread state on top of the shared graph."""
    if "thread_id" in st.session_state:
        return

    graph, persistence = _shared_runtime()
    handler = create_langfuse_handler()
    st.session_state.graph = graph
    st.session_state.persistence = persistence
    st.session_state.handler = handler
    st.session_state.thread_id = str(uuid4())
    st.session_state.config = graph_config(st.session_state.thread_id, handler)
    st.session_state.agent_state = INITIAL_STATE.copy()
    st.session_state.started = False


def _invoke(customer_text: str) -> None:
    """Submit one customer turn to the graph and retain its latest state."""
    graph_input: dict[str, Any] = {"messages": [HumanMessage(content=customer_text)]}
    if not st.session_state.started:
        graph_input.update(INITIAL_STATE)
        graph_input["messages"] = [HumanMessage(content=customer_text)]
        st.session_state.started = True

    try:
        result = st.session_state.graph.invoke(graph_input, config=st.session_state.config)
        st.session_state.agent_state = result
        flush_langfuse(st.session_state.handler)
    except Exception:
        st.session_state.started = bool(st.session_state.agent_state.get("messages"))
        st.error("The assistant could not process that request. Please try again.")


def _format_slot(option: TimeOption) -> str:
    start = datetime.fromisoformat(option.start_at)
    end = datetime.fromisoformat(option.end_at)
    return f"{start:%a, %d %b · %H:%M}–{end:%H:%M}"


def _render_summary(state: AgentState) -> None:
    details = state.get("booking_details", BookingDetails())
    backend = st.session_state.persistence.backend
    with st.sidebar:
        st.header("Booking summary")
        st.write("Service", (details.service_type or "—").replace("_", " ").title())
        size_label = "Home size" if details.service_type == "house_cleaning" else "Seats"
        size_value = f"{details.size_info:g}" if details.size_info is not None else "—"
        st.write(size_label, size_value)
        st.write("Cleaning", (details.cleaning_depth or "—").title())
        st.write("Address", details.address or "—")
        if state.get("calculated_price") is not None:
            st.metric("Quote", f"${state['calculated_price']:.2f}")
        st.divider()
        if backend == "postgres":
            st.caption("Postgres · conversation checkpoints and bookings persist across restarts.")
        else:
            st.caption(
                "In-memory · set DATABASE_URL for durable checkpoints and bookings."
            )
        if st.button("Start over", use_container_width=True):
            handler = st.session_state.get("handler")
            flush_langfuse(handler)
            for key in ("thread_id", "config", "agent_state", "started", "handler"):
                st.session_state.pop(key, None)
            st.rerun()


def _render_messages(state: AgentState) -> None:
    if not state.get("messages"):
        with st.chat_message("assistant"):
            st.write(
                "Hi! I can help you book house or couch cleaning. "
                "Tell me what you need, including the size and service address."
            )
        return

    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue
        with st.chat_message(role):
            st.write(str(message.content))


def _render_actions(state: AgentState) -> None:
    status = state.get("status")
    if status == "awaiting_price_acceptance":
        accept_col, decline_col = st.columns(2)
        if accept_col.button("Accept quote", type="primary", use_container_width=True):
            _invoke("I accept the quoted price.")
            st.rerun()
        if decline_col.button("Decline", use_container_width=True):
            _invoke("No, I decline the quoted price.")
            st.rerun()

    if status == "awaiting_slot_selection" and state.get("time_options"):
        st.subheader("Choose an appointment")
        columns = st.columns(len(state["time_options"]))
        for column, option in zip(columns, state["time_options"]):
            with column:
                st.markdown(f"**{_format_slot(option)}**")
                st.caption(f"Technician: {option.technician_id}")
                if st.button("Select", key=f"slot-{option.id}", use_container_width=True):
                    _invoke(f"I choose slot {option.id}.")
                    st.rerun()

    if status == "confirmed":
        st.success(f"Booking confirmed · Reference {state.get('booking_id', '—')}")
    elif status == "handoff":
        st.info("This request has been marked for specialist follow-up.")


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Cleaning Booking Assistant", page_icon="🧹")
    st.title("Cleaning Booking Assistant")
    st.caption("Get a quote and choose an appointment in a few messages.")

    if not os.getenv("OPENAI_API_KEY"):
        st.error(
            "OPENAI_API_KEY is not configured. Add it to .env locally or to "
            "your Streamlit app secrets."
        )
        st.stop()

    try:
        _initialize_session()
    except Exception as exc:
        st.error(
            "Could not initialize persistence. Check DATABASE_URL and that Postgres "
            f"is reachable.\n\n{exc}"
        )
        st.stop()

    state: AgentState = st.session_state.agent_state
    _render_summary(state)
    _render_messages(state)
    _render_actions(state)

    terminal = state.get("status") in {"confirmed", "handoff"}
    if prompt := st.chat_input("Describe your cleaning needs…", disabled=terminal):
        _invoke(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
