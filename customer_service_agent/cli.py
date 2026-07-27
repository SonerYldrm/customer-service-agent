"""Console entry point for the customer-service booking agent."""

from __future__ import annotations

import os
import warnings
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change.*",
    category=LangChainPendingDeprecationWarning,
)

from customer_service_agent.graph import build_graph
from customer_service_agent.models import BookingDetails
from customer_service_agent.observability import (
    create_langfuse_handler,
    flush_langfuse,
    graph_config,
)


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required. Copy .env.example to .env.")

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    graph = build_graph(llm)
    langfuse_handler = create_langfuse_handler()
    config = graph_config(str(uuid4()), langfuse_handler)

    print("Booking assistant ready. Type 'quit' to exit.")
    if langfuse_handler is not None:
        print("LangFuse tracing enabled.")
    first_turn = True
    try:
        while True:
            user_text = input("You: ").strip()
            if user_text.lower() in {"quit", "exit"}:
                break
            if not user_text:
                continue

            if first_turn:
                graph_input = {
                    "messages": [HumanMessage(content=user_text)],
                    "booking_details": BookingDetails(),
                    "calculated_price": None,
                    "time_options": [],
                    "selected_slot": None,
                    "status": "gathering_info",
                }
                first_turn = False
            else:
                graph_input = {"messages": [HumanMessage(content=user_text)]}

            result = graph.invoke(graph_input, config=config)
            flush_langfuse(langfuse_handler)
            print(f"Assistant: {result['messages'][-1].content}")
            if result["status"] in {"confirmed", "handoff"}:
                break
    finally:
        flush_langfuse(langfuse_handler)
