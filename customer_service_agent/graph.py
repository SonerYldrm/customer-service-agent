"""LangGraph workflow for conversational booking."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from customer_service_agent.database import InMemoryBookingRepository
from customer_service_agent.engines import calculate_price, generate_schedule_options
from customer_service_agent.models import AgentState, BookingDetails


EXTRACTION_PROMPT = """You extract booking information for a cleaning company.
Merge facts from the full conversation with the current details. Never erase a known
value unless the customer explicitly corrects it. Normalize service_type to
house_cleaning or couch_cleaning and cleaning_depth to standard or deep.

size_info means square footage for house_cleaning and number of seats for
couch_cleaning. Required pricing fields are service_type, size_info,
cleaning_depth, and address. add_ons may be empty and must not block completion.

If every required field is present, set is_complete=true immediately—even when all
facts arrived in the first message—and set next_question=null. Do not request
anything redundant. Otherwise set is_complete=false and write one short, friendly
next_question asking only for the remaining required field(s), adapted to the
service type. Do not invent facts.

Current details:
{current_details}

Conversation:
{conversation}
"""


def _required_missing(details: BookingDetails) -> list[str]:
    fields = []
    if details.service_type is None:
        fields.append("service type (house or couch cleaning)")
    if details.size_info is None:
        if details.service_type == "house_cleaning":
            fields.append("home size in square feet")
        elif details.service_type == "couch_cleaning":
            fields.append("number of couch seats")
        else:
            fields.append("size")
    if details.cleaning_depth is None:
        fields.append("standard or deep cleaning")
    if not details.address or not details.address.strip():
        fields.append("service address")
    return fields


def _fallback_question(missing: list[str]) -> str:
    if len(missing) == 1:
        requested = missing[0]
    else:
        requested = ", ".join(missing[:-1]) + f", and {missing[-1]}"
    return f"To prepare your quote, could you share the {requested}?"


def _conversation(state: AgentState) -> str:
    lines = []
    for message in state.get("messages", []):
        role = "Customer" if isinstance(message, HumanMessage) else "Assistant"
        lines.append(f"{role}: {message.content}")
    return "\n".join(lines)


def create_nodes(llm: BaseChatModel, repository: InMemoryBookingRepository) -> dict[str, Callable[..., Any]]:
    extractor = llm.with_structured_output(BookingDetails)
    prompt = ChatPromptTemplate.from_template(EXTRACTION_PROMPT)

    def gather_info_node(state: AgentState) -> dict[str, Any]:
        current = state.get("booking_details", BookingDetails())
        extracted = extractor.invoke(
            prompt.invoke(
                {
                    "current_details": current.model_dump_json(exclude={"next_question"}),
                    "conversation": _conversation(state),
                }
            )
        )
        # The server, not the model, owns this control-flow invariant.
        missing = _required_missing(extracted)
        extracted.is_complete = not missing
        reply = None if extracted.is_complete else (extracted.next_question or _fallback_question(missing))
        update: dict[str, Any] = {
            "booking_details": extracted,
            "status": "gathering_info",
        }
        if reply:
            update["messages"] = [AIMessage(content=reply)]
        return update

    def calculate_price_node(state: AgentState) -> dict[str, Any]:
        return {"calculated_price": calculate_price(state["booking_details"])}

    def handle_offer_node(state: AgentState) -> dict[str, Any]:
        price = state.get("calculated_price")
        if price is None:
            raise ValueError("Price must be calculated before presenting an offer.")
        if state.get("status") != "awaiting_price_acceptance":
            return {
                "status": "awaiting_price_acceptance",
                "messages": [AIMessage(content=f"Your total is ${price:.2f}. Would you like to book it?")],
            }

        answer = _latest_customer_text(state).lower()
        accepted = bool(re.search(r"\b(yes|accept|accepted|agree|book|sounds good|go ahead)\b", answer))
        rejected = bool(re.search(r"\b(no|reject|decline|too (?:much|expensive)|not interested)\b", answer))
        counter = bool(re.search(r"(?:\$|usd|dollars?)\s*\d|\d+(?:\.\d+)?\s*(?:\$|usd|dollars?)", answer))
        if accepted and not rejected:
            return {"status": "awaiting_slot_selection"}
        if rejected or counter:
            return {"status": "rejected"}
        return {
            "status": "awaiting_price_acceptance",
            "messages": [AIMessage(content="Please let me know whether you accept the quoted price.")],
        }

    def generate_schedule_options_node(state: AgentState) -> dict[str, Any]:
        options = generate_schedule_options(state["booking_details"], repository)
        lines = ["Great—please choose one of these optimized appointments:"]
        for index, option in enumerate(options, 1):
            lines.append(f"{index}. {option.start_at} ({option.technician_id})")
        return {
            "time_options": options,
            "status": "awaiting_slot_selection",
            "messages": [AIMessage(content="\n".join(lines))],
        }

    def select_slot_node(state: AgentState) -> dict[str, Any]:
        answer = _latest_customer_text(state).strip()
        options = state.get("time_options", [])
        selected = next((option for option in options if option.id.lower() in answer.lower()), None)
        if selected is None:
            match = re.search(r"\b([1-3])\b", answer)
            selected = options[int(match.group(1)) - 1] if match and len(options) >= int(match.group(1)) else None
        if selected is None:
            return {
                "status": "awaiting_slot_selection",
                "messages": [AIMessage(content="Please choose option 1, 2, or 3.")],
            }
        return {"selected_slot": selected}

    def confirm_booking_node(state: AgentState) -> dict[str, Any]:
        option = state.get("selected_slot")
        if option is None:
            raise ValueError("A slot must be selected before confirmation.")
        booking = repository.create_booking(
            option, state["booking_details"], float(state["calculated_price"])
        )
        return {
            "booking_id": booking.id,
            "status": "confirmed",
            "messages": [
                AIMessage(
                    content=(
                        f"Confirmed! Booking {booking.id} is scheduled for "
                        f"{option.start_at}. Your total is ${booking.price:.2f}."
                    )
                )
            ],
        }

    def handoff_node(state: AgentState) -> dict[str, Any]:
        return {
            "status": "handoff",
            "messages": [
                AIMessage(
                    content="I understand. I’ll hand this to a specialist who can discuss changes or pricing."
                )
            ],
        }

    return {
        "gather_info": gather_info_node,
        "calculate_price": calculate_price_node,
        "handle_offer": handle_offer_node,
        "generate_schedule_options": generate_schedule_options_node,
        "select_slot": select_slot_node,
        "confirm_booking": confirm_booking_node,
        "handoff": handoff_node,
    }


def _latest_customer_text(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def route_entry(state: AgentState) -> Literal["gather_info", "handle_offer", "select_slot", "end"]:
    status = state.get("status", "gathering_info")
    if status == "awaiting_price_acceptance":
        return "handle_offer"
    if status == "awaiting_slot_selection" and state.get("time_options"):
        return "select_slot"
    if status in {"confirmed", "handoff"}:
        return "end"
    return "gather_info"


def should_continue_to_pricing(state: AgentState) -> Literal["calculate_price", "end"]:
    return "calculate_price" if state["booking_details"].is_complete else "end"


def should_continue_to_scheduling(
    state: AgentState,
) -> Literal["generate_schedule_options", "handoff", "end"]:
    if state["status"] == "awaiting_slot_selection":
        return "generate_schedule_options"
    if state["status"] == "rejected":
        return "handoff"
    return "end"


def route_after_slot(state: AgentState) -> Literal["confirm_booking", "end"]:
    return "confirm_booking" if state.get("selected_slot") else "end"


def build_graph(
    llm: BaseChatModel,
    *,
    repository: InMemoryBookingRepository | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Build a compiled, multi-turn booking graph."""
    repository = repository or InMemoryBookingRepository()
    graph = StateGraph(AgentState)
    for name, node in create_nodes(llm, repository).items():
        graph.add_node(name, node)

    graph.add_conditional_edges(START, route_entry, {"gather_info": "gather_info", "handle_offer": "handle_offer", "select_slot": "select_slot", "end": END})
    graph.add_conditional_edges(
        "gather_info",
        should_continue_to_pricing,
        {"calculate_price": "calculate_price", "end": END},
    )
    graph.add_edge("calculate_price", "handle_offer")
    graph.add_conditional_edges(
        "handle_offer",
        should_continue_to_scheduling,
        {
            "generate_schedule_options": "generate_schedule_options",
            "handoff": "handoff",
            "end": END,
        },
    )
    graph.add_edge("generate_schedule_options", END)
    graph.add_conditional_edges(
        "select_slot",
        route_after_slot,
        {"confirm_booking": "confirm_booking", "end": END},
    )
    graph.add_edge("confirm_booking", END)
    graph.add_edge("handoff", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())
