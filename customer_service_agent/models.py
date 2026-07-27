"""State and domain schemas used by the booking graph."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from typing_extensions import NotRequired

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


ServiceType = Literal["house_cleaning", "couch_cleaning"]
CleaningDepth = Literal["standard", "deep"]
BookingStatus = Literal[
    "gathering_info",
    "awaiting_price_acceptance",
    "awaiting_slot_selection",
    "rejected",
    "handoff",
    "confirmed",
]


class BookingDetails(BaseModel):
    """Information extracted from the conversation.

    ``size_info`` is square footage for a house and seat count for a couch.
    Fields are optional because this model also represents partial extraction.
    """

    service_type: ServiceType | None = Field(default=None)
    size_info: float | None = Field(default=None, gt=0)
    cleaning_depth: CleaningDepth | None = Field(default=None)
    add_ons: list[str] = Field(default_factory=list)
    address: str | None = Field(default=None)
    is_complete: bool = False
    next_question: str | None = Field(
        default=None,
        exclude=True,
        description="A concise question asking only for information still missing.",
    )


class TimeOption(BaseModel):
    id: str
    technician_id: str
    start_at: str
    end_at: str
    estimated_travel_km: float


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    booking_details: BookingDetails
    calculated_price: NotRequired[float | None]
    time_options: NotRequired[list[TimeOption]]
    selected_slot: NotRequired[TimeOption | None]
    status: BookingStatus
    booking_id: NotRequired[str | None]
