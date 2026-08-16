"""Persistence factory and Postgres booking repository tests."""

from __future__ import annotations

import os
from datetime import datetime

import pytest

from customer_service_agent.database import InMemoryBookingRepository
from customer_service_agent.engines import ISTANBUL, calculate_price, generate_schedule_options
from customer_service_agent.models import BookingDetails
from customer_service_agent.persistence import create_persistence


def complete_house(**updates: object) -> BookingDetails:
    values = {
        "service_type": "house_cleaning",
        "size_info": 1000,
        "cleaning_depth": "standard",
        "add_ons": [],
        "address": "Kadikoy, Istanbul",
        "is_complete": True,
    }
    values.update(updates)
    return BookingDetails(**values)


def test_create_persistence_defaults_to_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    bundle = create_persistence()
    try:
        assert bundle.backend == "memory"
        assert isinstance(bundle.repository, InMemoryBookingRepository)
    finally:
        bundle.close()


@pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL is required for Postgres integration tests",
)
def test_postgres_repository_prevents_double_booking() -> None:
    bundle = create_persistence()
    try:
        assert bundle.backend == "postgres"
        repository = bundle.repository
        details = complete_house()
        option = generate_schedule_options(
            details, repository, now=datetime(2026, 7, 24, 12, tzinfo=ISTANBUL)
        )[0]
        first = repository.create_booking(option, details, calculate_price(details))
        assert first.id.startswith("BK-")
        with pytest.raises(ValueError, match="no longer available"):
            repository.create_booking(option, details, calculate_price(details))
        assert any(booking.id == first.id for booking in repository.list_bookings())
    finally:
        bundle.close()


@pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL is required for Postgres integration tests",
)
def test_postgres_checkpointer_setup_succeeds() -> None:
    bundle = create_persistence()
    try:
        assert bundle.backend == "postgres"
        assert bundle.checkpointer is not None
        # setup() is idempotent; calling again must not raise.
        bundle.checkpointer.setup()
    finally:
        bundle.close()
