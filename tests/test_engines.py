from datetime import datetime

from customer_service_agent.database import InMemoryBookingRepository
from customer_service_agent.engines import ISTANBUL, calculate_price, generate_schedule_options
from customer_service_agent.models import BookingDetails


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


def test_price_is_deterministic_and_deep_cleaning_costs_more() -> None:
    standard = calculate_price(complete_house())
    deep = calculate_price(complete_house(cleaning_depth="deep"))
    assert standard == calculate_price(complete_house())
    assert deep > standard


def test_known_and_unknown_addons_have_explicit_prices() -> None:
    base = calculate_price(complete_house())
    quoted = calculate_price(complete_house(add_ons=["fridge", "custom request"]))
    assert quoted == base + 35.0


def test_scheduler_returns_three_unique_future_slots() -> None:
    now = datetime(2026, 7, 24, 12, tzinfo=ISTANBUL)
    options = generate_schedule_options(complete_house(), InMemoryBookingRepository(), now=now)
    assert len(options) == 3
    assert len({option.start_at for option in options}) == 3
    assert all(datetime.fromisoformat(option.start_at) > now for option in options)


def test_repository_prevents_double_booking() -> None:
    repository = InMemoryBookingRepository()
    details = complete_house()
    option = generate_schedule_options(
        details, repository, now=datetime(2026, 7, 24, 12, tzinfo=ISTANBUL)
    )[0]
    repository.create_booking(option, details, calculate_price(details))
    try:
        repository.create_booking(option, details, calculate_price(details))
    except ValueError as exc:
        assert "no longer available" in str(exc)
    else:
        raise AssertionError("Double booking must be rejected")

