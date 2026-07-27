"""Deterministic pricing and scheduling engines."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from customer_service_agent.database import (
    InMemoryBookingRepository,
    distance_km,
    geocode,
)
from customer_service_agent.models import BookingDetails, TimeOption

ISTANBUL = ZoneInfo("Europe/Istanbul")
ADD_ON_PRICES = {"fridge": 20.0, "oven": 25.0, "windows": 35.0, "pet_hair": 30.0}


def calculate_price(details: BookingDetails) -> float:
    """Calculate a quote in USD from documented, auditable rules."""
    if not details.is_complete or details.size_info is None or not details.address:
        raise ValueError("Complete booking details are required for pricing.")

    if details.service_type == "house_cleaning":
        subtotal = 55.0 + details.size_info * 0.08
    elif details.service_type == "couch_cleaning":
        subtotal = 45.0 + details.size_info * 18.0
    else:
        raise ValueError("Unsupported service type.")

    if details.cleaning_depth == "deep":
        subtotal *= 1.35
    subtotal += sum(ADD_ON_PRICES.get(item.lower().strip(), 15.0) for item in details.add_ons)

    depot = geocode("Taksim, Istanbul")
    travel_km = distance_km(depot, geocode(details.address))
    subtotal += max(0.0, travel_km - 8.0) * 1.5
    return round(subtotal, 2)


def generate_schedule_options(
    details: BookingDetails,
    repository: InMemoryBookingRepository,
    *,
    now: datetime | None = None,
) -> list[TimeOption]:
    """Rank free slots by travel distance and nearby back-to-back bookings."""
    if not details.address:
        raise ValueError("An address is required for scheduling.")
    now = (now or datetime.now(ISTANBUL)).astimezone(ISTANBUL)
    destination = geocode(details.address)
    existing = repository.list_bookings()
    duration = timedelta(hours=3 if details.cleaning_depth == "deep" else 2)
    candidates: list[tuple[float, TimeOption]] = []

    for day_offset in range(1, 8):
        day = (now + timedelta(days=day_offset)).date()
        if day.weekday() == 6:  # Sunday
            continue
        for technician in repository.technicians.values():
            tech_bookings = [b for b in existing if b.technician_id == technician.id]
            for hour in (9, 11, 13, 15):
                start = datetime.combine(day, time(hour), tzinfo=ISTANBUL)
                end = start + duration
                if end.hour > 18 or any(start < b.end_at and end > b.start_at for b in tech_bookings):
                    continue

                nearby_distance = distance_km(technician.home, destination)
                back_to_back_bonus = 0.0
                for booking in tech_bookings:
                    gap = min(abs((start - booking.end_at).total_seconds()), abs((booking.start_at - end).total_seconds()))
                    if gap <= 3600:
                        route_distance = distance_km(booking.location, destination)
                        if route_distance < nearby_distance:
                            nearby_distance = route_distance
                        if route_distance <= 5:
                            back_to_back_bonus = 10.0
                score = nearby_distance - back_to_back_bonus + day_offset * 0.25
                option = TimeOption(
                    id=f"{technician.id}-{start:%Y%m%d%H%M}",
                    technician_id=technician.id,
                    start_at=start.isoformat(),
                    end_at=end.isoformat(),
                    estimated_travel_km=round(nearby_distance, 1),
                )
                candidates.append((score, option))

    candidates.sort(key=lambda item: (item[0], item[1].start_at))
    selected: list[TimeOption] = []
    for _, option in candidates:
        if option.start_at not in {item.start_at for item in selected}:
            selected.append(option)
        if len(selected) == 3:
            break
    if len(selected) < 3:
        raise RuntimeError("Unable to find three available appointments.")
    return selected

