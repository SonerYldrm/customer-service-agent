"""Thread-safe in-memory repository and deterministic location helpers."""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from customer_service_agent.models import BookingDetails, TimeOption


@dataclass(frozen=True)
class Location:
    lat: float
    lon: float


@dataclass(frozen=True)
class Technician:
    id: str
    name: str
    home: Location


@dataclass(frozen=True)
class Booking:
    id: str
    technician_id: str
    start_at: datetime
    end_at: datetime
    location: Location
    address: str
    details: BookingDetails
    price: float


def geocode(address: str) -> Location:
    """Stable mock geocoder centered around Istanbul (replace in production)."""
    digest = hashlib.sha256(address.strip().lower().encode("utf-8")).digest()
    lat_offset = (int.from_bytes(digest[:2], "big") / 65535 - 0.5) * 0.36
    lon_offset = (int.from_bytes(digest[2:4], "big") / 65535 - 0.5) * 0.48
    return Location(41.0082 + lat_offset, 28.9784 + lon_offset)


def distance_km(a: Location, b: Location) -> float:
    """Haversine distance."""
    radius = 6371.0
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    dlat = lat2 - lat1
    dlon = math.radians(b.lon - a.lon)
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return radius * 2 * math.asin(math.sqrt(h))


class InMemoryBookingRepository:
    """Minimal repository interface suitable for swapping with a real DB."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.technicians = {
            "tech-1": Technician("tech-1", "Aylin", Location(41.015, 28.979)),
            "tech-2": Technician("tech-2", "Mehmet", Location(41.041, 29.009)),
            "tech-3": Technician("tech-3", "Deniz", Location(40.991, 29.027)),
        }
        self._bookings: list[Booking] = []

    def list_bookings(self) -> list[Booking]:
        with self._lock:
            return list(self._bookings)

    def create_booking(
        self, option: TimeOption, details: BookingDetails, price: float
    ) -> Booking:
        start = datetime.fromisoformat(option.start_at)
        end = datetime.fromisoformat(option.end_at)
        with self._lock:
            if any(
                b.technician_id == option.technician_id
                and start < b.end_at
                and end > b.start_at
                for b in self._bookings
            ):
                raise ValueError("The selected time slot is no longer available.")
            booking = Booking(
                id=f"BK-{uuid4().hex[:10].upper()}",
                technician_id=option.technician_id,
                start_at=start,
                end_at=end,
                location=geocode(details.address or ""),
                address=details.address or "",
                details=details.model_copy(deep=True),
                price=price,
            )
            self._bookings.append(booking)
            return booking

