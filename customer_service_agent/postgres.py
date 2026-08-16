"""PostgreSQL booking repository with overlap protection."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from psycopg.errors import ExclusionViolation
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from customer_service_agent.database import (
    DEFAULT_TECHNICIANS,
    Booking,
    Location,
    Technician,
    geocode,
)
from customer_service_agent.models import BookingDetails, TimeOption

BOOKING_SCHEMA_STATEMENTS = (
    "CREATE EXTENSION IF NOT EXISTS btree_gist",
    """
    CREATE TABLE IF NOT EXISTS technicians (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        home_lat DOUBLE PRECISION NOT NULL,
        home_lon DOUBLE PRECISION NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bookings (
        id TEXT PRIMARY KEY,
        technician_id TEXT NOT NULL REFERENCES technicians (id),
        start_at TIMESTAMPTZ NOT NULL,
        end_at TIMESTAMPTZ NOT NULL,
        location_lat DOUBLE PRECISION NOT NULL,
        location_lon DOUBLE PRECISION NOT NULL,
        address TEXT NOT NULL,
        details JSONB NOT NULL,
        price DOUBLE PRECISION NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT bookings_no_overlap EXCLUDE USING gist (
            technician_id WITH =,
            tstzrange(start_at, end_at, '[)') WITH &&
        )
    )
    """,
)


class PostgresBookingRepository:
    """Durable booking store shared across Streamlit sessions and workers."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool
        self._technicians: dict[str, Technician] = {}

    @property
    def technicians(self) -> dict[str, Technician]:
        return self._technicians

    def setup(self) -> None:
        """Create tables, seed default technicians, and refresh the cache."""
        with self._pool.connection() as conn:
            for statement in BOOKING_SCHEMA_STATEMENTS:
                conn.execute(statement)
            for technician in DEFAULT_TECHNICIANS.values():
                conn.execute(
                    """
                    INSERT INTO technicians (id, name, home_lat, home_lon)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        technician.id,
                        technician.name,
                        technician.home.lat,
                        technician.home.lon,
                    ),
                )
            rows = conn.execute(
                "SELECT id, name, home_lat, home_lon FROM technicians ORDER BY id"
            ).fetchall()
        self._technicians = {
            row["id"]: Technician(
                id=row["id"],
                name=row["name"],
                home=Location(row["home_lat"], row["home_lon"]),
            )
            for row in rows
        }

    def list_bookings(self) -> list[Booking]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, technician_id, start_at, end_at, location_lat, location_lon,
                       address, details, price
                FROM bookings
                ORDER BY start_at
                """
            ).fetchall()
        return [_row_to_booking(row) for row in rows]

    def create_booking(
        self, option: TimeOption, details: BookingDetails, price: float
    ) -> Booking:
        start = datetime.fromisoformat(option.start_at)
        end = datetime.fromisoformat(option.end_at)
        location = geocode(details.address or "")
        booking_id = f"BK-{uuid4().hex[:10].upper()}"
        payload = details.model_dump(mode="json")

        try:
            with self._pool.connection() as conn:
                with conn.transaction():
                    conn.execute(
                        """
                        INSERT INTO bookings (
                            id, technician_id, start_at, end_at,
                            location_lat, location_lon, address, details, price
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            booking_id,
                            option.technician_id,
                            start,
                            end,
                            location.lat,
                            location.lon,
                            details.address or "",
                            Jsonb(payload),
                            price,
                        ),
                    )
        except ExclusionViolation as exc:
            raise ValueError("The selected time slot is no longer available.") from exc

        return Booking(
            id=booking_id,
            technician_id=option.technician_id,
            start_at=start,
            end_at=end,
            location=location,
            address=details.address or "",
            details=details.model_copy(deep=True),
            price=price,
        )


def _row_to_booking(row: dict[str, Any]) -> Booking:
    return Booking(
        id=row["id"],
        technician_id=row["technician_id"],
        start_at=row["start_at"],
        end_at=row["end_at"],
        location=Location(row["location_lat"], row["location_lon"]),
        address=row["address"],
        details=BookingDetails.model_validate(row["details"]),
        price=float(row["price"]),
    )


def create_connection_pool(database_url: str) -> ConnectionPool:
    """Build a pool compatible with PostgresSaver and the booking repository."""
    return ConnectionPool(
        conninfo=database_url,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        open=True,
    )
