"""Shared persistence wiring for Streamlit and the CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from customer_service_agent.database import BookingRepository, InMemoryBookingRepository
from customer_service_agent.postgres import PostgresBookingRepository, create_connection_pool


@dataclass
class PersistenceBundle:
    """Runtime repository + checkpointer, optionally backed by one Postgres pool."""

    repository: BookingRepository
    checkpointer: Any
    backend: str
    pool: ConnectionPool | None = None

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()


def create_persistence(database_url: str | None = None) -> PersistenceBundle:
    """Create persistence adapters.

    When ``database_url`` (or ``DATABASE_URL``) is set, both the booking repository
    and LangGraph checkpointer use Postgres. Otherwise both stay in memory so tests
    and quick local runs need no database.
    """
    database_url = database_url if database_url is not None else os.getenv("DATABASE_URL")
    if not database_url:
        return PersistenceBundle(
            repository=InMemoryBookingRepository(),
            checkpointer=MemorySaver(),
            backend="memory",
        )

    pool = create_connection_pool(database_url)
    repository = PostgresBookingRepository(pool)
    repository.setup()
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    return PersistenceBundle(
        repository=repository,
        checkpointer=checkpointer,
        backend="postgres",
        pool=pool,
    )
