# Customer Service & Booking Agent

A multi-turn LangGraph agent that extracts booking details with an LLM and keeps
pricing, scheduling, and persistence deterministic.

## Setup

```bash
poetry install
cp .env.example .env
```

Set `OPENAI_API_KEY`. To enable traces, also set `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL`.

### Durable persistence (recommended for Streamlit)

Start Postgres, then set `DATABASE_URL` in `.env`:

```bash
docker compose up -d
```

```env
DATABASE_URL=postgresql://booking:booking@localhost:5432/booking_agent
```

With `DATABASE_URL` set, Streamlit and the CLI share:

- a **Postgres LangGraph checkpointer** for conversation state
- a **Postgres booking repository** with technician overlap protection

Without `DATABASE_URL`, both fall back to in-memory adapters (fine for unit tests
and quick demos; state is lost on restart).

Then run:

```bash
poetry run booking-agent
```

For the Streamlit product interface:

```bash
poetry run streamlit run customer_service_agent/streamlit_app.py
```

Each browser session gets an independent conversation `thread_id`. The graph,
checkpointer, and booking repository are shared for the Streamlit process so
confirmed bookings are visible to scheduling across sessions.

Run the test suite with:

```bash
poetry run pytest
```

Postgres integration tests run only when `DATABASE_URL` is set:

```bash
DATABASE_URL=postgresql://booking:booking@localhost:5432/booking_agent poetry run pytest
```

Poetry creates an isolated virtual environment and installs the application and
development dependencies from `pyproject.toml`. Commit the `poetry.lock` file
produced by `poetry lock` or `poetry install` for reproducible installations.

Every invocation uses a stable LangGraph `thread_id` and passes the LangFuse
callback through `config`, so node and model activity belongs to one trace.

## Flow

`gather_info → calculate_price → handle_offer → generate_schedule_options → select_slot → confirm_booking`

Rejected quotes and counteroffers route to `handoff`. The server recomputes the
LLM's completeness flag before allowing pricing, rechecks slot availability at
write time, and prevents overlapping technician bookings (in Postgres via an
exclusion constraint on technician + time range).
