# Customer Service & Booking Agent

A multi-turn LangGraph agent that extracts booking details with an LLM and keeps
pricing, scheduling, and persistence deterministic.

## Setup

```bash
poetry install
cp .env.example .env
```

Set `OPENAI_API_KEY`. To enable traces, also set `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL`. Then run:

```bash
poetry run booking-agent
```

Run the test suite with:

```bash
poetry run pytest
```

Poetry creates an isolated virtual environment and installs the application and
development dependencies from `pyproject.toml`. Commit the `poetry.lock` file
produced by `poetry lock` or `poetry install` for reproducible installations.

Every invocation uses a stable LangGraph `thread_id` and passes the LangFuse
callback through `config`, so node and model activity belongs to one trace. The
in-memory checkpointer and repository are demonstration adapters; replace them
with persistent implementations for a multi-process deployment.

## Flow

`gather_info → calculate_price → handle_offer → generate_schedule_options → select_slot → confirm_booking`

Rejected quotes and counteroffers route to `handoff`. The server recomputes the
LLM's completeness flag before allowing pricing, rechecks slot availability at
write time, and prevents overlapping technician bookings.
