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

For the Streamlit demo interface, run:

```bash
poetry run streamlit run customer_service_agent/streamlit_app.py
```

The web demo uses the same environment variables and booking graph as the CLI.
Each browser session gets an independent conversation thread. Bookings and graph
state remain in memory and can be lost whenever the app restarts.

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
