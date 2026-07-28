# Repository Guidelines

## Project Overview

This repository contains a Python customer-service and cleaning-booking agent
built with LangGraph, LangChain, and Pydantic. Langfuse provides optional
observability. The application currently uses a terminal interface, an in-memory
LangGraph checkpointer, and an in-memory booking repository.

## Development Environment

- Support Python 3.10 and newer, as declared in `pyproject.toml`.
- Use Poetry for dependency management. Do not add `requirements.txt`.
- Add runtime dependencies with `poetry add <package>` and development-only
  dependencies with `poetry add --group dev <package>`.
- Commit `poetry.lock` whenever dependency metadata changes.
- Run the application with `poetry run booking-agent`.

## Architecture

- Define shared graph and domain schemas in `customer_service_agent/models.py`.
- Define LangGraph nodes, edges, and routing in
  `customer_service_agent/graph.py`.
- Keep deterministic pricing and scheduling rules in
  `customer_service_agent/engines.py`.
- Keep persistence concerns behind the repository in
  `customer_service_agent/database.py`.
- Keep Langfuse setup isolated in `customer_service_agent/observability.py`.
- Keep CLI-specific behavior in `customer_service_agent/cli.py`.

## Implementation Conventions

- Add type hints to functions and public data structures.
- Use Pydantic models for validated domain data and `AgentState` for LangGraph
  state.
- Graph nodes should return partial state updates rather than mutating state.
- Preserve the `add_messages` reducer for conversation history.
- Keep prices, availability checks, scheduling, and booking persistence
  deterministic and server-controlled. Do not delegate business invariants or
  final authorization decisions to the LLM.
- When changing graph routing, explicitly map every conditional result,
  including `END`, and verify that the graph compiles.
- Preserve multi-turn behavior: use a stable `thread_id` and do not treat a
  customer reply to an offer or slot selection as new booking information.
- Keep external integrations optional and prevent observability failures from
  crashing the booking flow.

## Testing and Verification

- Run `poetry run pytest` after code changes.
- Add or update tests for pricing, scheduling, routing, and persistence changes.
- Tests should not require real OpenAI or Langfuse credentials unless an
  integration test is explicitly requested.
- For graph changes, verify graph compilation in addition to unit tests.
- Run `poetry check` after changing `pyproject.toml`.

## Security

- Never commit `.env`, API keys, access tokens, customer addresses, phone
  numbers, or other credentials and personal data.
- Keep `.env.example` limited to empty placeholders and safe example values.
- Do not log secrets or include them in test fixtures.
- Validate inbound data and verify webhook signatures when adding external
  messaging integrations.

## Documentation

- Update `README.md` when commands, environment variables, setup steps, or the
  public workflow change.
- Clearly distinguish current behavior from proposed production extensions such
  as PostgreSQL, WhatsApp, payments, or staff notifications.
