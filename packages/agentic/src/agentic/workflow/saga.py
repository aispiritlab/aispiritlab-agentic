"""Workflow / Saga pattern for long-running processes spanning multiple aggregates.

A Saga coordinates a multi-step process by reacting to events and emitting
commands. It follows the same decide/evolve/initial_state shape as a Decider,
but operates on cross-aggregate coordination rather than single-aggregate logic.

Each step in the saga can:
- Emit commands to other aggregates
- Emit events to record saga progress
- Fail and trigger compensating actions
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import reduce
from typing import Callable, Generic, Sequence, TypeVar

from agentic.workflow.messages import Event, UserCommand


I = TypeVar("I")  # Input (events from other aggregates)
S = TypeVar("S")  # Saga state
O = TypeVar("O")  # Output (commands/events to emit)


class SagaAction(StrEnum):
    INITIATED_BY = "initiated_by"
    RECEIVED = "received"
    SENT = "sent"
    PUBLISHED = "published"
    COMPENSATED = "compensated"


@dataclass(frozen=True, slots=True)
class SagaStep(Generic[O]):
    """A single output from a saga decide function."""

    action: SagaAction
    message: O


@dataclass(frozen=True, slots=True)
class Saga(Generic[I, S, O]):
    """Long-running process coordinator: decide + evolve + initial_state.

    - decide(input, state) -> steps: what commands/events to emit
    - evolve(state, input) -> state: track saga progress
    - initial_state() -> state: factory for new saga
    """

    decide: Callable[[I, S], Sequence[SagaStep[O]]]
    evolve: Callable[[S, I], S]
    initial_state: Callable[[], S]


@dataclass(frozen=True, slots=True)
class SagaResult(Generic[S, O]):
    steps: tuple[SagaStep[O], ...]
    new_state: S


def run_saga_step(
    saga: Saga[I, S, O],
    state: S,
    input_event: I,
) -> SagaResult[S, O]:
    """Execute a single saga step: evolve state, then decide on outputs."""
    new_state = saga.evolve(state, input_event)
    steps = tuple(saga.decide(input_event, new_state))
    return SagaResult(steps=steps, new_state=new_state)


def replay_saga(
    saga: Saga[I, S, O],
    events: Sequence[I],
) -> S:
    """Reconstruct saga state from event history."""
    return reduce(saga.evolve, events, saga.initial_state())
