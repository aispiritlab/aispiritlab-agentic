"""Where a fan-in waits, and how it survives the process that started it.

A graph that fans out to three searchers and joins on a summarizer has to
remember three things between the first completion and the last: how many it is
waiting for, which have arrived, and whether the summarizer has already been
fired. Held in dictionaries on one object, all three die with the process — and
a worker restarting between the second and third completion loses the join with
nothing to say it happened.

## Two implementations, one protocol

`MemoryJoinLedger` is those dictionaries, and it stays the default: a preview
that runs once in one process needs nothing else, and paying for durability
there would be paying for a restart that cannot happen.

`StreamJoinLedger` folds the same state out of an event stream and appends to
it. Which stream is the caller's business — `agentic.workflow.InMemoryEventStore`
for a test, `AiwatcherEventStore` for a worker whose history has to outlive it.

## A turn is the unit, and the stream name says so

The key is `graph:<graph_id>:<turn_id>`. A turn is a traversal: the join it
opens is answered inside it and is never read again, so a stream that lived as
long as the graph made every completion fold every turn that came before it —
quadratic inside a turn, and unbounded across a graph's life.

A store whose *execution* is already one turn has nothing to scope, and
:meth:`StreamJoinLedger.on_one_stream` is that caller — named for the reason
rather than switched on by a flag.

## A deadline is what turns silence into a decision

A node that never completes leaves the fan-in waiting for ever, and nothing in
the stream distinguishes that from one still thinking. So a deadline is written
down when the fan-in is reserved, and :class:`OnDeadline` is what happens when
it arrives. Scheduling one is opt-in: it needs something that wakes up and
looks, and an in-process preview has nothing that does.

The deadline fires through `claim_summary` like everything else, which is what
keeps it from answering beside a completion that landed first.

## Why claiming is a write and not a check

`claim_summary` is the one operation that cannot be a read followed by a
decision. Two workers that both see the last completion arrive would both find
no claim and both fire the summarizer, and a graph that answered twice is worse
than one that answered late. On a stream it is a compare-and-append: the loser's
append is refused, it re-reads, and it finds the claim that beat it.

## Why a claim can be taken over, and why that is not the same relaxation

The claim is taken *before* the summarizer runs, so a worker that dies holding
one used to leave a join that never fired and said nothing. A claim carries its
holder and the instant it was taken, and one older than the lease may be taken
over — by anybody, including the process that took it, because "expired" means
the summarizer did not finish and firing again is the recovery.

What is never taken over is a claim that *completed*: `graph.summary_completed`
is appended when the summarizer returns, so a finished join is final however
old it gets, and a live one is not mistaken for a dead one.

The trade is configurable rather than hidden. A summarizer slower than the
lease is one a second worker may fire beside, and there are two dials for it:
raise `lease_seconds`, or renew. `renew_claim` refreshes the instant while the
summarizer runs and is refused the moment the claim stopped being the caller's,
so a worker that was taken over finds out rather than assuming.

Renewing is opt-in because it costs a heartbeat, and being taken over is
sometimes an acceptable outcome — a search fan-out re-run beside itself is a
wasted call, not a wrong answer. What is *not* optional is the check after the
fact: whoever fires re-reads the claim when the summarizer returns, and an
answer whose claim is no longer the caller's is discarded rather than written
beside the one that replaced it. That is what makes "answer once" hold even
with no heartbeat at all.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import os
import socket
import time
from typing import Any, Protocol

__all__ = [
    "AnswerDiscarded",
    "Completion",
    "JoinLedger",
    "JoinTimedOut",
    "MemoryJoinLedger",
    "OnDeadline",
    "StreamJoinLedger",
    "this_process",
]

#: The event types this ledger folds. Namespaced, because the stream they live
#: in may be shared with a decider's own messages.
RESERVED = "graph.fan_in_reserved"
RECORDED = "graph.completion_recorded"
CLAIMED = "graph.summary_claimed"
COMPLETED = "graph.summary_completed"
DEADLINE = "graph.join_deadline_scheduled"

#: How many times a claim re-reads and tries again.
#:
#: Contention here is a fan-in's own completions landing between the read and
#: the write, which resolves in a pass or two — the number is small because a
#: fourth conflict is a busy stream rather than a race, and looping on that
#: hides it.
_CLAIM_ATTEMPTS = 3

#: How long a claim is somebody's before anybody else may take it.
#:
#: Five minutes rather than five seconds, because the failure a short lease
#: produces is the one this module exists to prevent: a summarizer that is
#: merely slow, taken over and fired beside itself. A worker that actually died
#: costs a turn five minutes, and a turn that answered twice costs an answer
#: nobody can tell from the other afterwards.
_LEASE_SECONDS = 300.0


def this_process() -> str:
    """Who is holding a claim, derived rather than generated.

    A host and a pid: it survives being asked twice, it means something in a log
    line, and two ledgers in one process are deliberately the same holder —
    whether a claim may be taken is decided by its age, never by who is asking.
    """
    return f"{socket.gethostname()}:{os.getpid()}"


class OnDeadline(StrEnum):
    """What a join does when its deadline arrives.

    A product decision rather than a technical one, and the graph cannot tell
    which it is without being told: a partial answer is useful for a search
    fan-out and unacceptable where somebody is counting on a complete set. Both
    are reachable per graph, because one instance runs both kinds.

    `SUMMARIZE` is the default because the failure it produces is visible — an
    answer recorded as partial, naming what is missing — while `FAIL` produces
    a turn with no answer at all, which is the right outcome only where
    somebody decided it is.

    Neither summarizes *nothing*. A fan-in where no node came back has nothing
    to be partial about, so it is recorded as missed whichever of these is set:
    an answer composed out of no results is worse than a turn that says it
    never got any.
    """

    #: Fire with what arrived, and record that the answer was partial.
    SUMMARIZE = "summarize"
    #: Do not answer, and record which node never came back.
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class JoinTimedOut:
    """One join whose deadline arrived before every source did.

    `outcome` is what was *done*, not what was configured: a `FAIL` graph whose
    every source had in fact arrived summarizes, because there is nothing left
    to fail on.
    """

    turn_id: str
    summarizer_node_id: str
    arrived: tuple[str, ...]
    missing: tuple[str, ...]
    outcome: OnDeadline


@dataclass(frozen=True, slots=True)
class AnswerDiscarded:
    """A summary that ran and whose answer was thrown away.

    The claim stopped being the caller's while the summarizer was running, so
    somebody else owns this join now. Two answers nobody can tell apart is the
    one outcome none of this is worth, and the work is already spent either way
    — so what is dropped is the writing down, not the run.
    """

    turn_id: str
    summarizer_node_id: str
    taken_over_by: str


@dataclass(frozen=True, slots=True)
class Completion:
    """One arrival at a join.

    Three strings rather than the whole event: what firing the summarizer needs
    is the agent's name and what it said, and carrying the rest would put a
    message's whole metadata in the ledger for no reader.
    """

    source_node_id: str
    source_alias: str
    text: str


@dataclass(frozen=True, slots=True)
class Claim:
    """Who took the right to fire a summarizer, and when.

    `previous_holder` is set only on a takeover, and it is the difference
    between reading a recovered join afterwards and reading a slow one.
    """

    holder: str
    claimed_at: float
    previous_holder: str | None = None


@dataclass(frozen=True, slots=True)
class ClaimState:
    """What the stream says about one join's claim.

    Two facts rather than one, because "nobody holds it" and "it is finished"
    are different answers to whether it may be taken.
    """

    held: Claim | None
    completed: bool


class JoinLedger(Protocol):
    """What a fan-in remembers between the first completion and the last."""

    @property
    def holder(self) -> str:
        """Who this ledger is when it claims.

        Asked by whoever fires, to find out afterwards whether the claim it took
        is still the one on the stream. Two ledgers in one process are
        deliberately the same holder — whether a claim may be taken is decided
        by its age, never by who is asking — so what this distinguishes is one
        process from another, which is the takeover that can actually happen.
        """
        ...

    def reserve_fan_in(
        self, turn_id: str, source_node_id: str, summarizer_node_ids: tuple[str, ...]
    ) -> None:
        """Count this source into every summarizer it feeds, once.

        Idempotent by source: a node dispatched twice in one turn is still one
        arrival to wait for, and counting it twice is a join that never
        completes.
        """
        ...

    def record_completion(
        self, turn_id: str, summarizer_node_id: str, completion: Completion
    ) -> None:
        """Remember one arrival, unless the same agent already said the same thing."""
        ...

    def get_expected(self, turn_id: str, summarizer_node_id: str) -> int:
        """How many arrivals this summarizer is waiting for. `0` means nobody said."""
        ...

    def get_reserved_sources(self, turn_id: str, summarizer_node_id: str) -> tuple[str, ...]:
        """Which nodes this summarizer is waiting for, in the order they were reserved.

        The count on its own answers whether a join is complete; the names
        answer which node never came back, which is the only thing a deadline
        has worth saying.
        """
        ...

    def get_completions(self, turn_id: str, summarizer_node_id: str) -> tuple[Completion, ...]:
        """What has arrived, in the order it arrived."""
        ...

    def schedule_deadline(self, turn_id: str, summarizer_node_id: str, due_at: float) -> None:
        """Say when waiting stops being worth it. The first one wins.

        A fan-in of three reserves three times, and a deadline that moved with
        each reservation would be a deadline that a slow dispatcher pushes past
        the thing it is protecting against.
        """
        ...

    def get_deadline(self, turn_id: str, summarizer_node_id: str) -> float | None:
        """When this join stops waiting, or `None` when nobody set one."""
        ...

    def due_joins(self, turn_id: str, now: float) -> tuple[str, ...]:
        """The summarizers in this turn a tick should decide now.

        Past their deadline, not finished, and **not currently claimed**. The
        last of those is what makes a repeating tick self-healing rather than
        one-shot: a worker that claimed and died holds the join until its lease
        runs out, after which the same tick finds it due again with nothing
        needing to be re-armed.
        """
        ...

    def claim_summary(self, turn_id: str, summarizer_node_id: str) -> bool:
        """Take the right to fire this summarizer. `False` when somebody else has it."""
        ...

    def renew_claim(self, turn_id: str, summarizer_node_id: str) -> bool:
        """Push this claim's instant out while the summarizer is still running.

        `False` when it is no longer the caller's — taken over, or finished —
        which is a worker finding out that somebody else owns this join rather
        than assuming it still does.
        """
        ...

    def complete_summary(self, turn_id: str, summarizer_node_id: str) -> None:
        """Say the summarizer returned, so this join is final and never taken over."""
        ...

    def get_claim(self, turn_id: str, summarizer_node_id: str) -> ClaimState:
        """Who holds this join, and whether it has finished."""
        ...


class MemoryJoinLedger:
    """The join, in this process, for as long as it lives.

    The default, and the behaviour every caller had before this module existed.
    It keeps the lease too: a claim whose holder is this process and whose
    summarizer raised is exactly the stall the takeover exists for, and a rule
    that only one of two implementations keeps is a rule with a hole in it.
    """

    def __init__(
        self,
        *,
        holder: str | None = None,
        lease_seconds: float = _LEASE_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._sources: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._deadlines: dict[tuple[str, str], float] = {}
        self._buckets: dict[tuple[str, str], list[Completion]] = defaultdict(list)
        self._claims: dict[tuple[str, str], Claim] = {}
        self._completed: set[tuple[str, str]] = set()
        self._reserved: set[tuple[str, str]] = set()
        self._holder = holder or this_process()
        self._lease = lease_seconds
        self._clock = clock

    @property
    def holder(self) -> str:
        return self._holder

    def reserve_fan_in(
        self, turn_id: str, source_node_id: str, summarizer_node_ids: tuple[str, ...]
    ) -> None:
        source_key = (turn_id, source_node_id)
        if source_key in self._reserved:
            return
        self._reserved.add(source_key)
        for summarizer_node_id in summarizer_node_ids:
            self._sources[(turn_id, summarizer_node_id)].append(source_node_id)

    def record_completion(
        self, turn_id: str, summarizer_node_id: str, completion: Completion
    ) -> None:
        bucket = self._buckets[(turn_id, summarizer_node_id)]
        if not any(
            held.source_node_id == completion.source_node_id and held.text == completion.text
            for held in bucket
        ):
            bucket.append(completion)

    def get_expected(self, turn_id: str, summarizer_node_id: str) -> int:
        return len(self.get_reserved_sources(turn_id, summarizer_node_id))

    def get_reserved_sources(self, turn_id: str, summarizer_node_id: str) -> tuple[str, ...]:
        return tuple(self._sources.get((turn_id, summarizer_node_id), ()))

    def get_completions(self, turn_id: str, summarizer_node_id: str) -> tuple[Completion, ...]:
        return tuple(self._buckets[(turn_id, summarizer_node_id)])

    def schedule_deadline(self, turn_id: str, summarizer_node_id: str, due_at: float) -> None:
        self._deadlines.setdefault((turn_id, summarizer_node_id), due_at)

    def get_deadline(self, turn_id: str, summarizer_node_id: str) -> float | None:
        return self._deadlines.get((turn_id, summarizer_node_id))

    def due_joins(self, turn_id: str, now: float) -> tuple[str, ...]:
        return tuple(
            summarizer_node_id
            for (held_turn, summarizer_node_id), due_at in self._deadlines.items()
            if held_turn == turn_id
            and due_at <= now
            and _decidable(self.get_claim(turn_id, summarizer_node_id), now, self._lease)
        )

    def claim_summary(self, turn_id: str, summarizer_node_id: str) -> bool:
        key = (turn_id, summarizer_node_id)
        if key in self._completed:
            return False
        held = self._claims.get(key)
        if held is not None and not _expired(held, self._clock(), self._lease):
            return False
        self._claims[key] = Claim(
            holder=self._holder,
            claimed_at=self._clock(),
            previous_holder=held.holder if held is not None else None,
        )
        return True

    def renew_claim(self, turn_id: str, summarizer_node_id: str) -> bool:
        key = (turn_id, summarizer_node_id)
        held = self._claims.get(key)
        if key in self._completed or held is None or held.holder != self._holder:
            return False
        self._claims[key] = Claim(
            holder=self._holder,
            claimed_at=self._clock(),
            # Carried, or a renewal would erase the record of the stall it is
            # keeping alive through.
            previous_holder=held.previous_holder,
        )
        return True

    def complete_summary(self, turn_id: str, summarizer_node_id: str) -> None:
        self._completed.add((turn_id, summarizer_node_id))

    def get_claim(self, turn_id: str, summarizer_node_id: str) -> ClaimState:
        key = (turn_id, summarizer_node_id)
        return ClaimState(held=self._claims.get(key), completed=key in self._completed)


class StreamJoinLedger:
    """The join, folded out of an event stream and appended to.

    Every read folds the stream rather than caching, which is what makes a
    second process see what the first one wrote. A turn's join is a handful of
    events, so the fold is cheap; a graph whose fan-in ran to thousands would
    want a projection, and that is a different design rather than a tuning of
    this one.
    """

    def __init__(
        self,
        store: Any,
        stream_prefix: str,
        *,
        holder: str | None = None,
        lease_seconds: float = _LEASE_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = store
        self._stream_for: Callable[[str], str] = lambda turn_id: f"{stream_prefix}:{turn_id}"
        self._holder = holder or this_process()
        self._lease = lease_seconds
        self._clock = clock

    @classmethod
    def on_one_stream(
        cls,
        store: Any,
        stream_name: str,
        *,
        holder: str | None = None,
        lease_seconds: float = _LEASE_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> StreamJoinLedger:
        """Every turn on one stream, for a store whose execution is already the turn.

        `AiwatcherEventStore` is that store: it is opened on one execution and
        refuses a stream name it was not opened for, so a per-turn name would be
        rejected rather than scoped — and there is nothing to scope, because the
        execution bounds the fold already.
        """
        ledger = cls(store, stream_name, holder=holder, lease_seconds=lease_seconds, clock=clock)
        ledger._stream_for = lambda _turn_id: stream_name
        return ledger

    @property
    def holder(self) -> str:
        return self._holder

    # ── Reads ────────────────────────────────────────────────────────────

    def get_expected(self, turn_id: str, summarizer_node_id: str) -> int:
        return len(self.get_reserved_sources(turn_id, summarizer_node_id))

    def get_reserved_sources(self, turn_id: str, summarizer_node_id: str) -> tuple[str, ...]:
        sources: list[str] = []
        for event in self._events(turn_id):
            if event.type != RESERVED or event.data.get("turn_id") != turn_id:
                continue
            if summarizer_node_id not in event.data.get("summarizer_node_ids", ()):
                continue
            source_node_id = str(event.data.get("source_node_id", ""))
            if source_node_id not in sources:
                sources.append(source_node_id)
        return tuple(sources)

    def get_deadline(self, turn_id: str, summarizer_node_id: str) -> float | None:
        for event in self._events(turn_id):
            if event.type != DEADLINE or event.data.get("turn_id") != turn_id:
                continue
            if event.data.get("summarizer_node_id") != summarizer_node_id:
                continue
            # The first, not the last: three reservations are one fan-in, and a
            # deadline that moved with each would be pushed out by the very
            # dispatcher slowness it is there to bound.
            return float(event.data.get("due_at", 0.0))
        return None

    def due_joins(self, turn_id: str, now: float) -> tuple[str, ...]:
        deadlines: dict[str, float] = {}
        finished: set[str] = set()
        held = self._events(turn_id)
        for event in held:
            if event.data.get("turn_id") != turn_id:
                continue
            summarizer_node_id = str(event.data.get("summarizer_node_id", ""))
            if event.type == DEADLINE:
                deadlines.setdefault(summarizer_node_id, float(event.data.get("due_at", 0.0)))
            elif event.type == COMPLETED:
                finished.add(summarizer_node_id)
        return tuple(
            summarizer_node_id
            for summarizer_node_id, due_at in deadlines.items()
            if due_at <= now
            and summarizer_node_id not in finished
            and _decidable(self._claim_state(held, turn_id, summarizer_node_id), now, self._lease)
        )

    def get_completions(self, turn_id: str, summarizer_node_id: str) -> tuple[Completion, ...]:
        gathered: list[Completion] = []
        for event in self._events(turn_id):
            if event.type != RECORDED or event.data.get("turn_id") != turn_id:
                continue
            if event.data.get("summarizer_node_id") != summarizer_node_id:
                continue
            completion = Completion(
                source_node_id=str(event.data.get("source_node_id", "")),
                source_alias=str(event.data.get("source_alias", "")),
                text=str(event.data.get("text", "")),
            )
            # Deduplicated on the way out rather than refused on the way in: an
            # append that had to read first would be a read-then-write race, and
            # what makes a repeat harmless is that the fold ignores it.
            if not any(
                held.source_node_id == completion.source_node_id and held.text == completion.text
                for held in gathered
            ):
                gathered.append(completion)
        return tuple(gathered)

    def get_claim(self, turn_id: str, summarizer_node_id: str) -> ClaimState:
        return self._claim_state(self._events(turn_id), turn_id, summarizer_node_id)

    # ── Writes ───────────────────────────────────────────────────────────

    def reserve_fan_in(
        self, turn_id: str, source_node_id: str, summarizer_node_ids: tuple[str, ...]
    ) -> None:
        if not summarizer_node_ids:
            return
        # Idempotent by source, and folded that way too: `get_expected` counts
        # distinct sources, so a second reservation changes no answer even if
        # two processes both write one.
        if any(
            event.type == RESERVED
            and event.data.get("turn_id") == turn_id
            and event.data.get("source_node_id") == source_node_id
            for event in self._events(turn_id)
        ):
            return
        self._append(
            turn_id,
            RESERVED,
            {
                "turn_id": turn_id,
                "source_node_id": source_node_id,
                "summarizer_node_ids": list(summarizer_node_ids),
            },
        )

    def schedule_deadline(self, turn_id: str, summarizer_node_id: str, due_at: float) -> None:
        if self.get_deadline(turn_id, summarizer_node_id) is not None:
            return
        self._append(
            turn_id,
            DEADLINE,
            {
                "turn_id": turn_id,
                "summarizer_node_id": summarizer_node_id,
                "due_at": due_at,
            },
        )

    def record_completion(
        self, turn_id: str, summarizer_node_id: str, completion: Completion
    ) -> None:
        self._append(
            turn_id,
            RECORDED,
            {
                "turn_id": turn_id,
                "summarizer_node_id": summarizer_node_id,
                "source_node_id": completion.source_node_id,
                "source_alias": completion.source_alias,
                "text": completion.text,
            },
        )

    def claim_summary(self, turn_id: str, summarizer_node_id: str) -> bool:
        """Compare-and-append, because two workers may both see the last arrival.

        `True` means **this call appended the claim** — never "the claim is not
        there". The difference is the whole guarantee: a conflict is usually
        caused by somebody else's *completion* rather than by a competing claim,
        and answering "not claimed, go ahead" to that is two workers firing one
        summarizer, which is precisely what this exists to prevent.

        So a conflict is retried rather than interpreted. `False` is returned
        when a live claim is found, when the summary has already completed, or
        when the retries run out — and the last of those is deliberately the
        safe answer: a summary that did not fire is a turn somebody notices, and
        one that fired twice is two answers nobody can tell apart afterwards.

        A claim older than the lease is *not* a live one, and taking it over
        appends a fresh claim naming who held it before. That is the one case
        where this returns `True` over an existing claim, and it is the whole of
        what turns a worker that died holding one into a turn that still ends.
        """
        return self._write_claim(
            turn_id,
            summarizer_node_id,
            admits=lambda state: (
                state.held is None or _expired(state.held, self._clock(), self._lease)
            ),
            # A takeover names who it took it from.
            previous_holder=lambda state: None if state.held is None else state.held.holder,
        )

    def renew_claim(self, turn_id: str, summarizer_node_id: str) -> bool:
        """Push the instant out, for a summarizer that is taking longer than the lease.

        The mirror of a takeover and it shares its machinery: the same
        compare-and-append, under the opposite precondition. A takeover admits a
        claim that is *not* the caller's and has expired; a renewal admits one
        that *is* the caller's, whatever its age.

        `False` means the claim stopped being the caller's while the summarizer
        was running, and the honest thing to do with the answer then is to throw
        it away.
        """
        return self._write_claim(
            turn_id,
            summarizer_node_id,
            admits=lambda state: state.held is not None and state.held.holder == self._holder,
            # A renewal carries the trace rather than making one: keeping a
            # claim alive must not erase why it was taken.
            previous_holder=lambda state: (
                None if state.held is None else state.held.previous_holder
            ),
        )

    def _write_claim(
        self,
        turn_id: str,
        summarizer_node_id: str,
        *,
        admits: Callable[[ClaimState], bool],
        previous_holder: Callable[[ClaimState], str | None],
    ) -> bool:
        """One compare-and-append, under whichever precondition the caller brought.

        The retry is here rather than in each of them because a conflict means
        the same thing to both — somebody appended between the read and the
        write, and whether it was a claim is the next pass's question.
        """
        from agentic.workflow.errors import ConcurrencyConflictError

        for _ in range(_CLAIM_ATTEMPTS):
            stream = self._store.read_stream(self._stream_for(turn_id))
            state = self._claim_state(_dicts(stream.events), turn_id, summarizer_node_id)
            if state.completed or not admits(state):
                return False
            data: dict[str, Any] = {
                "turn_id": turn_id,
                "summarizer_node_id": summarizer_node_id,
                "holder": self._holder,
                "claimed_at": self._clock(),
            }
            # Whose stall this is recovering from. `AttemptRow`'s
            # `previous_owner`, and it is here for the same reason: a takeover
            # that left no trace reads afterwards as a join that simply took a
            # long time.
            previous = previous_holder(state)
            if previous is not None:
                data["previous_holder"] = previous
            try:
                self._append(turn_id, CLAIMED, data, expected_version=stream.current_version)
            except ConcurrencyConflictError:
                continue
            return True
        return False

    def complete_summary(self, turn_id: str, summarizer_node_id: str) -> None:
        """Say the summarizer returned.

        Not a compare-and-append: it records something that happened rather than
        taking a right, so two of them are one fact and a lost one is the stall
        the lease already covers.
        """
        self._append(
            turn_id,
            COMPLETED,
            {
                "turn_id": turn_id,
                "summarizer_node_id": summarizer_node_id,
                "holder": self._holder,
            },
        )

    # ── Inside ───────────────────────────────────────────────────────────

    @staticmethod
    def _claim_state(events: tuple[Any, ...], turn_id: str, summarizer_node_id: str) -> ClaimState:
        held: Claim | None = None
        completed = False
        for event in events:
            if event.data.get("turn_id") != turn_id:
                continue
            if event.data.get("summarizer_node_id") != summarizer_node_id:
                continue
            if event.type == CLAIMED:
                previous = event.data.get("previous_holder")
                held = Claim(
                    holder=str(event.data.get("holder", "")),
                    claimed_at=float(event.data.get("claimed_at", 0.0)),
                    previous_holder=str(previous) if isinstance(previous, str) else None,
                )
            elif event.type == COMPLETED:
                completed = True
        return ClaimState(held=held, completed=completed)

    def _events(self, turn_id: str) -> tuple[Any, ...]:
        return _dicts(self._store.read_stream(self._stream_for(turn_id)).events)

    def _append(self, turn_id: str, event_type: str, data: dict[str, Any], **options: Any) -> None:
        from agentic.workflow.messages import Event

        self._store.append_to_stream(
            self._stream_for(turn_id), (Event(type=event_type, data=data),), **options
        )


def _expired(claim: Claim, now: float, lease: float) -> bool:
    return now - claim.claimed_at >= lease


def _decidable(state: ClaimState, now: float, lease: float) -> bool:
    """Whether a tick has anything to do about this join.

    Not finished, and nobody live is holding it. A tick that woke a summarizer
    somebody else is already running would be firing beside it, which is the one
    thing none of this is worth.
    """
    if state.completed:
        return False
    return state.held is None or _expired(state.held, now, lease)


def _dicts(events: Any) -> tuple[Any, ...]:
    return tuple(event for event in events if isinstance(getattr(event, "data", None), dict))
