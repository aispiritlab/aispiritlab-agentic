"""A fan-in that outlives the process that started it.

The graph's join used to be three dictionaries on one object: how many
arrivals a summarizer waits for, which have arrived, and whether it has fired.
All three died with the process, so a worker restarting between the second and
third completion lost the join with nothing to say it happened.

These are written against `agentic.workflow.InMemoryEventStore` rather than a
running aiwatcher, because what is being tested is the ledger's own rules — the
same `EventStore` protocol `AiwatcherEventStore` satisfies, so a ledger that
keeps them here keeps them over a shared history too.
"""

from __future__ import annotations

import pytest

from agentic.workflow import InMemoryEventStore
from agentic_graph.join import Completion, MemoryJoinLedger, StreamJoinLedger

TURN = "turn-1"
SUMMARIZER = "summarizer"
LEASE = 60.0


def arrival(source: str, text: str = "found something") -> Completion:
    return Completion(source_node_id=source, source_alias=source.title(), text=text)


class Clock:
    """A clock a test moves, because a lease tested against a real one is a sleep."""

    def __init__(self, now: float = 1_700_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def after(self, seconds: float) -> None:
        self.now += seconds


def ledgers(clock: Clock | None = None) -> list[tuple[str, object]]:
    """Both implementations, over one clock and one lease.

    Parametrising over the pair is the point: a rule only one of them keeps is a
    rule that holds in a preview and not in the worker it was written for.
    """
    ticking = clock or Clock()
    store = InMemoryEventStore()
    return [
        ("memory", MemoryJoinLedger(lease_seconds=LEASE, clock=ticking)),
        ("stream", StreamJoinLedger(store, "graph:g-1", lease_seconds=LEASE, clock=ticking)),
    ]


def taken_over(name: str) -> list[tuple[str, object, Clock]]:
    """One implementation and the clock its lease is measured against."""
    clock = Clock()
    return [(label, ledger, clock) for label, ledger in ledgers(clock) if label == name]


@pytest.mark.parametrize(("name", "ledger"), ledgers())
def test_a_fan_in_waits_for_every_source_that_was_reserved(name: str, ledger: object) -> None:
    for source in ("search-a", "search-b", "search-c"):
        ledger.reserve_fan_in(TURN, source, (SUMMARIZER,))
    assert ledger.get_expected(TURN, SUMMARIZER) == 3, name

    ledger.record_completion(TURN, SUMMARIZER, arrival("search-a"))
    ledger.record_completion(TURN, SUMMARIZER, arrival("search-b"))
    assert len(ledger.get_completions(TURN, SUMMARIZER)) == 2, name


@pytest.mark.parametrize(("name", "ledger"), ledgers())
def test_one_source_dispatched_twice_is_still_one_arrival_to_wait_for(
    name: str, ledger: object
) -> None:
    # Counting it twice is a join that never completes: the summarizer waits for
    # an arrival that no second node is ever going to make.
    ledger.reserve_fan_in(TURN, "search-a", (SUMMARIZER,))
    ledger.reserve_fan_in(TURN, "search-a", (SUMMARIZER,))
    assert ledger.get_expected(TURN, SUMMARIZER) == 1, name


@pytest.mark.parametrize(("name", "ledger"), ledgers())
def test_the_same_agent_saying_the_same_thing_twice_fills_one_slot(
    name: str, ledger: object
) -> None:
    # A redelivery, not a second arrival. Two of them would complete a fan-in of
    # three with two agents having answered.
    ledger.reserve_fan_in(TURN, "search-a", (SUMMARIZER,))
    ledger.record_completion(TURN, SUMMARIZER, arrival("search-a", "same"))
    ledger.record_completion(TURN, SUMMARIZER, arrival("search-a", "same"))
    assert len(ledger.get_completions(TURN, SUMMARIZER)) == 1, name

    # And the same agent saying something *different* is a second arrival, which
    # is what a retried search that found more looks like.
    ledger.record_completion(TURN, SUMMARIZER, arrival("search-a", "more"))
    assert len(ledger.get_completions(TURN, SUMMARIZER)) == 2, name


@pytest.mark.parametrize(("name", "ledger"), ledgers())
def test_only_one_caller_ever_claims_a_summary(name: str, ledger: object) -> None:
    # The one operation that cannot be a read followed by a decision: a graph
    # that answered twice is worse than one that answered late.
    assert ledger.claim_summary(TURN, SUMMARIZER) is True, name
    assert ledger.claim_summary(TURN, SUMMARIZER) is False, name


@pytest.mark.parametrize(("name", "ledger"), ledgers())
def test_two_turns_of_one_graph_do_not_share_a_join(name: str, ledger: object) -> None:
    ledger.reserve_fan_in(TURN, "search-a", (SUMMARIZER,))
    assert ledger.get_expected("turn-2", SUMMARIZER) == 0, name
    ledger.claim_summary(TURN, SUMMARIZER)
    assert ledger.claim_summary("turn-2", SUMMARIZER) is True, name


def test_a_fan_in_of_three_survives_a_restart_between_the_second_and_the_third() -> None:
    # Phase 13's own exit, at the ledger. The store is the shared, durable half,
    # so what restarts is the *worker*: one ledger takes two arrivals and goes
    # away, and a second one — holding nothing it learnt — takes the third and
    # finds a join that is complete.
    store = InMemoryEventStore()
    stream = "graph:searcher-summarizer"

    before = StreamJoinLedger(store, stream)
    for source in ("search-a", "search-b", "search-c"):
        before.reserve_fan_in(TURN, source, (SUMMARIZER,))
    before.record_completion(TURN, SUMMARIZER, arrival("search-a", "a"))
    before.record_completion(TURN, SUMMARIZER, arrival("search-b", "b"))
    assert len(before.get_completions(TURN, SUMMARIZER)) == 2
    assert before.get_expected(TURN, SUMMARIZER) == 3
    del before

    # The worker is gone. Everything it knew is in the stream.
    after = StreamJoinLedger(store, stream)
    assert after.get_expected(TURN, SUMMARIZER) == 3, "the fan-in survived"
    assert len(after.get_completions(TURN, SUMMARIZER)) == 2, "so did the arrivals"

    after.record_completion(TURN, SUMMARIZER, arrival("search-c", "c"))
    gathered = after.get_completions(TURN, SUMMARIZER)
    assert len(gathered) == 3
    assert [one.source_node_id for one in gathered] == ["search-a", "search-b", "search-c"]

    # And it fires once: the replacement claims it, and anything that comes
    # after — a redelivery, a third worker, the original coming back — does not.
    assert after.claim_summary(TURN, SUMMARIZER) is True
    assert after.claim_summary(TURN, SUMMARIZER) is False
    assert StreamJoinLedger(store, stream).claim_summary(TURN, SUMMARIZER) is False


class Contended:
    """A store where the first append of each call loses a race it did not enter.

    The conflict is caused by an unrelated write — a *completion* landing
    between the claim's read and its append, which is the ordinary traffic on a
    fan-in's stream rather than an exotic case. A claim that read "no claim
    found" from that would let two workers fire one summarizer.
    """

    def __init__(self, store: InMemoryEventStore, stream: str) -> None:
        self._store = store
        self._stream = stream
        self.interfered = 0

    def read_stream(self, stream_name: str, **options: object):
        return self._store.read_stream(stream_name, **options)

    def append_to_stream(self, stream_name: str, events, **options: object):
        if self.interfered == 0 and options.get("expected_version") is not None:
            self.interfered += 1
            # Somebody else's completion, appended first.
            StreamJoinLedger(self._store, self._stream).record_completion(
                TURN, SUMMARIZER, arrival("late-arrival", "late")
            )
        return self._store.append_to_stream(stream_name, events, **options)


def test_a_conflict_caused_by_a_completion_is_not_read_as_permission_to_fire() -> None:
    # The bug this test exists for: a conflict is usually somebody else's
    # completion, and answering "no claim found, go ahead" to that is two
    # workers firing one summarizer. `True` has to mean "this call appended the
    # claim", never "the claim is not there".
    store = InMemoryEventStore()
    stream = "graph:contended"
    contended = Contended(store, stream)

    ledger = StreamJoinLedger(contended, stream)
    ledger.reserve_fan_in(TURN, "search-a", (SUMMARIZER,))
    assert ledger.claim_summary(TURN, SUMMARIZER) is True
    assert contended.interfered == 1, "the race was actually forced"

    # And the claim is on the stream, so nobody else gets one.
    assert StreamJoinLedger(store, stream).claim_summary(TURN, SUMMARIZER) is False


def test_two_workers_that_both_see_the_last_arrival_fire_once_between_them() -> None:
    # Neither is a redelivery of the other: two live processes, one stream, one
    # summarizer. The claim is a compare-and-append, so exactly one wins.
    store = InMemoryEventStore()
    stream = "graph:race"
    one = StreamJoinLedger(store, stream)
    other = StreamJoinLedger(store, stream)
    one.reserve_fan_in(TURN, "search-a", (SUMMARIZER,))
    one.record_completion(TURN, SUMMARIZER, arrival("search-a"))

    claims = [one.claim_summary(TURN, SUMMARIZER), other.claim_summary(TURN, SUMMARIZER)]
    assert claims.count(True) == 1, claims


# ── The stream is scoped to a turn ───────────────────────────────────────


def test_two_turns_of_one_graph_write_two_streams() -> None:
    # A turn is a traversal: what a join opens is answered inside it and never
    # read again. One stream for the graph meant every completion folded every
    # turn that came before it.
    store = InMemoryEventStore()
    ledger = StreamJoinLedger(store, "graph:two-turns")

    ledger.reserve_fan_in("turn-a", "search-a", (SUMMARIZER,))
    ledger.reserve_fan_in("turn-b", "search-a", (SUMMARIZER,))

    assert store.read_stream("graph:two-turns:turn-a").current_version == 1
    assert store.read_stream("graph:two-turns:turn-b").current_version == 1
    assert store.read_stream("graph:two-turns").stream_exists is False, (
        "nothing is written to the graph-wide name any more"
    )


class Counting:
    """A store that remembers the largest fold it served."""

    def __init__(self, store: InMemoryEventStore) -> None:
        self._store = store
        self.largest_fold = 0

    def read_stream(self, stream_name: str, **options: object):
        answer = self._store.read_stream(stream_name, **options)
        self.largest_fold = max(self.largest_fold, len(answer.events))
        return answer

    def append_to_stream(self, stream_name: str, events, **options: object):
        return self._store.append_to_stream(stream_name, events, **options)


def test_a_fold_after_a_hundred_turns_reads_only_this_turns_events() -> None:
    # The cost this scoping exists for. Unscoped, the hundredth turn's first
    # completion folds every event the ninety-nine before it wrote.
    counting = Counting(InMemoryEventStore())
    ledger = StreamJoinLedger(counting, "graph:many")

    for turn in range(100):
        ledger.reserve_fan_in(f"turn-{turn}", "search-a", (SUMMARIZER,))
        ledger.record_completion(f"turn-{turn}", SUMMARIZER, arrival("search-a"))

    assert ledger.get_expected("turn-99", SUMMARIZER) == 1
    assert len(ledger.get_completions("turn-99", SUMMARIZER)) == 1
    assert counting.largest_fold <= 3, (
        f"a fold saw {counting.largest_fold} events, so a turn is reading another turn's"
    )


# ── A claim that can be taken over ───────────────────────────────────────


@pytest.mark.parametrize(("name", "ledger", "clock"), taken_over("memory") + taken_over("stream"))
def test_a_claim_whose_holder_stopped_is_taken_over_after_the_lease(
    name: str, ledger: object, clock: Clock
) -> None:
    # The stall this exists for: the claim is taken *before* the summarizer
    # runs and nothing releases it, so a worker that died holding one left a
    # join that never fired and said nothing.
    ledger.reserve_fan_in(TURN, "search-a", (SUMMARIZER,))
    assert ledger.claim_summary(TURN, SUMMARIZER) is True, name

    clock.after(LEASE - 1)
    assert ledger.claim_summary(TURN, SUMMARIZER) is False, f"{name}: still somebody's"

    clock.after(2)
    assert ledger.claim_summary(TURN, SUMMARIZER) is True, f"{name}: taken over"
    held = ledger.get_claim(TURN, SUMMARIZER).held
    assert held is not None and held.previous_holder is not None, (
        f"{name}: a takeover that left no trace reads as a join that was slow"
    )


@pytest.mark.parametrize(("name", "ledger", "clock"), taken_over("memory") + taken_over("stream"))
def test_a_claim_that_completed_is_never_taken_over_however_old(
    name: str, ledger: object, clock: Clock
) -> None:
    # The other half, and the one that keeps "never fire twice": a finished
    # join is final, so the lease may pass a hundred times over without
    # anything deciding the summarizer never ran.
    ledger.reserve_fan_in(TURN, "search-a", (SUMMARIZER,))
    assert ledger.claim_summary(TURN, SUMMARIZER) is True, name
    ledger.complete_summary(TURN, SUMMARIZER)

    clock.after(LEASE * 100)
    assert ledger.claim_summary(TURN, SUMMARIZER) is False, name
    assert ledger.get_claim(TURN, SUMMARIZER).completed is True, name


def test_a_taken_over_claim_names_the_worker_that_stopped() -> None:
    clock = Clock()
    store = InMemoryEventStore()
    stopped = StreamJoinLedger(
        store, "graph:takeover", holder="worker-a", lease_seconds=LEASE, clock=clock
    )
    replacement = StreamJoinLedger(
        store, "graph:takeover", holder="worker-b", lease_seconds=LEASE, clock=clock
    )

    stopped.reserve_fan_in(TURN, "search-a", (SUMMARIZER,))
    assert stopped.claim_summary(TURN, SUMMARIZER) is True
    assert replacement.claim_summary(TURN, SUMMARIZER) is False, "worker-a is alive"

    clock.after(LEASE)
    assert replacement.claim_summary(TURN, SUMMARIZER) is True
    held = replacement.get_claim(TURN, SUMMARIZER)
    assert held.held is not None
    assert held.held.holder == "worker-b"
    assert held.held.previous_holder == "worker-a"

    # And the worker that stopped does not come back and fire beside it.
    replacement.complete_summary(TURN, SUMMARIZER)
    assert stopped.claim_summary(TURN, SUMMARIZER) is False


def test_a_summary_that_is_still_running_holds_its_claim_against_a_second_worker() -> None:
    # The trade the lease makes, written down: within it, nobody else fires.
    clock = Clock()
    store = InMemoryEventStore()
    running = StreamJoinLedger(
        store, "graph:slow", holder="worker-a", lease_seconds=LEASE, clock=clock
    )
    other = StreamJoinLedger(
        store, "graph:slow", holder="worker-b", lease_seconds=LEASE, clock=clock
    )

    running.reserve_fan_in(TURN, "search-a", (SUMMARIZER,))
    assert running.claim_summary(TURN, SUMMARIZER) is True
    for _ in range(5):
        clock.after(LEASE / 10)
        assert other.claim_summary(TURN, SUMMARIZER) is False


# ── A deadline, and what a tick may act on ───────────────────────────────


@pytest.mark.parametrize(("name", "ledger", "clock"), taken_over("memory") + taken_over("stream"))
def test_a_deadline_is_due_once_it_has_passed_and_final_once_it_has_answered(
    name: str, ledger: object, clock: Clock
) -> None:
    ledger.reserve_fan_in(TURN, "search-a", (SUMMARIZER,))
    ledger.schedule_deadline(TURN, SUMMARIZER, clock.now + 30)
    assert ledger.due_joins(TURN, clock.now) == (), name

    clock.after(30)
    assert ledger.due_joins(TURN, clock.now) == (SUMMARIZER,), name

    ledger.complete_summary(TURN, SUMMARIZER)
    assert ledger.due_joins(TURN, clock.now) == (), f"{name}: answered is final"


@pytest.mark.parametrize(("name", "ledger", "clock"), taken_over("memory") + taken_over("stream"))
def test_a_tick_leaves_a_join_somebody_is_running_alone_until_the_lease_runs_out(
    name: str, ledger: object, clock: Clock
) -> None:
    # What makes a repeating tick self-healing rather than one-shot: a worker
    # that claimed and died holds the join until its lease expires, and then the
    # same tick finds it due again with nothing needing to be re-armed. Waking a
    # summarizer somebody else is running would be firing beside it.
    ledger.reserve_fan_in(TURN, "search-a", (SUMMARIZER,))
    ledger.schedule_deadline(TURN, SUMMARIZER, clock.now)
    assert ledger.due_joins(TURN, clock.now) == (SUMMARIZER,), name

    assert ledger.claim_summary(TURN, SUMMARIZER) is True, name
    assert ledger.due_joins(TURN, clock.now) == (), f"{name}: somebody is running it"

    clock.after(LEASE)
    assert ledger.due_joins(TURN, clock.now) == (SUMMARIZER,), f"{name}: and it stopped"


@pytest.mark.parametrize(("name", "ledger", "clock"), taken_over("memory") + taken_over("stream"))
def test_the_first_deadline_of_a_fan_in_is_the_one_that_counts(
    name: str, ledger: object, clock: Clock
) -> None:
    # Three sources reserve one fan-in. A deadline that moved with each would be
    # pushed out by the dispatcher slowness it is there to bound.
    ledger.schedule_deadline(TURN, SUMMARIZER, clock.now + 30)
    ledger.schedule_deadline(TURN, SUMMARIZER, clock.now + 3_000)
    assert ledger.get_deadline(TURN, SUMMARIZER) == clock.now + 30, name


# ── Renewing a claim, for a summarizer slower than the lease ─────────────


@pytest.mark.parametrize(("name", "ledger", "clock"), taken_over("memory") + taken_over("stream"))
def test_a_renewed_claim_is_not_taken_over_however_long_the_summarizer_runs(
    name: str, ledger: object, clock: Clock
) -> None:
    # The other dial beside `lease_seconds`. Renewing is opt-in because it costs
    # a heartbeat and being taken over is sometimes acceptable; what it buys is
    # a long summarizer that is never fired beside itself.
    ledger.reserve_fan_in(TURN, "search-a", (SUMMARIZER,))
    assert ledger.claim_summary(TURN, SUMMARIZER) is True, name

    for _ in range(5):
        clock.after(LEASE - 1)
        assert ledger.renew_claim(TURN, SUMMARIZER) is True, name
        assert ledger.due_joins(TURN, clock.now) == (), f"{name}: still running"


@pytest.mark.parametrize(("name", "ledger", "clock"), taken_over("memory") + taken_over("stream"))
def test_a_renewal_after_the_summary_completed_is_refused(
    name: str, ledger: object, clock: Clock
) -> None:
    ledger.reserve_fan_in(TURN, "search-a", (SUMMARIZER,))
    ledger.claim_summary(TURN, SUMMARIZER)
    ledger.complete_summary(TURN, SUMMARIZER)
    assert ledger.renew_claim(TURN, SUMMARIZER) is False, name


@pytest.mark.parametrize(("name", "ledger", "clock"), taken_over("memory") + taken_over("stream"))
def test_a_renewal_of_a_claim_nobody_ever_took_is_refused(
    name: str, ledger: object, clock: Clock
) -> None:
    # A renewal admits a claim that *is* the caller's, whatever its age. There
    # is nothing here to be the caller's.
    assert ledger.renew_claim(TURN, SUMMARIZER) is False, name


def test_a_worker_that_was_taken_over_finds_out_when_it_tries_to_renew() -> None:
    # Which is the point of `renew_claim` returning something: the honest thing
    # to do with an answer whose claim is somebody else's is to throw it away.
    clock = Clock()
    store = InMemoryEventStore()
    slow = StreamJoinLedger(
        store, "graph:renew", holder="worker-a", lease_seconds=LEASE, clock=clock
    )
    replacement = StreamJoinLedger(
        store, "graph:renew", holder="worker-b", lease_seconds=LEASE, clock=clock
    )

    slow.reserve_fan_in(TURN, "search-a", (SUMMARIZER,))
    assert slow.claim_summary(TURN, SUMMARIZER) is True
    clock.after(LEASE)
    assert replacement.claim_summary(TURN, SUMMARIZER) is True, "it looked dead"

    assert slow.renew_claim(TURN, SUMMARIZER) is False, "and it finds out"
    held = replacement.get_claim(TURN, SUMMARIZER).held
    assert held is not None and held.holder == "worker-b"


def test_a_renewal_keeps_the_record_of_the_takeover_it_is_holding() -> None:
    # A renewal carries the trace rather than making one: keeping a claim alive
    # must not erase why it was taken.
    clock = Clock()
    store = InMemoryEventStore()
    stopped = StreamJoinLedger(
        store, "graph:trace", holder="worker-a", lease_seconds=LEASE, clock=clock
    )
    replacement = StreamJoinLedger(
        store, "graph:trace", holder="worker-b", lease_seconds=LEASE, clock=clock
    )

    stopped.reserve_fan_in(TURN, "search-a", (SUMMARIZER,))
    stopped.claim_summary(TURN, SUMMARIZER)
    clock.after(LEASE)
    replacement.claim_summary(TURN, SUMMARIZER)
    clock.after(LEASE - 1)
    assert replacement.renew_claim(TURN, SUMMARIZER) is True

    held = replacement.get_claim(TURN, SUMMARIZER).held
    assert held is not None
    assert held.holder == "worker-b"
    assert held.previous_holder == "worker-a"


# ── What the engine's own delivery does to the fold ──────────────────────────


def test_a_deadline_the_engine_hands_back_does_not_move_the_one_it_was_given() -> None:
    """The shape aiwatcher's timer actually has, written the way it arrives.

    `JoinTimers` stores the scheduling message *as* the timer's payload, so the
    tick appends that same message back when it fires — a second
    `graph.join_deadline_scheduled` in the stream. It does not come through
    :meth:`StreamJoinLedger.schedule_deadline`, whose own guard would have
    stopped it, so the guard that matters is in the fold: first wins, and a fold
    that took the last one would push the deadline out by exactly the lateness
    that made it fire.
    """
    from agentic.workflow.messages import Event

    store = InMemoryEventStore()
    stream = "graph:g-1:" + TURN
    ledger = StreamJoinLedger(store, "graph:g-1", lease_seconds=LEASE)
    ledger.reserve_fan_in(TURN, "search-a", (SUMMARIZER,))
    ledger.schedule_deadline(TURN, SUMMARIZER, 1_700_000_030.0)

    # The engine's delivery, straight onto the stream, twice — a tick that fired
    # and a retry of the same delivery.
    delivered = Event(
        type="graph.join_deadline_scheduled",
        data={
            "turn_id": TURN,
            "summarizer_node_id": SUMMARIZER,
            "due_at": 1_700_000_999.0,
        },
    )
    store.append_to_stream(stream, (delivered,))
    store.append_to_stream(stream, (delivered,))

    assert ledger.get_deadline(TURN, SUMMARIZER) == 1_700_000_030.0
    assert ledger.due_joins(TURN, 1_700_000_031.0) == (SUMMARIZER,)
    assert ledger.due_joins(TURN, 1_700_000_029.0) == ()


def test_one_stream_holds_two_turns_without_either_reading_the_other() -> None:
    """The worker's constructor, and the property it rests on.

    `AiwatcherEventStore` is opened on one execution and refuses a stream name it
    was not opened for, so a per-turn stream name is not available there. What
    keeps two turns apart on one stream is that every fold filters on `turn_id`
    — which is worth a test of its own, because the per-turn stream name is what
    makes it true everywhere else.
    """
    store = InMemoryEventStore()
    ledger = StreamJoinLedger.on_one_stream(store, "execution:run-7", lease_seconds=LEASE)

    ledger.reserve_fan_in("turn-a", "search-a", (SUMMARIZER,))
    ledger.reserve_fan_in("turn-b", "search-a", (SUMMARIZER,))
    ledger.reserve_fan_in("turn-b", "search-b", (SUMMARIZER,))
    ledger.record_completion("turn-a", SUMMARIZER, arrival("search-a", "a's answer"))

    assert ledger.get_expected("turn-a", SUMMARIZER) == 1
    assert ledger.get_expected("turn-b", SUMMARIZER) == 2
    assert [one.text for one in ledger.get_completions("turn-a", SUMMARIZER)] == ["a's answer"]
    assert ledger.get_completions("turn-b", SUMMARIZER) == ()

    # And a claim is per turn, not per stream: answering one turn must not make
    # the next one look answered.
    assert ledger.claim_summary("turn-a", SUMMARIZER) is True
    ledger.complete_summary("turn-a", SUMMARIZER)
    assert ledger.get_claim("turn-a", SUMMARIZER).completed is True
    assert ledger.get_claim("turn-b", SUMMARIZER).completed is False
    assert ledger.claim_summary("turn-b", SUMMARIZER) is True
