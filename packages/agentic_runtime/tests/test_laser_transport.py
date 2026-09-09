"""The parts of the Laser transport that decide correctness, without a broker.

Everything here is about the one thing an offset-based log does differently
from Redis Streams: an ack is not per entry, it is a watermark. A test that
needed Apache Iggy running would not be run, and this is exactly the logic that
must not be wrong.
"""

from __future__ import annotations

from agentic_runtime.distributed.transport import (
    BEGINNING,
    PartitionLedger,
    entry_id_of,
    parse_entry_id,
)


def test_an_entry_id_round_trips_to_its_partition_and_offset() -> None:
    assert parse_entry_id(entry_id_of(0, 0)) == (0, 0)
    assert parse_entry_id(entry_id_of(3, 41)) == (3, 41)


def test_the_first_entry_of_a_topic_is_not_the_cursor_that_precedes_it() -> None:
    """`0-0` means *before everything*, so nothing may ever be published as it."""
    assert entry_id_of(0, 0) != BEGINNING
    assert parse_entry_id(BEGINNING)[1] == -1


def test_a_contiguous_run_of_acks_commits_the_last_of_them() -> None:
    ledger = PartitionLedger()
    for offset in range(3):
        ledger.deliver(offset, f"payload-{offset}", now_ns=0)

    assert ledger.ack(0) is True
    assert ledger.watermark() == 0
    ledger.ack(1)
    ledger.ack(2)
    assert ledger.watermark() == 2


def test_a_message_left_for_a_retry_holds_the_watermark_behind_it() -> None:
    """The whole reason the ledger exists.

    Offset 1 failed and was not acked. Committing 2 would tell the broker the
    group is past 1, and nothing would ever redeliver it.
    """
    ledger = PartitionLedger()
    for offset in range(3):
        ledger.deliver(offset, f"payload-{offset}", now_ns=0)

    ledger.ack(0)
    ledger.ack(2)
    assert ledger.watermark() == 0

    ledger.ack(1)
    assert ledger.watermark() == 2


def test_a_watermark_that_has_not_moved_is_not_stored_again() -> None:
    ledger = PartitionLedger()
    ledger.deliver(0, "payload", now_ns=0)
    ledger.ack(0)

    assert ledger.watermark() == 0
    assert ledger.watermark() is None


def test_a_group_that_resumes_mid_topic_commits_from_where_it_resumed() -> None:
    """A restarted process is handed offsets above the stored watermark."""
    ledger = PartitionLedger()
    ledger.deliver(17, "payload-17", now_ns=0)
    ledger.ack(17)

    assert ledger.watermark() == 17


def test_acking_twice_reports_the_second_ack_as_nothing_new() -> None:
    ledger = PartitionLedger()
    ledger.deliver(0, "payload", now_ns=0)

    assert ledger.ack(0) is True
    assert ledger.ack(0) is False


def test_acking_an_offset_nobody_delivered_changes_nothing() -> None:
    ledger = PartitionLedger()
    assert ledger.ack(4) is False
    assert ledger.watermark() is None


def test_a_redelivery_of_something_already_committed_does_not_hold_the_run() -> None:
    """A rebalance replays below the watermark; that must not stall it."""
    ledger = PartitionLedger()
    ledger.deliver(0, "payload-0", now_ns=0)
    ledger.ack(0)
    ledger.watermark()

    ledger.deliver(0, "payload-0", now_ns=0)
    ledger.deliver(1, "payload-1", now_ns=0)
    ledger.ack(0)
    ledger.ack(1)

    assert ledger.watermark() == 1


def test_only_a_quiet_unacked_entry_is_redelivered() -> None:
    ledger = PartitionLedger()
    ledger.deliver(0, "payload-0", now_ns=0)
    ledger.deliver(1, "payload-1", now_ns=5_000_000_000)

    reclaimed = ledger.idle(min_idle_ns=1_000_000_000, now_ns=5_000_000_000, limit=10)

    assert reclaimed == [(0, "payload-0")]


def test_an_acked_entry_is_never_redelivered() -> None:
    ledger = PartitionLedger()
    ledger.deliver(0, "payload-0", now_ns=0)
    ledger.ack(0)

    assert ledger.idle(min_idle_ns=0, now_ns=10, limit=10) == []


def test_a_redelivered_entry_starts_its_idle_clock_again() -> None:
    """Otherwise one slow message is handed out on every single poll."""
    ledger = PartitionLedger()
    ledger.deliver(0, "payload-0", now_ns=0)

    assert ledger.idle(min_idle_ns=1_000, now_ns=2_000, limit=10) == [(0, "payload-0")]
    assert ledger.idle(min_idle_ns=1_000, now_ns=2_500, limit=10) == []
    assert ledger.idle(min_idle_ns=1_000, now_ns=3_500, limit=10) == [(0, "payload-0")]


def test_redelivery_respects_the_count_it_was_asked_for() -> None:
    ledger = PartitionLedger()
    for offset in range(5):
        ledger.deliver(offset, f"payload-{offset}", now_ns=0)

    assert ledger.idle(min_idle_ns=0, now_ns=1_000, limit=2) == [
        (0, "payload-0"),
        (1, "payload-1"),
    ]
