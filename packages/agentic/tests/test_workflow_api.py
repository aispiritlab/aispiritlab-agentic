from __future__ import annotations

import inspect

import pytest

from agentic.workflow import Decider
from agentic.workflow.decider import Decider as EventSourcingDecider


def test_top_level_decider_is_event_sourcing_class() -> None:
    assert Decider is EventSourcingDecider
    assert inspect.isclass(Decider)


def test_reactor_module_does_not_export_legacy_decider_alias() -> None:
    with pytest.raises(ImportError):
        exec("from agentic.workflow.reactor import Decider", {})


def test_runtime_reactor_module_does_not_export_legacy_decider_alias() -> None:
    with pytest.raises(ImportError):
        exec("from agentic_runtime.reactor import Decider", {})
