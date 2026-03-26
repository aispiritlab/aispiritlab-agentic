from __future__ import annotations

import pytest

from evaluation.print_full_rank_list import (
    _get_full_rank_list,
    _render_table,
    _select_columns,
)


def test_get_full_rank_list_requires_list_payload() -> None:
    with pytest.raises(SystemExit, match="Missing `full_rank_list`"):
        _get_full_rank_list({})

    with pytest.raises(SystemExit, match="must be a list"):
        _get_full_rank_list({"full_rank_list": "invalid"})


def test_select_columns_prefers_known_order_before_remaining() -> None:
    rows = [
        {
            "model_id": "a",
            "rank": 1,
            "custom_metric": 0.9,
            "status": "ok",
        }
    ]

    columns = _select_columns(rows)

    assert columns[:3] == ["rank", "model_id", "status"]
    assert columns[-1] == "custom_metric"


def test_render_table_top_n_limits_rows() -> None:
    rows = [
        {"rank": 1, "model_id": "a", "status": "ok"},
        {"rank": 2, "model_id": "b", "status": "ok"},
    ]

    table = _render_table(rows, top=1)

    assert "model_id" in table
    assert " a " in table
    assert " b " not in table
