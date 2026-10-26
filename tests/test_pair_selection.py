import pandas as pd
import pytest

from src.statarb.research.pair_selection import rank_pairs


def test_ranks_unique_pairs_by_absolute_correlation():
    correlation = pd.DataFrame(
        [
            [1.0, 0.9, -0.7],
            [0.9, 1.0, 0.2],
            [-0.7, 0.2, 1.0],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )

    result = rank_pairs(correlation, min_correlation=0.5)

    assert len(result) == 2
    assert result.iloc[0]["ticker_a"] == "A"
    assert result.iloc[0]["ticker_b"] == "B"
    assert result.iloc[0]["correlation"] == pytest.approx(0.9)
    assert result.iloc[1]["correlation"] == pytest.approx(-0.7)


def test_no_pairs_above_threshold():
    correlation = pd.DataFrame(
        [[1.0, 0.2], [0.2, 1.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    result = rank_pairs(correlation, min_correlation=0.5)

    assert result.empty
    assert list(result.columns) == [
        "ticker_a", "ticker_b", "correlation", "abs_correlation"
    ]


def test_empty_matrix_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        rank_pairs(pd.DataFrame())


def test_invalid_threshold_rejected():
    correlation = pd.DataFrame(
        [[1.0, 0.8], [0.8, 1.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    with pytest.raises(ValueError, match="between -1 and 1"):
        rank_pairs(correlation, min_correlation=1.5)


def test_mismatched_labels_rejected():
    correlation = pd.DataFrame(
        [[1.0, 0.8], [0.8, 1.0]],
        index=["A", "B"],
        columns=["A", "C"],
    )

    with pytest.raises(ValueError, match="matching row and column labels"):
        rank_pairs(correlation)
        