import pandas as pd
import pytest

from src.statarb.research.multiple_testing import adjust_pvalues


def test_adjusts_pvalues_correctly():
    p_values = pd.Series([0.01, 0.04, 0.03, 0.20])

    result = adjust_pvalues(p_values)

    expected = [0.04, 0.0533333333, 0.0533333333, 0.20]

    assert result.tolist() == pytest.approx(expected)


def test_preserves_original_order_and_index():
    p_values = pd.Series(
        [0.04, 0.01, 0.20],
        index=["A-B", "A-C", "B-C"],
    )

    result = adjust_pvalues(p_values)

    assert result.index.tolist() == ["A-B", "A-C", "B-C"]
    assert result.loc["A-C"] == pytest.approx(0.03)


def test_empty_pvalues_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        adjust_pvalues(pd.Series(dtype=float))


def test_missing_pvalues_rejected():
    with pytest.raises(ValueError, match="missing"):
        adjust_pvalues(pd.Series([0.01, float("nan")]))


@pytest.mark.parametrize("invalid", [-0.1, 1.1])
def test_out_of_range_pvalues_rejected(invalid):
    with pytest.raises(ValueError, match="between 0 and 1"):
        adjust_pvalues(pd.Series([0.01, invalid]))


def test_adjusted_pvalues_are_monotonic_when_sorted():
    p_values = pd.Series([0.03, 0.001, 0.02, 0.005])

    result = adjust_pvalues(p_values)
    sorted_indices = p_values.sort_values().index
    sorted_adjusted = result.loc[sorted_indices].tolist()

    assert sorted_adjusted == sorted(sorted_adjusted)


def test_adjusted_pvalues_never_exceed_one():
    result = adjust_pvalues(pd.Series([0.5, 0.8, 1.0]))

    assert (result <= 1.0).all()
    