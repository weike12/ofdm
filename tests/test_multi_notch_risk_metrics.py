import pytest
import torch

from si_fdjscc.evaluation.metrics import assigned_mapping_risk_metrics


def _metrics(notch):
    # Two symbols occupy the two notch intervals; padding in both intervals
    # must not contribute to the exposure denominator or numerator.
    return assigned_mapping_risk_metrics(
        torch.tensor([[[0, 1, -1, 2, 3, 4, -1, 5]]]),
        torch.tensor([0, 0, 0, 1, 1, 1]),
        torch.tensor([[2.0, 1.0]]),
        ((0, 1, 2, 3), (4, 5, 6, 7)),
        spectral_notch=notch,
    )


def test_two_notch_exposure_uses_union_and_ignores_padding():
    rows = _metrics({
        "count": 2,
        "notches": [
            {"start": 1, "width": 2, "attenuation_db": 20.0},
            {"start": 5, "width": 2, "attenuation_db": 20.0},
        ],
        "min_gap": 2,
        "total_width": 4,
    })
    assert rows["mapping_symbol_count"].item() == 6
    assert rows["mapping_notch_fraction"].item() == pytest.approx(2 / 6)
    assert rows["mapping_high_notch_fraction"].item() == pytest.approx(1 / 3)
    assert rows["mapping_low_notch_fraction"].item() == pytest.approx(1 / 3)


def test_nested_one_notch_matches_legacy_metadata():
    legacy = _metrics({"start": 1, "width": 2})
    nested = _metrics({"count": 1, "notches": [{"start": 1, "width": 2}]})
    for key in (
        "mapping_notch_fraction", "mapping_high_notch_fraction", "mapping_low_notch_fraction"
    ):
        assert torch.equal(legacy[key], nested[key])


@pytest.mark.parametrize(
    "notch, error",
    [
        ({"count": 2, "notches": [{"start": 1, "width": 2}]}, "count"),
        ({"count": 2, "notches": [{"start": 1, "width": 2}, {"start": 7, "width": 2}]}, "outside"),
        ({"count": 2, "notches": [{"start": 1, "width": 2}, {"start": 5, "width": 0}]}, "positive"),
        ({"count": 2, "notches": []}, "nonempty"),
    ],
)
def test_invalid_two_notch_metadata_is_rejected(notch, error):
    with pytest.raises(ValueError, match=error):
        _metrics(notch)
