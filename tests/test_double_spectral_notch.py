import math

import pytest
import torch

from si_fdjscc.comm import OFDMFrequencySelectiveChannel
from si_fdjscc.evaluation.evaluate import _notch_metadata


def test_two_random_notches_are_seeded_separated_and_applied_to_data_only() -> None:
    channel = OFDMFrequencySelectiveChannel()
    specification = {
        "count": 2,
        "start": "random",
        "width_range": [8, 12],
        "attenuation_db_range": [15, 25],
        "min_gap": 4,
    }
    first = channel.prepare_state(
        1, 2, torch.device("cpu"), channel_seed=123, spectral_notch=specification
    )
    repeated = channel.prepare_state(
        1, 2, torch.device("cpu"), channel_seed=123, spectral_notch=specification
    )
    different = channel.prepare_state(
        1, 2, torch.device("cpu"), channel_seed=124, spectral_notch=specification
    )

    assert first.spectral_notch == repeated.spectral_notch
    assert torch.equal(first.frequency_gain, repeated.frequency_gain)
    assert first.spectral_notch != different.spectral_notch
    resolved = first.spectral_notch
    assert resolved["count"] == 2
    assert resolved["min_gap"] == 4
    left, right = resolved["notches"]
    assert 8 <= left["width"] <= 12
    assert 8 <= right["width"] <= 12
    assert right["start"] >= left["start"] + left["width"] + 4
    assert right["start"] + right["width"] <= channel.plan.n_data
    assert resolved["total_width"] == left["width"] + right["width"]

    expected_gain = torch.ones_like(first.frequency_gain)
    data_indices = channel.plan.tensor_indices("data", torch.device("cpu"))
    for notch in resolved["notches"]:
        assert 15 <= notch["attenuation_db"] <= 25
        assert notch["amplitude"] == pytest.approx(
            10 ** (-notch["attenuation_db"] / 20)
        )
        expected_gain[data_indices[notch["start"] : notch["start"] + notch["width"]]] = (
            notch["amplitude"]
        )
    assert torch.allclose(first.frequency_gain, expected_gain)
    assert torch.allclose(
        first.h_freq, first.base_h_freq * expected_gain.view(1, 1, -1)
    )
    pilot_indices = channel.plan.tensor_indices("pilot", torch.device("cpu"))
    assert torch.equal(first.frequency_gain[pilot_indices], torch.ones(len(pilot_indices)))


def test_two_fixed_notches_support_minus_5_db_csi_nmse() -> None:
    channel = OFDMFrequencySelectiveChannel()
    state = channel.prepare_state(
        1,
        2,
        torch.device("cpu"),
        csi_mode="imperfect",
        csi_nmse_db=-5.0,
        channel_seed=7,
        spectral_notch={
            "count": 2,
            "starts": [2, 28],
            "width": 8,
            "attenuation_db": 20,
            "min_gap": 4,
        },
    )
    assert [notch["start"] for notch in state.spectral_notch["notches"]] == [2, 28]
    assert state.h_hat is not None
    error_power = (state.h_hat - state.h_freq).abs().square().mean()
    channel_power = state.h_freq.abs().square().mean()
    assert 10 * math.log10((error_power / channel_power).item()) == pytest.approx(-5.0, abs=0.05)


def test_two_notches_are_recorded_without_pretending_to_be_one_interval() -> None:
    metadata = _notch_metadata({
        "count": 2,
        "notches": [
            {"start": 28, "width": 9, "attenuation_db": 22.0},
            {"start": 4, "width": 11, "attenuation_db": 18.0},
        ],
    })
    assert metadata["notch_count"] == 2
    assert metadata["notch_total_width"] == 20
    assert metadata["notch_start"] == -1
    assert metadata["notch_width"] == 0
    assert metadata["notch_1_start"] == 4
    assert metadata["notch_2_start"] == 28
    assert metadata["notches_json"] == (
        '[{"attenuation_db":18.0,"start":4,"width":11},'
        '{"attenuation_db":22.0,"start":28,"width":9}]'
    )


@pytest.mark.parametrize(
    "specification, message",
    [
        ({"count": 3}, "count"),
        ({"count": 2, "width": 24}, "do not fit"),
        ({"count": 2, "width": 8, "starts": [2, 10]}, "overlap"),
        ({"count": 2, "width": 8, "min_gap": 0}, "min_gap"),
    ],
)
def test_two_notches_reject_invalid_geometry(specification, message) -> None:
    channel = OFDMFrequencySelectiveChannel()
    with pytest.raises(ValueError, match=message):
        channel.prepare_state(
            1, 1, torch.device("cpu"), channel_seed=7, spectral_notch=specification
        )
