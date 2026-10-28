import torch

from si_fdjscc.comm import SubcarrierPlan


def test_default_subcarrier_plan() -> None:
    plan = SubcarrierPlan()
    assert len(plan.occupied_shifted) == 52
    assert plan.n_data == 48
    assert plan.n_pilot == 4
    assert 0 not in plan.occupied_shifted
    assert set(plan.pilot_shifted).isdisjoint(plan.data_shifted)
    assert plan.shifted_to_fft((-26, -1, 1, 26), 64) == (38, 63, 1, 26)
    assert [len(band) for band in plan.band_data_positions] == [12, 12, 12, 12]


def test_tensor_indices_match_shifted_modulo() -> None:
    plan = SubcarrierPlan()
    indices = plan.tensor_indices("data", torch.device("cpu"))
    assert indices.tolist() == [index % 64 for index in plan.data_shifted]


def test_default_occupancy_scales_with_fft_size() -> None:
    plan = SubcarrierPlan(n_fft=128)
    assert plan.n_occupied == 104
    assert plan.n_data == 100
    assert plan.n_pilot == 4
    assert plan.pilot_shifted == (-42, -14, 14, 42)
    assert max(plan.occupied_shifted) == 52
    assert min(plan.occupied_shifted) == -52


def test_exact_occupied_count_can_be_configured() -> None:
    plan = SubcarrierPlan(n_fft=128, n_occupied=100)
    assert len(plan.occupied_shifted) == 100
    assert plan.n_data == 96
    assert [len(band) for band in plan.band_data_positions] == [24, 24, 24, 24]
