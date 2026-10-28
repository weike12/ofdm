import json
import math

import pytest
import torch

from si_fdjscc.evaluation.metrics import (
    MetricAccumulator,
    assigned_mapping_risk_metrics,
)


BANDS = ((0, 1), (2, 3))


def _batched_metrics(**kwargs):
    grid = torch.tensor(
        [[[0, 1, 2, 3], [4, 5, -1, -1]],
         [[4, 0, 2, -1], [5, 1, 3, -1]]]
    )
    return assigned_mapping_risk_metrics(
        grid,
        torch.tensor([0, 0, 1, 1, 2, 2]),
        torch.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]]),
        BANDS,
        **kwargs,
    )


def test_mapping_risk_metrics_ignore_padding_and_separate_batch_items():
    rows = _batched_metrics(spectral_notch={"start": 2, "width": 2})
    assert torch.equal(rows["mapping_symbol_count"], torch.tensor([6.0, 6.0], dtype=torch.float64))
    assert torch.allclose(rows["mapping_notch_fraction"], torch.full((2,), 1.0 / 3, dtype=torch.float64))
    assert torch.equal(rows["mapping_high_symbol_count"], torch.tensor([2.0, 2.0], dtype=torch.float64))
    assert torch.equal(rows["mapping_high_notch_fraction"], torch.zeros(2, dtype=torch.float64))
    assert torch.equal(rows["mapping_mid_notch_fraction"], torch.ones(2, dtype=torch.float64))
    assert torch.equal(rows["mapping_low_notch_fraction"], torch.zeros(2, dtype=torch.float64))
    assert torch.allclose(
        rows["mapping_band_0_fraction"] + rows["mapping_band_1_fraction"],
        torch.ones(2, dtype=torch.float64),
    )


def test_notch_absent_means_zero_exposure_for_nonempty_groups():
    rows = _batched_metrics()
    for group in ("", "high_", "mid_", "low_"):
        assert torch.equal(rows[f"mapping_{group}notch_fraction"], torch.zeros(2, dtype=torch.float64))


def test_ties_use_mapper_high_first_grouping_and_empty_groups_are_nan():
    rows = assigned_mapping_risk_metrics(
        torch.tensor([[[0, 1, 2, 3]]]),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([[3.0, 3.0, 1.0]]),
        BANDS,
    )
    assert rows["mapping_high_symbol_count"].item() == 4
    assert rows["mapping_mid_symbol_count"].item() == 0
    assert rows["mapping_low_symbol_count"].item() == 0
    for group in ("mid", "low"):
        for suffix in ("notch_fraction", "max_band_fraction", "token_band_count_mean"):
            assert math.isnan(rows[f"mapping_{group}_{suffix}"].item())


def test_group_band_coverage_is_not_mistaken_for_token_diversity():
    rows = assigned_mapping_risk_metrics(
        torch.tensor([[[0, 1, 2, 3]]]),
        torch.tensor([0, 0, 1, 1]),
        torch.ones(1, 2),
        BANDS,
    )
    # The high group spans two bands, but each of its tokens uses only one.
    assert rows["mapping_high_max_band_fraction"].item() == 0.5
    assert rows["mapping_high_token_band_count_mean"].item() == 1.0
    assert rows["mapping_high_token_max_band_fraction_mean"].item() == 1.0


def test_token_diversity_counts_each_tokens_actual_bands():
    rows = assigned_mapping_risk_metrics(
        torch.tensor([[[0, 2, 1, 3]]]),
        torch.tensor([0, 0, 1, 1]),
        torch.ones(1, 2),
        BANDS,
    )
    assert rows["mapping_high_token_band_count_mean"].item() == 2.0
    assert rows["mapping_high_token_max_band_fraction_mean"].item() == 0.5


def test_rejected_bands_and_spill_count_actual_symbols_not_padding():
    rows = _batched_metrics(
        risk_eligible_bands=torch.tensor([[True, False], [False, True]]),
        risk_band_scores=torch.tensor([[1.0, 0.1], [0.1, 1.0]]),
        band_reference_quality=torch.tensor([[1.0, 3.0, 2.0, 4.0], [2.0, 4.0, 4.0, 6.0]]),
    )
    assert torch.equal(rows["risk_rejected_band_count"], torch.ones(2, dtype=torch.float64))
    assert torch.allclose(rows["risk_spill_fraction"], torch.tensor([1.0 / 3, 2.0 / 3], dtype=torch.float64))
    assert torch.equal(rows["risk_high_spill_fraction"], torch.tensor([0.0, 1.0], dtype=torch.float64))
    assert torch.equal(rows["risk_mid_spill_fraction"], torch.tensor([1.0, 0.0], dtype=torch.float64))
    assert torch.equal(rows["risk_low_spill_fraction"], torch.tensor([0.0, 1.0], dtype=torch.float64))
    assert torch.equal(rows["mapping_band_0_reference_quality"], torch.tensor([2.0, 3.0], dtype=torch.float64))


def test_legacy_metadata_preserves_schema_without_fabricating_no_rejection():
    legacy = _batched_metrics()
    bandmean = _batched_metrics(risk_eligible_bands=torch.ones(2, 2, dtype=torch.bool))
    assert set(legacy) == set(bandmean)
    assert torch.isnan(legacy["risk_rejected_band_count"]).all()
    assert torch.isnan(legacy["risk_spill_fraction"]).all()
    assert torch.equal(bandmean["risk_rejected_band_count"], torch.zeros(2, dtype=torch.float64))
    assert torch.equal(bandmean["risk_spill_fraction"], torch.zeros(2, dtype=torch.float64))


def test_batched_token_ids_are_resolved_per_sample():
    rows = assigned_mapping_risk_metrics(
        torch.tensor([[[0, 1, -1, -1]], [[0, 1, -1, -1]]]),
        torch.tensor([[0, 0], [1, 1]]),
        torch.tensor([[2.0, 1.0], [2.0, 1.0]]),
        BANDS,
    )
    assert rows["mapping_high_symbol_count"].tolist() == [2.0, 0.0]
    assert rows["mapping_low_symbol_count"].tolist() == [0.0, 2.0]
    assert math.isnan(rows["mapping_high_notch_fraction"][1].item())


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"band_data_positions": ((0, 1), (1, 2))}, "partition"),
        ({"spectral_notch": {"start": 3, "width": 2}}, "outside"),
        ({"risk_eligible_bands": torch.ones(1, 2)}, "boolean"),
        ({"risk_band_scores": torch.ones(1, 3)}, "shape"),
        ({"symbol_index_grid": torch.tensor([[[0, 1, 2, 4]]])}, "source index"),
    ],
)
def test_mapping_risk_metrics_reject_invalid_inputs(kwargs, error):
    arguments = {
        "symbol_index_grid": torch.tensor([[[0, 1, 2, 3]]]),
        "token_ids": torch.tensor([0, 0, 1, 1]),
        "sensitivity": torch.ones(1, 2),
        "band_data_positions": BANDS,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=error):
        assigned_mapping_risk_metrics(**arguments)


def test_accumulator_summarizes_scalar_diagnostics_including_empty_groups():
    metrics = _batched_metrics()
    metrics["mapping_mid_notch_fraction"] = torch.full((2,), float("nan"))
    accumulator = MetricAccumulator(compute_msssim=False, compute_lpips=False)
    target = torch.zeros(2, 3, 2, 2)
    accumulator.add(torch.full_like(target, 0.1), target, ["first", "second"], metadata=metrics)
    summary = accumulator.summary()
    assert summary["mapping_symbol_count_mean"] == 6.0
    assert summary["mapping_mid_notch_fraction_mean"] is None
    assert summary["risk_rejected_band_count_mean"] is None
    json.dumps(summary, allow_nan=False)
    assert all(isinstance(value, float) for row in accumulator.rows for key, value in row.items() if key.startswith(("mapping_", "risk_")))
