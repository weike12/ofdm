import json

import pytest
import torch

from si_fdjscc.evaluation.evaluate import evaluate
from si_fdjscc.evaluation.metrics import (
    ENERGY_DIAGNOSTIC_FIELDS,
    SYMBOL_RESIDUAL_FIELDS,
    MetricAccumulator,
    energy_diagnostic_metrics,
    symbol_residual_metrics,
)


def _example(
    *,
    scores=None,
    symbols_tx=None,
    symbols_rx=None,
    h_hat=None,
    receiver_h=None,
    true_h=None,
):
    tx = symbols_tx if symbols_tx is not None else torch.tensor(
        [[1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j]], dtype=torch.complex64
    )
    rx = symbols_rx if symbols_rx is not None else tx + torch.tensor(
        [[1j, 2j, 3j, 4j]], dtype=torch.complex64
    )
    channel = h_hat if h_hat is not None else torch.tensor(
        [[[0j, 1 + 0j, 0j, 3 + 0j],
          [0j, 1 + 0j, 0j, 3 + 0j]]], dtype=torch.complex64
    )
    return symbol_residual_metrics(
        tx,
        rx,
        torch.tensor([0, 0, 1, 1]),
        scores if scores is not None else torch.tensor([[0.75, 0.25]]),
        torch.tensor([[[0, 1], [2, 3]]]),
        channel,
        receiver_h if receiver_h is not None else channel,
        true_h if true_h is not None else channel,
        torch.tensor([1.0]),
        torch.tensor([1, 3]),
    )


def test_symbol_residuals_have_measured_mse_and_proxy_in_distinct_units():
    metrics = _example()
    assert set(metrics) == set(SYMBOL_RESIDUAL_FIELDS)
    assert metrics["residual_symbol_mse"].item() == pytest.approx(7.5)
    # The two within-token MSE values are 2.5 and 12.5.
    assert metrics["residual_sds_weighted_token_mse"].item() == pytest.approx(5.0)
    # b_carrier=(1/(1+|H|^2))^2 gives .25 and .01; symbol powers are 1,4,9,16.
    assert metrics["residual_estimated_energy_proxy"].item() == pytest.approx(2.7)
    assert metrics["residual_mapper_csi_nmse"].item() == pytest.approx(0.0)
    assert metrics["residual_receiver_csi_nmse"].item() == pytest.approx(0.0)


def test_proxy_complex_averages_estimated_h_and_truth_is_reporting_only():
    tx = torch.ones(1, 4, dtype=torch.complex64)
    estimated = torch.tensor(
        [[[0j, 1 + 0j, 0j, 1 + 0j],
          [0j, -1 + 0j, 0j, 1 + 0j]]], dtype=torch.complex64
    )
    truth = torch.tensor(
        [[[0j, 1 + 0j, 0j, 1 + 0j],
          [0j, 1 + 0j, 0j, 1 + 0j]]], dtype=torch.complex64
    )
    receiver = torch.tensor([[[0j, 0j, 0j, 1 + 0j]]], dtype=torch.complex64)
    first = _example(symbols_tx=tx, symbols_rx=tx, h_hat=estimated, receiver_h=receiver, true_h=truth)
    different_truth = _example(
        symbols_tx=tx, symbols_rx=tx, h_hat=estimated, receiver_h=receiver,
        true_h=truth * 2,
    )
    # Mean complex H on carrier 1 is zero: its factor is 1, not .25.
    assert first["residual_estimated_energy_proxy"].item() == pytest.approx(2.5)
    assert first["residual_mapper_csi_nmse"].item() == pytest.approx(1.0)
    assert first["residual_receiver_csi_nmse"].item() == pytest.approx(0.5)
    for field in (
        "residual_symbol_mse", "residual_sds_weighted_token_mse",
        "residual_estimated_energy_proxy",
    ):
        assert torch.equal(first[field], different_truth[field])
    assert not torch.equal(first["residual_mapper_csi_nmse"], different_truth["residual_mapper_csi_nmse"])


def test_present_token_uniform_fallback_and_exact_complex_symbol_input():
    tx = torch.tensor([[1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j]], dtype=torch.complex64)
    rx = tx + torch.tensor([[1 + 0j, 1 + 0j, 3 + 0j, 3 + 0j]], dtype=torch.complex64)
    channel = torch.ones(1, 2, 4, dtype=torch.complex64)
    metrics = symbol_residual_metrics(
        tx, rx, torch.tensor([0, 0, 2, 2]), torch.zeros(1, 3),
        torch.tensor([[[0, 1], [2, 3]]]),
        channel, channel, channel, torch.tensor([1.0]), torch.tensor([1, 3]),
    )
    assert metrics["residual_symbol_mse"].item() == pytest.approx(5.0)
    assert metrics["residual_sds_weighted_token_mse"].item() == pytest.approx(5.0)


def test_residual_includes_recovered_final_imaginary_component_and_ignores_padding():
    # ComplexAdapter.inverse drops a padded final imaginary scalar. The direct
    # symbols_rx tensor retains the actual equalized error on that component.
    tx = torch.tensor([[1 + 2j, 3 + 0j]], dtype=torch.complex64)
    rx = torch.tensor([[1 + 2j, 3 + 5j]], dtype=torch.complex64)
    channel = torch.ones(1, 1, 3, dtype=torch.complex64)
    metrics = symbol_residual_metrics(
        tx, rx, torch.tensor([0, 1]), torch.ones(1, 2),
        torch.tensor([[[0, 1, -1]]]),
        channel, channel, channel, torch.tensor([1.0]), torch.tensor([0, 1, 2]),
    )
    assert metrics["residual_symbol_mse"].item() == pytest.approx(12.5)
    assert metrics["residual_sds_weighted_token_mse"].item() == pytest.approx(12.5)
    assert metrics["residual_estimated_energy_proxy"].item() == pytest.approx(3.5)


def test_residual_metrics_reject_duplicate_mapping_and_do_not_backpropagate():
    tx = torch.ones(1, 4, dtype=torch.complex64, requires_grad=True)
    rx = torch.zeros(1, 4, dtype=torch.complex64, requires_grad=True)
    metrics = _example(symbols_tx=tx, symbols_rx=rx)
    assert not any(value.requires_grad for value in metrics.values())

    channel = torch.ones(1, 2, 4, dtype=torch.complex64)
    with pytest.raises(ValueError, match="every source symbol exactly once"):
        symbol_residual_metrics(
            tx, rx, torch.tensor([0, 0, 1, 1]), torch.ones(1, 2),
            torch.tensor([[[0, 1], [1, 3]]]),
            channel, channel, channel, torch.tensor([1.0]), torch.tensor([1, 3]),
        )


def test_energy_metadata_has_stable_legacy_schema_and_json_null_summary():
    reported = {
        "energy_proxy_initial": torch.tensor([10.0, 20.0]),
        "energy_proxy_final": torch.tensor([9.0, 19.0]),
        "energy_changed_symbol_fraction": torch.tensor([0.25, 0.50]),
        "energy_changed_symbol_count": torch.tensor([1.0, 2.0]),
    }
    new = energy_diagnostic_metrics(reported, 2, torch.device("cpu"))
    old = energy_diagnostic_metrics(None, 2, torch.device("cpu"))
    assert set(new) == set(old) == set(ENERGY_DIAGNOSTIC_FIELDS)
    assert all(torch.isnan(value).all() for value in old.values())
    assert all(torch.isfinite(value).all() for value in new.values())
    accumulator = MetricAccumulator(compute_msssim=False)
    target = torch.zeros(2, 3, 2, 2)
    residual = {key: value.expand(2) for key, value in _example().items()}
    reconstruction = torch.full_like(target, 0.1)
    accumulator.add(reconstruction, target, ["one", "two"], {**new, **residual})
    assert accumulator.summary()["energy_proxy_final_mean"] == pytest.approx(14.0)
    assert accumulator.summary()["residual_symbol_mse_mean"] == pytest.approx(7.5)
    json.dumps(accumulator.summary(), allow_nan=False)
    legacy = MetricAccumulator(compute_msssim=False)
    legacy.add(reconstruction, target, ["one", "two"], old)
    assert legacy.summary()["energy_proxy_final_mean"] is None
    json.dumps(legacy.summary(), allow_nan=False)


@pytest.mark.parametrize("bad, error, pattern", [
    ({"energy_proxy_initial": torch.ones(2)}, ValueError, "missing fields"),
    ({key: torch.ones(2, 1) for key in ENERGY_DIAGNOSTIC_FIELDS}, ValueError, "shape"),
    ({key: torch.tensor([0.0, float("nan")]) for key in ENERGY_DIAGNOSTIC_FIELDS}, ValueError, "finite"),
])
def test_energy_metadata_rejects_partial_or_non_scalar_fields(bad, error, pattern):
    with pytest.raises(error, match=pattern):
        energy_diagnostic_metrics(bad, 2, torch.device("cpu"))


def test_energy_mapper_source_guard_runs_before_output_creation(tmp_path):
    target = tmp_path / "should_not_exist"
    with pytest.raises(ValueError, match="requires evaluation.mapper_csi_source='estimated'"):
        evaluate({
            "evaluation": {
                "mapper_modes": ["greedy_token_energy"],
                "mapper_csi_source": "oracle",
                "output_dir": str(target),
            }
        })
    assert not target.exists()
