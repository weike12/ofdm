import pytest

from si_fdjscc.train.stage_c_stability import (
    AmpOverflowGuard,
    CurriculumSchedule,
    DivergenceGuard,
)


def _schedule() -> CurriculumSchedule:
    return CurriculumSchedule(
        {
            "snr_db_range": [0, 20],
            "csi_modes": ["perfect", "imperfect"],
            "n_channel_draws": 4,
        },
        {
            "enabled": True,
            "phases": [
                {
                    "name": "warmup",
                    "until_step": 2000,
                    "channel": {"snr_db_range": [10, 20], "csi_modes": ["perfect"]},
                    "n_channel_draws": 1,
                    "loss_scales": {"latent": 0.0, "tail": 0.0},
                },
                {
                    "name": "bridge",
                    "until_step": 16000,
                    "channel": {"snr_db_range": [0, 20]},
                    "n_channel_draws": 2,
                    "loss_scales": {"latent": 0.5, "tail": 0.0},
                },
                {
                    "name": "transition",
                    "until_step": 30000,
                    "channel": {"snr_db_range": [0, 20]},
                    "n_channel_draws": 4,
                    "loss_scales": {"latent": 0.75, "tail": 0.75},
                },
                {
                    "name": "robust",
                    "until_step": 80000,
                    "n_channel_draws": 4,
                },
            ],
        },
    )


def test_curriculum_changes_only_at_phase_boundaries() -> None:
    schedule = _schedule()
    assert schedule.at(1).name == "warmup"
    assert schedule.at(2000).n_channel_draws == 1
    assert schedule.at(2001).name == "bridge"
    assert schedule.at(2001).n_channel_draws == 2
    assert schedule.at(16001).name == "transition"
    assert schedule.at(16001).n_channel_draws == 4
    assert schedule.at(30001).name == "robust"
    assert schedule.at(30001).n_channel_draws == 4
    assert schedule.at(30001).channel["snr_db_range"] == [0, 20]
    assert all(
        phase.n_channel_draws >= 4
        for phase in schedule.phases
        if phase.loss_scales["tail"] > 0
    )


def test_curriculum_rejects_non_monotonic_boundaries() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        CurriculumSchedule(
            {},
            {
                "enabled": True,
                "phases": [
                    {"until_step": 100},
                    {"until_step": 100},
                ],
            },
        )


def test_divergence_guard_stops_sustained_growth() -> None:
    guard = DivergenceGuard(
        {
            "window_size": 3,
            "check_every": 1,
            "warmup_steps": 3,
            "growth_factor": 2.0,
            "absolute_threshold": 0.1,
            "patience": 2,
            "raw_abs_limit": 5.0,
        }
    )
    assert guard.update(1, 0.01, 0.0, 1.0) is None
    assert guard.update(2, 0.01, 0.0, 1.0) is None
    assert guard.update(3, 0.01, 0.0, 1.0) is None
    assert guard.update(4, 0.30, 0.0, 1.0) is None
    reason = guard.update(5, 0.30, 0.0, 1.0)
    assert reason is not None
    assert "moving image loss diverged" in reason


def test_divergence_guard_stops_implausible_raw_range() -> None:
    guard = DivergenceGuard({"raw_abs_limit": 5.0})
    reason = guard.update(1, 0.01, -0.1, 5.1)
    assert reason is not None
    assert "exceeded" in reason


def test_divergence_guard_stops_clean_anchor_drift() -> None:
    guard = DivergenceGuard(
        {
            "window_size": 2,
            "check_every": 1,
            "warmup_steps": 1,
            "growth_factor": 100.0,
            "absolute_threshold": 1.0,
            "anchor_growth_factor": 2.0,
            "anchor_absolute_threshold": 0.02,
            "patience": 2,
        }
    )
    assert guard.update(1, 0.01, 0.0, 1.0, 0.001, "bridge") is None
    assert guard.update(2, 0.01, 0.0, 1.0, 0.001, "bridge") is None
    assert guard.update(3, 0.01, 0.0, 1.0, 0.010, "bridge") is None
    reason = guard.update(4, 0.01, 0.0, 1.0, 0.010, "bridge")
    assert reason is not None
    assert "clean reconstruction anchor diverged" in reason


def test_divergence_guard_resets_baseline_at_curriculum_boundary() -> None:
    guard = DivergenceGuard(
        {"window_size": 2, "check_every": 1, "warmup_steps": 1, "patience": 1}
    )
    guard.update(1, 0.001, 0.0, 1.0, 0.001, "warmup")
    guard.update(2, 0.001, 0.0, 1.0, 0.001, "warmup")
    assert guard.update(3, 0.05, 0.0, 1.0, 0.001, "robust") is None
    assert guard.metrics()["guard_phase"] == "robust"
    assert guard.metrics()["guard_best_image_loss"] is None


def test_amp_overflow_guard_tolerates_isolated_overflow() -> None:
    guard = AmpOverflowGuard(
        {"max_consecutive": 3, "window_size": 5, "max_in_window": 3}
    )
    assert guard.update(True) is None
    assert guard.metrics()["amp_overflow_consecutive"] == 1
    assert guard.update(False) is None
    assert guard.metrics()["amp_overflow_consecutive"] == 0
    assert guard.metrics()["amp_overflow_total"] == 1


def test_amp_overflow_guard_stops_persistent_overflow() -> None:
    guard = AmpOverflowGuard(
        {"max_consecutive": 3, "window_size": 10, "max_in_window": 8}
    )
    assert guard.update(True) is None
    assert guard.update(True) is None
    reason = guard.update(True)
    assert reason is not None
    assert "3 consecutive" in reason


def test_amp_overflow_guard_stops_frequent_overflow() -> None:
    guard = AmpOverflowGuard(
        {"max_consecutive": 3, "window_size": 5, "max_in_window": 3}
    )
    reasons = [guard.update(value) for value in [True, False, True, False, True]]
    assert reasons[-1] is not None
    assert "3/5" in reasons[-1]


def test_amp_overflow_guard_catches_early_overflow_burst() -> None:
    guard = AmpOverflowGuard(
        {
            "max_consecutive": 8,
            "window_size": 100,
            "max_in_window": 20,
            "min_observations": 20,
            "max_fraction": 0.20,
        }
    )
    pattern = [True, False, True, False, True, False, False, True]
    pattern += [False] * 12
    reasons = [guard.update(value) for value in pattern]
    assert reasons[-1] is not None
    assert "4/20" in reasons[-1]
