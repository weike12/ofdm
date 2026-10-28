from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Mapping, Optional


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    until_step: int
    channel: Dict[str, Any]
    n_channel_draws: int
    loss_scales: Dict[str, float]


class CurriculumSchedule:
    """Resolve a monotonic Stage C channel/loss curriculum by global step."""

    def __init__(
        self,
        base_channel: Mapping[str, Any],
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.base_channel = dict(base_channel)
        settings = dict(config or {})
        self.enabled = bool(settings.get("enabled", bool(settings.get("phases"))))
        raw_phases = list(settings.get("phases", []))
        self.phases = []
        previous_until = 0
        for index, raw in enumerate(raw_phases):
            phase = dict(raw)
            until_step = int(phase.get("until_step", 0))
            if until_step <= previous_until:
                raise ValueError("curriculum until_step values must be strictly increasing")
            n_draws = int(
                phase.get(
                    "n_channel_draws",
                    self.base_channel.get("n_channel_draws", 1),
                )
            )
            if n_draws < 1:
                raise ValueError("curriculum n_channel_draws must be positive")
            scales = {
                "importance": 1.0,
                "latent": 1.0,
                "tail": 1.0,
                **{
                    key: float(value)
                    for key, value in dict(phase.get("loss_scales", {})).items()
                },
            }
            if any(value < 0 for value in scales.values()):
                raise ValueError("curriculum loss scales must be non-negative")
            self.phases.append(
                CurriculumPhase(
                    name=str(phase.get("name", f"phase_{index + 1}")),
                    until_step=until_step,
                    channel={
                        **self.base_channel,
                        **dict(phase.get("channel", {})),
                    },
                    n_channel_draws=n_draws,
                    loss_scales=scales,
                )
            )
            previous_until = until_step
        if self.enabled and not self.phases:
            raise ValueError("enabled curriculum requires at least one phase")

    def at(self, step: int) -> CurriculumPhase:
        if step < 1:
            raise ValueError("curriculum step must be at least one")
        if not self.enabled:
            return CurriculumPhase(
                name="default",
                until_step=step,
                channel=dict(self.base_channel),
                n_channel_draws=int(self.base_channel.get("n_channel_draws", 1)),
                loss_scales={"importance": 1.0, "latent": 1.0, "tail": 1.0},
            )
        for phase in self.phases:
            if step <= phase.until_step:
                return phase
        return self.phases[-1]


class DivergenceGuard:
    """Detect codec drift, image-loss growth, or implausible decoder outputs."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        settings = dict(config or {})
        self.enabled = bool(settings.get("enabled", True))
        self.window_size = int(settings.get("window_size", 100))
        self.check_every = int(settings.get("check_every", 50))
        self.warmup_steps = int(settings.get("warmup_steps", 100))
        self.growth_factor = float(settings.get("growth_factor", 2.5))
        self.absolute_threshold = float(settings.get("absolute_threshold", 0.25))
        self.anchor_growth_factor = float(
            settings.get("anchor_growth_factor", self.growth_factor)
        )
        self.anchor_absolute_threshold = float(
            settings.get("anchor_absolute_threshold", 0.02)
        )
        self.patience = int(settings.get("patience", 3))
        self.raw_abs_limit = float(settings.get("raw_abs_limit", 5.0))
        if self.window_size < 1 or self.check_every < 1 or self.patience < 1:
            raise ValueError("divergence guard window/check/patience must be positive")
        if (
            self.growth_factor <= 1.0
            or self.absolute_threshold <= 0
            or self.anchor_growth_factor <= 1.0
            or self.anchor_absolute_threshold <= 0
        ):
            raise ValueError("divergence guard thresholds are invalid")
        self.history: Deque[float] = deque(maxlen=self.window_size)
        self.anchor_history: Deque[float] = deque(maxlen=self.window_size)
        self.best_moving_mean: Optional[float] = None
        self.last_moving_mean: Optional[float] = None
        self.best_anchor_mean: Optional[float] = None
        self.last_anchor_mean: Optional[float] = None
        self.bad_checks = 0
        self.anchor_bad_checks = 0
        self.phase: Optional[str] = None

    def _reset_phase(self, phase: str) -> None:
        # Channel difficulty legitimately changes at a curriculum boundary, so
        # compare each phase to its own baseline instead of the warm-up minimum.
        self.phase = phase
        self.history.clear()
        self.anchor_history.clear()
        self.best_moving_mean = None
        self.last_moving_mean = None
        self.best_anchor_mean = None
        self.last_anchor_mean = None
        self.bad_checks = 0
        self.anchor_bad_checks = 0

    def update(
        self,
        step: int,
        image_loss: float,
        raw_min: float,
        raw_max: float,
        anchor_loss: Optional[float] = None,
        phase: Optional[str] = None,
    ) -> Optional[str]:
        values = (image_loss, raw_min, raw_max)
        if anchor_loss is not None:
            values = (*values, anchor_loss)
        if not all(math.isfinite(value) for value in values):
            return "non-finite loss or reconstruction statistic"
        if not self.enabled:
            return None
        phase_name = str(phase) if phase is not None else "default"
        if self.phase != phase_name:
            self._reset_phase(phase_name)
        if max(abs(raw_min), abs(raw_max)) > self.raw_abs_limit:
            return (
                f"raw reconstruction exceeded ±{self.raw_abs_limit:g}: "
                f"min={raw_min:.6g}, max={raw_max:.6g}"
            )
        self.history.append(float(image_loss))
        if anchor_loss is not None:
            self.anchor_history.append(float(anchor_loss))
        if (
            step < self.warmup_steps
            or len(self.history) < self.window_size
            or step % self.check_every != 0
        ):
            return None

        moving_mean = sum(self.history) / len(self.history)
        self.last_moving_mean = moving_mean
        if self.best_moving_mean is None or moving_mean < self.best_moving_mean:
            self.best_moving_mean = moving_mean
            self.bad_checks = 0
        image_threshold = min(
            self.absolute_threshold,
            self.best_moving_mean * self.growth_factor,
        )
        if moving_mean > image_threshold:
            self.bad_checks += 1
        else:
            self.bad_checks = 0
        if self.bad_checks >= self.patience:
            return (
                "moving image loss diverged: "
                f"current={moving_mean:.6g}, best={self.best_moving_mean:.6g}, "
                f"threshold={image_threshold:.6g}, bad_checks={self.bad_checks}"
            )

        if len(self.anchor_history) == self.window_size:
            anchor_mean = sum(self.anchor_history) / len(self.anchor_history)
            self.last_anchor_mean = anchor_mean
            if self.best_anchor_mean is None or anchor_mean < self.best_anchor_mean:
                self.best_anchor_mean = anchor_mean
                self.anchor_bad_checks = 0
            anchor_threshold = min(
                self.anchor_absolute_threshold,
                self.best_anchor_mean * self.anchor_growth_factor,
            )
            if anchor_mean > anchor_threshold:
                self.anchor_bad_checks += 1
            else:
                self.anchor_bad_checks = 0
            if self.anchor_bad_checks >= self.patience:
                return (
                    "clean reconstruction anchor diverged: "
                    f"current={anchor_mean:.6g}, best={self.best_anchor_mean:.6g}, "
                    f"threshold={anchor_threshold:.6g}, "
                    f"bad_checks={self.anchor_bad_checks}"
                )
        return None

    def metrics(self) -> Dict[str, Any]:
        return {
            "guard_moving_image_loss": self.last_moving_mean,
            "guard_best_image_loss": self.best_moving_mean,
            "guard_bad_checks": self.bad_checks,
            "guard_moving_anchor_loss": self.last_anchor_mean,
            "guard_best_anchor_loss": self.best_anchor_mean,
            "guard_anchor_bad_checks": self.anchor_bad_checks,
            "guard_phase": self.phase,
        }


class AmpOverflowGuard:
    """Allow transient GradScaler overflows but stop persistent bad gradients."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        settings = dict(config or {})
        self.enabled = bool(settings.get("enabled", True))
        self.max_consecutive = int(settings.get("max_consecutive", 8))
        self.window_size = int(settings.get("window_size", 100))
        self.max_in_window = int(settings.get("max_in_window", 20))
        self.min_observations = int(
            settings.get("min_observations", min(20, self.window_size))
        )
        self.max_fraction = float(
            settings.get(
                "max_fraction", self.max_in_window / float(self.window_size)
            )
        )
        if (
            self.max_consecutive < 1
            or self.window_size < 1
            or self.max_in_window < 1
            or self.max_in_window > self.window_size
            or self.min_observations < 1
            or self.min_observations > self.window_size
            or not 0.0 < self.max_fraction <= 1.0
        ):
            raise ValueError("AMP overflow guard settings are invalid")
        self.history: Deque[bool] = deque(maxlen=self.window_size)
        self.consecutive = 0
        self.total = 0

    def update(self, optimizer_step_skipped: bool) -> Optional[str]:
        skipped = bool(optimizer_step_skipped)
        self.history.append(skipped)
        if skipped:
            self.consecutive += 1
            self.total += 1
        else:
            self.consecutive = 0
        if not self.enabled:
            return None
        if self.consecutive >= self.max_consecutive:
            return (
                "persistent non-finite gradients: "
                f"{self.consecutive} consecutive optimizer steps skipped"
            )
        overflow_count = sum(self.history)
        if len(self.history) >= self.min_observations and (
            overflow_count / len(self.history) >= self.max_fraction
        ):
            return (
                "frequent non-finite gradients: "
                f"{overflow_count}/{len(self.history)} optimizer steps skipped"
            )
        return None

    def metrics(self) -> Dict[str, Any]:
        return {
            "amp_overflow_consecutive": self.consecutive,
            "amp_overflow_in_window": sum(self.history),
            "amp_overflow_rate": (
                sum(self.history) / len(self.history) if self.history else 0.0
            ),
            "amp_overflow_total": self.total,
        }
