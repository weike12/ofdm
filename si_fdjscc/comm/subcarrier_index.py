from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch


_REFERENCE_N_FFT = 64
_REFERENCE_N_OCCUPIED = 52


@dataclass(frozen=True)
class SubcarrierPlan:
    """Frequency-ordered occupied, pilot, and data carrier definitions.

    Public carrier numbers use the fftshift convention ``[-N/2, N/2-1]``.
    Tensor indices use PyTorch's unshifted FFT order and are obtained with
    ``shifted_index % n_fft``.

    If ``occupied_shifted`` is omitted, the 52/64 reference occupancy is
    scaled to ``n_fft`` and adjusted to keep the data carriers evenly
    divisible by ``n_bands``. ``n_occupied`` can be supplied to request an
    exact contiguous occupied-carrier count. The default remains the usual
    64-point, 52-occupied, four-pilot plan.
    """

    n_fft: int = 64
    occupied_shifted: Optional[Tuple[int, ...]] = None
    pilot_shifted: Optional[Tuple[int, ...]] = None
    n_bands: int = 4
    n_occupied: Optional[int] = None

    @staticmethod
    def _shifted_bounds(n_fft: int) -> tuple[int, int]:
        return -(n_fft // 2), (n_fft - 1) // 2

    @classmethod
    def _centered_occupied(cls, n_fft: int, count: int) -> Tuple[int, ...]:
        """Build ``count`` contiguous non-DC carriers around zero."""
        lower, upper = cls._shifted_bounds(n_fft)
        negative_capacity = -lower
        positive_capacity = upper
        n_negative = count // 2
        n_positive = count - n_negative
        if n_positive > positive_capacity:
            overflow = n_positive - positive_capacity
            n_positive -= overflow
            n_negative += overflow
        if n_negative > negative_capacity:
            overflow = n_negative - negative_capacity
            n_negative -= overflow
            n_positive += overflow
        if n_negative > negative_capacity or n_positive > positive_capacity:
            raise ValueError("n_occupied exceeds the number of non-DC FFT carriers")
        return tuple(range(-n_negative, 0)) + tuple(range(1, n_positive + 1))

    @classmethod
    def _scaled_default_pilots(cls, n_fft: int) -> Tuple[int, ...]:
        lower, upper = cls._shifted_bounds(n_fft)
        if upper < 2:
            raise ValueError("n_fft is too small for the default four-pilot plan")
        magnitudes = []
        for reference in (7, 21):
            scaled = int(round(reference * n_fft / _REFERENCE_N_FFT))
            magnitudes.append(min(max(scaled, 1), upper))
        if magnitudes[0] == magnitudes[1]:
            alternatives = sorted(
                range(1, upper + 1),
                key=lambda value: (abs(value - magnitudes[1]), -value),
            )
            magnitudes[1] = next(
                (value for value in alternatives if value != magnitudes[0]),
                magnitudes[1],
            )
        inner, outer = sorted(magnitudes)
        pilots = (-outer, -inner, inner, outer)
        if any(index < lower or index > upper for index in pilots) or len(set(pilots)) != 4:
            raise ValueError("n_fft is too small for the default four-pilot plan")
        return pilots

    @classmethod
    def _auto_occupied(
        cls,
        n_fft: int,
        pilots: Tuple[int, ...],
        n_bands: int,
    ) -> Tuple[int, ...]:
        target = int(round(n_fft * _REFERENCE_N_OCCUPIED / _REFERENCE_N_FFT))
        pilot_set = set(pilots)
        candidates = []
        for count in range(len(pilots), n_fft):
            if (count - len(pilots)) % n_bands:
                continue
            occupied = cls._centered_occupied(n_fft, count)
            if pilot_set.issubset(occupied):
                candidates.append(occupied)
        if not candidates:
            raise ValueError(
                "No scalable occupied-carrier count is compatible with the pilots and n_bands"
            )
        return min(candidates, key=lambda values: (abs(len(values) - target), -len(values)))

    def __post_init__(self) -> None:
        if isinstance(self.n_fft, bool) or int(self.n_fft) != self.n_fft or self.n_fft < 2:
            raise ValueError("n_fft must be an integer greater than one")
        if isinstance(self.n_bands, bool) or int(self.n_bands) != self.n_bands or self.n_bands < 1:
            raise ValueError("n_bands must be a positive integer")
        object.__setattr__(self, "n_fft", int(self.n_fft))
        object.__setattr__(self, "n_bands", int(self.n_bands))

        if self.pilot_shifted is None:
            pilots = self._scaled_default_pilots(self.n_fft)
        else:
            pilots = tuple(sorted(int(index) for index in self.pilot_shifted))
        if len(set(pilots)) != len(pilots):
            raise ValueError("pilot_shifted contains duplicates")
        if 0 in pilots:
            raise ValueError("DC carrier cannot be a pilot")

        if self.occupied_shifted is None:
            if self.n_occupied is None:
                occupied = self._auto_occupied(self.n_fft, pilots, self.n_bands)
            else:
                if isinstance(self.n_occupied, bool) or int(self.n_occupied) != self.n_occupied:
                    raise ValueError("n_occupied must be an integer")
                occupied = self._centered_occupied(self.n_fft, int(self.n_occupied))
        else:
            occupied = tuple(sorted(int(index) for index in self.occupied_shifted))
            if self.n_occupied is not None and int(self.n_occupied) != len(occupied):
                raise ValueError("n_occupied must match len(occupied_shifted)")

        object.__setattr__(self, "occupied_shifted", occupied)
        object.__setattr__(self, "pilot_shifted", pilots)
        object.__setattr__(self, "n_occupied", len(occupied))
        if len(set(occupied)) != len(occupied):
            raise ValueError("occupied_shifted contains duplicates")
        if 0 in occupied:
            raise ValueError("DC carrier must remain unused")
        if not set(pilots).issubset(occupied):
            raise ValueError("Every pilot must be an occupied carrier")
        lower, upper = self._shifted_bounds(self.n_fft)
        if any(index < lower or index > upper for index in occupied):
            raise ValueError("A shifted carrier index is outside the FFT range")
        if self.n_data <= 0:
            raise ValueError("The plan must contain at least one data carrier")
        if self.n_data % self.n_bands != 0:
            raise ValueError("Data carriers must split evenly across frequency bands")

    @property
    def data_shifted(self) -> Tuple[int, ...]:
        pilot_set = set(self.pilot_shifted or ())
        return tuple(index for index in self.occupied_shifted or () if index not in pilot_set)

    @property
    def n_data(self) -> int:
        return len(self.occupied_shifted or ()) - len(self.pilot_shifted or ())

    @property
    def n_pilot(self) -> int:
        return len(self.pilot_shifted or ())

    @staticmethod
    def shifted_to_fft(indices: Sequence[int], n_fft: int) -> Tuple[int, ...]:
        return tuple(int(index) % n_fft for index in indices)

    @property
    def data_fft(self) -> Tuple[int, ...]:
        return self.shifted_to_fft(self.data_shifted, self.n_fft)

    @property
    def pilot_fft(self) -> Tuple[int, ...]:
        return self.shifted_to_fft(self.pilot_shifted or (), self.n_fft)

    @property
    def occupied_fft(self) -> Tuple[int, ...]:
        return self.shifted_to_fft(self.occupied_shifted or (), self.n_fft)

    @property
    def band_data_positions(self) -> Tuple[Tuple[int, ...], ...]:
        """Return positions in the frequency-ordered ``data_shifted`` list."""
        width = self.n_data // self.n_bands
        return tuple(
            tuple(range(band * width, (band + 1) * width)) for band in range(self.n_bands)
        )

    def tensor_indices(self, kind: str, device: torch.device) -> torch.Tensor:
        values = {
            "data": self.data_fft,
            "pilot": self.pilot_fft,
            "occupied": self.occupied_fft,
        }.get(kind)
        if values is None:
            raise ValueError("kind must be one of: data, pilot, occupied")
        return torch.tensor(values, dtype=torch.long, device=device)
