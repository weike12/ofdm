from __future__ import annotations

import math
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import nn

from .subcarrier_index import SubcarrierPlan


@dataclass
class PermutationMeta:
    """Exact slot-to-semantic-symbol assignment used for inverse mapping."""

    symbol_index_grid: torch.Tensor  # [B,T,N_data], -1 denotes OFDM padding
    n_symbols: int
    mode: str
    band_reference_quality: Optional[torch.Tensor] = None  # [B,N_data]
    risk_band_scores: Optional[torch.Tensor] = None  # [B,n_bands]
    risk_eligible_bands: Optional[torch.Tensor] = None  # bool [B,n_bands]
    swap_diagnostics: Optional[dict[str, torch.Tensor]] = None  # scalar tensors [B]
    energy_diagnostics: Optional[dict[str, torch.Tensor]] = None  # scalar tensors [B]

    @property
    def valid_mask(self) -> torch.Tensor:
        return self.symbol_index_grid >= 0


class SemanticFrequencyMapper(nn.Module):
    MODES = {
        "sequential",
        "random",
        "greedy",
        "uniform4",
        "ia_fdi",
        "uniform_lowcorr",
        "proposed_ufd",
        "proposed_ufd_bandmean",
        "proposed_ufd_risk",
        "proposed_ufd_swap",
        "greedy_token_random",
        "greedy_token_energy",
    }
    UNIFORM_LOWCORR_MODES = {"uniform4", "uniform_lowcorr"}
    IMPORTANCE_AWARE_MODES = {
        "ia_fdi", "proposed_ufd", "proposed_ufd_bandmean", "proposed_ufd_risk",
        "proposed_ufd_swap",
    }
    BAND_REFERENCE_MODES = {"proposed_ufd_bandmean", "proposed_ufd_risk"}

    def __init__(
        self,
        plan: Optional[SubcarrierPlan] = None,
        gamma: float = 1.0,
        high_quantile: float = 0.80,
        mid_quantile: float = 0.50,
        quality_weight: float = 1.0,
        correlation_weight: float = 1.0,
        importance_correlation_gain: float = 1.0,
        diversity_levels: Sequence[int] = (1, 2, 4),
        risk_band_threshold: float = 0.25,
        risk_min_bands: int = 2,
        swap_error_budget: float = 0.005,
        swap_max_swaps: int = 128,
        swap_candidates_per_symbol: int = 8,
    ) -> None:
        super().__init__()
        self.plan = plan or SubcarrierPlan()
        if not 0.0 <= mid_quantile <= high_quantile <= 1.0:
            raise ValueError("Expected 0 <= mid_quantile <= high_quantile <= 1")
        if not math.isfinite(quality_weight) or quality_weight < 0.0:
            raise ValueError("quality_weight must be finite and non-negative")
        if not math.isfinite(correlation_weight) or correlation_weight < 0.0:
            raise ValueError("correlation_weight must be finite and non-negative")
        if not math.isfinite(importance_correlation_gain) or importance_correlation_gain < 0.0:
            raise ValueError(
                "importance_correlation_gain must be finite and non-negative"
            )
        levels = tuple(int(level) for level in diversity_levels)
        if len(levels) != 3 or any(level < 1 for level in levels):
            raise ValueError("diversity_levels must contain three positive integers")
        if levels != tuple(sorted(levels)):
            raise ValueError("diversity_levels must be ordered low-to-high")
        if not math.isfinite(risk_band_threshold) or not 0.0 <= risk_band_threshold <= 1.0:
            raise ValueError("risk_band_threshold must be finite and between zero and one")
        if (
            isinstance(risk_min_bands, bool)
            or not isinstance(risk_min_bands, int)
            or risk_min_bands < 1
        ):
            raise ValueError("risk_min_bands must be a positive integer")
        self.gamma = float(gamma)
        self.high_quantile = float(high_quantile)
        self.mid_quantile = float(mid_quantile)
        self.quality_weight = float(quality_weight)
        self.correlation_weight = float(correlation_weight)
        self.importance_correlation_gain = float(importance_correlation_gain)
        self.diversity_levels = levels
        self.risk_band_threshold = float(risk_band_threshold)
        self.risk_min_bands = risk_min_bands
        if (
            isinstance(swap_error_budget, bool)
            or not math.isfinite(swap_error_budget)
            or not 0.0 <= swap_error_budget <= 1.0
        ):
            raise ValueError("swap_error_budget must be finite and between zero and one")
        if (
            isinstance(swap_max_swaps, bool)
            or not isinstance(swap_max_swaps, int)
            or swap_max_swaps < 0
        ):
            raise ValueError("swap_max_swaps must be a non-negative integer")
        if (
            isinstance(swap_candidates_per_symbol, bool)
            or not isinstance(swap_candidates_per_symbol, int)
            or swap_candidates_per_symbol < 1
        ):
            raise ValueError("swap_candidates_per_symbol must be a positive integer")
        self.swap_error_budget = float(swap_error_budget)
        self.swap_max_swaps = swap_max_swaps
        self.swap_candidates_per_symbol = swap_candidates_per_symbol

    def _quality(
        self,
        h_hat: Optional[torch.Tensor],
        batch: int,
        n_ofdm_symbols: int,
        device: torch.device,
        real_dtype: torch.dtype,
        noise_variance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if h_hat is None:
            return torch.zeros(
                batch, n_ofdm_symbols, self.plan.n_data, device=device, dtype=real_dtype
            )
        if h_hat.ndim == 2:
            h_hat = h_hat[:, None, :]
        if h_hat.shape[0] != batch or h_hat.shape[-1] != self.plan.n_fft:
            raise ValueError(
                f"h_hat must have shape [B,{self.plan.n_fft}] "
                f"or [B,T,{self.plan.n_fft}]"
            )
        if h_hat.shape[1] == 1:
            h_hat = h_hat.expand(batch, n_ofdm_symbols, self.plan.n_fft)
        elif h_hat.shape[1] != n_ofdm_symbols:
            raise ValueError("h_hat OFDM-symbol dimension does not match the mapper grid")
        data_indices = self.plan.tensor_indices("data", device)
        data_h = h_hat.index_select(-1, data_indices)
        quality = self.gamma * data_h.abs().square()
        if noise_variance is not None:
            variance = torch.as_tensor(
                noise_variance, device=device, dtype=real_dtype
            )
            if variance.ndim == 0:
                variance = variance.expand(batch)
            variance = variance.reshape(-1)
            if variance.numel() != batch:
                raise ValueError("noise_variance must be scalar or have one value per batch item")
            quality = quality / variance.clamp_min(
                torch.finfo(real_dtype).tiny
            ).view(batch, 1, 1)
        return quality

    @staticmethod
    def _token_ids_for_batch(token_ids: torch.Tensor, batch: int, n_symbols: int) -> torch.Tensor:
        if token_ids.ndim == 1:
            if token_ids.numel() != n_symbols:
                raise ValueError("token_ids length must match the semantic symbol count")
            token_ids = token_ids.view(1, -1).expand(batch, -1)
        if token_ids.shape != (batch, n_symbols):
            raise ValueError("token_ids must have shape [K] or [B,K]")
        return token_ids.to(dtype=torch.long)

    def _correlation_for_batch(
        self,
        correlation_matrix: Optional[torch.Tensor],
        batch: int,
        device: torch.device,
        real_dtype: torch.dtype,
    ) -> torch.Tensor:
        n_data = self.plan.n_data
        if correlation_matrix is None:
            return torch.eye(n_data, device=device, dtype=real_dtype).unsqueeze(0).expand(
                batch, -1, -1
            )
        correlation = torch.as_tensor(correlation_matrix, device=device)
        if correlation.ndim == 2:
            if correlation.shape != (n_data, n_data):
                raise ValueError(
                    f"correlation_matrix must have shape [{n_data},{n_data}] or "
                    f"[B,{n_data},{n_data}]"
                )
            correlation = correlation.unsqueeze(0).expand(batch, -1, -1)
        elif correlation.ndim == 3:
            if correlation.shape[1:] != (n_data, n_data) or correlation.shape[0] not in {
                1,
                batch,
            }:
                raise ValueError(
                    f"correlation_matrix must have shape [{n_data},{n_data}] or "
                    f"[B,{n_data},{n_data}]"
                )
            if correlation.shape[0] == 1:
                correlation = correlation.expand(batch, -1, -1)
        else:
            raise ValueError(
                f"correlation_matrix must have shape [{n_data},{n_data}] or "
                f"[B,{n_data},{n_data}]"
            )
        correlation = correlation.abs().to(dtype=real_dtype)
        if not bool(torch.isfinite(correlation).all().item()):
            raise ValueError("correlation_matrix must contain only finite values")
        # Empirical estimates can be slightly asymmetric. The pairwise penalty
        # should not depend on the ordering of two data carriers.
        return 0.5 * (correlation + correlation.transpose(-1, -2))

    @staticmethod
    def _assign_greedy(
        token_ids: torch.Tensor, importance: torch.Tensor, quality_grid: torch.Tensor
    ) -> torch.Tensor:
        """The established greedy order, including token and slot tie breaks."""
        token_values = token_ids.detach().to(device="cpu", dtype=torch.long).tolist()
        importance_values = importance.detach().to(device="cpu", dtype=torch.float64).tolist()
        quality_values = quality_grid.detach().to(device="cpu", dtype=torch.float64).flatten().tolist()
        token_order = sorted(
            range(len(importance_values)),
            key=lambda token: (-importance_values[token], token),
        )
        symbols_by_token: List[List[int]] = [[] for _ in importance_values]
        # Preserve legacy behavior for out-of-range IDs: they are not assigned.
        for symbol, token in enumerate(token_values):
            if 0 <= token < len(symbols_by_token):
                symbols_by_token[token].append(symbol)
        source_order = [symbol for token in token_order for symbol in symbols_by_token[token]]
        slots = sorted(range(len(quality_values)), key=lambda slot: (-quality_values[slot], slot))
        assignment = torch.full((len(quality_values),), -1, dtype=torch.long)
        for slot, source in zip(slots, source_order):
            assignment[slot] = source
        return assignment.to(token_ids.device)

    @staticmethod
    def _mapper_random_generator(
        mode: str, seed: Optional[int], sample: int
    ) -> torch.Generator:
        """Use mapper-only CPU RNG streams without advancing channel/global RNGs.

        Explicit seeds are separated by mode and batch position. CPU sampling
        also gives identical permutations when inputs move between CPU and GPU.
        Unseeded calls obtain entropy on this local generator only.
        """
        generator = torch.Generator(device="cpu")
        if seed is None:
            generator.seed()
        else:
            offsets = {"random": 104_729, "greedy_token_random": 130_363}
            generator.manual_seed(
                (int(seed) + offsets[mode] + sample * 1_000_003) % (2**63 - 1)
            )
        return generator

    def _assign_greedy_token_random(
        self,
        token_ids: torch.Tensor,
        importance: torch.Tensor,
        quality_grid: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Uniformly shuffle each token within its exact greedy slot set.

        Padding and token allocations remain unchanged. Identity permutations
        are valid draws; no energy proxy is consulted and no draw is rejected.
        """
        greedy = self._assign_greedy(token_ids, importance, quality_grid)
        original = greedy.detach().to(device="cpu", dtype=torch.long).tolist()
        assignment = original.copy()
        tokens = token_ids.detach().to(device="cpu", dtype=torch.long).tolist()
        slots_by_token: dict[int, list[int]] = {}
        for slot, source in enumerate(original):
            if source >= 0:
                slots_by_token.setdefault(tokens[source], []).append(slot)
        for token in sorted(slots_by_token):
            slots = slots_by_token[token]
            sources = [original[slot] for slot in slots]
            permutation = torch.randperm(len(sources), generator=generator).tolist()
            for slot, index in zip(slots, permutation):
                assignment[slot] = sources[index]
        return greedy.new_tensor(assignment)

    def _assign_greedy_token_energy(
        self,
        token_ids: torch.Tensor,
        importance: torch.Tensor,
        quality_grid: torch.Tensor,
        symbol_power: torch.Tensor,
        carrier_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Sort symbol power against estimated MMSE bias inside greedy token slots.

        A token keeps exactly its greedy slot multiset. Equal-bias slots keep
        their original symbols whenever those symbols remain in the optimal
        bias group, avoiding waveform changes with no proxy benefit.
        """
        greedy = self._assign_greedy(token_ids, importance, quality_grid)
        original = greedy.detach().to(device="cpu", dtype=torch.long).tolist()
        assignment = original.copy()
        tokens = token_ids.detach().to(device="cpu", dtype=torch.long).tolist()
        powers = symbol_power.detach().to(device="cpu", dtype=torch.float64).tolist()
        biases = carrier_bias.detach().to(device="cpu", dtype=torch.float64).tolist()
        slots_by_token: dict[int, list[int]] = {}
        for slot, source in enumerate(original):
            if source >= 0:
                slots_by_token.setdefault(tokens[source], []).append(slot)

        def proxy(values: list[int]) -> float:
            try:
                result = math.fsum(
                    powers[source] * biases[slot % self.plan.n_data]
                    for slot, source in enumerate(values)
                    if source >= 0
                )
            except OverflowError as error:
                raise ValueError("energy proxy exceeds finite precision") from error
            if not math.isfinite(result):
                raise ValueError("energy proxy must be finite")
            return result

        initial = proxy(original)
        for slots in slots_by_token.values():
            sources = [original[slot] for slot in slots]
            sources.sort(key=lambda source: (-powers[source], source))
            slots.sort(key=lambda slot: (biases[slot % self.plan.n_data], slot))
            offset = 0
            while offset < len(slots):
                bias = biases[slots[offset] % self.plan.n_data]
                stop = offset + 1
                while (
                    stop < len(slots)
                    and biases[slots[stop] % self.plan.n_data] == bias
                ):
                    stop += 1
                group_slots = slots[offset:stop]
                selected = sources[offset:stop]
                selected_set = set(selected)
                retained = {original[slot] for slot in group_slots if original[slot] in selected_set}
                incoming = iter(source for source in selected if source not in retained)
                for slot in group_slots:
                    if original[slot] not in selected_set:
                        assignment[slot] = next(incoming)
                offset = stop

        final = proxy(assignment)
        # In particular, a flat bias map must be exactly greedy. A numerical
        # non-improvement must never induce a gratuitous signal permutation.
        if final >= initial:
            assignment = original
            final = initial
        changed = sum(after != before for before, after in zip(original, assignment))
        return greedy.new_tensor(assignment), {
            "energy_proxy_initial": initial,
            "energy_proxy_final": final,
            "energy_changed_symbol_count": float(changed),
            "energy_changed_symbol_fraction": changed / len(tokens) if tokens else 0.0,
        }

    def _assign_quality_budgeted_swaps(
        self,
        token_ids: torch.Tensor,
        importance: torch.Tensor,
        quality_grid: torch.Tensor,
        correlation: torch.Tensor,
        reference_error: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Improve weighted within-token correlation within one image's D budget.

        D uses normalized, non-negative SDS and estimated temporal-mean MMSE
        error. The occupied slots remain exactly those of greedy. Each symbol
        participates in at most one exchange, and each initiating symbol has a
        bounded number of candidate evaluations. No channel-truth input is used.
        """
        greedy = self._assign_greedy(token_ids, importance, quality_grid)
        assignment = greedy.detach().to(device="cpu").tolist()
        original = assignment.copy()
        tokens = token_ids.detach().to(device="cpu", dtype=torch.long).tolist()
        scores = importance.detach().to(device="cpu", dtype=torch.float64).tolist()
        rho = correlation.detach().to(device="cpu", dtype=torch.float64).tolist()
        errors = reference_error.detach().to(device="cpu", dtype=torch.float64).tolist()
        n_data = self.plan.n_data
        present = sorted(set(tokens))
        if present and (present[0] < 0 or present[-1] >= len(scores)):
            raise ValueError("token_ids contains an index outside importance")
        token_order = sorted(present, key=lambda token: (-scores[token], token))
        rank = {token: index for index, token in enumerate(token_order)}
        positive_scores = {token: max(0.0, scores[token]) for token in present}
        # Scaling first preserves normalization even for very large finite SDS.
        score_scale = max(positive_scores.values(), default=0.0)
        scaled_scores = {
            token: (positive_scores[token] / score_scale if score_scale > 0.0 else 0.0)
            for token in present
        }
        score_sum = math.fsum(scaled_scores.values())
        weights = {
            token: (scaled_scores[token] / score_sum if score_sum > 0.0 else 1.0 / len(present))
            for token in present
        }
        sources = {token: [] for token in present}
        for symbol, token in enumerate(tokens):
            sources[token].append(symbol)
        counts = {token: len(sources[token]) for token in present}
        pairs = {token: count * (count - 1) / 2 for token, count in counts.items()}
        slot_of_symbol = [-1] * len(tokens)
        histograms = {token: [0] * n_data for token in present}
        # Lists by initial greedy token rank prevent a 48-symbol contiguous
        # token block from exhausting a short naive neighbor search.
        carrier_queues: List[dict[int, deque[int]]] = [{} for _ in range(n_data)]
        for slot, symbol in enumerate(assignment):
            if symbol < 0:
                continue
            token, carrier = tokens[symbol], slot % n_data
            slot_of_symbol[symbol] = slot
            histograms[token][carrier] += 1
            carrier_queues[carrier].setdefault(rank[token], deque()).append(slot)
        carrier_ranks = [sorted(queues) for queues in carrier_queues]
        vectors = {
            token: [
                math.fsum(histograms[token][j] * rho[carrier][j] for j in range(n_data))
                for carrier in range(n_data)
            ]
            for token in present
        }

        def full_objectives(values: Sequence[int]) -> tuple[float, float]:
            hist = {token: [0] * n_data for token in present}
            distortion_terms = []
            for slot, symbol in enumerate(values):
                if symbol >= 0:
                    token, carrier = tokens[symbol], slot % n_data
                    hist[token][carrier] += 1
                    distortion_terms.append(weights[token] * errors[carrier])
            corr_terms = []
            for token in present:
                if not pairs[token]:
                    continue
                pair_sum = math.fsum(
                    hist[token][i] * (hist[token][i] - 1) / 2 * rho[i][i]
                    for i in range(n_data)
                ) + math.fsum(
                    hist[token][i] * hist[token][j] * rho[i][j]
                    for i in range(n_data) for j in range(i + 1, n_data)
                )
                corr_terms.append(weights[token] * pair_sum / pairs[token])
            return math.fsum(distortion_terms), math.fsum(corr_terms)

        initial_d, initial_c = full_objectives(original)
        budget_limit = initial_d * (1.0 + self.swap_error_budget)
        current_d, current_c = initial_d, initial_c
        used_symbols = set()
        swap_count = candidate_count = rejected_count = 0
        if scores:
            mid_score = float(torch.quantile(
                torch.tensor(scores, dtype=torch.float64),
                self.mid_quantile,
            ).item())
        else:
            mid_score = math.inf

        def partner_slots(carrier: int, source_rank: int):
            """Yield unused different-token slots in nearest rank order."""
            ranks = carrier_ranks[carrier]
            center = bisect_left(ranks, source_rank)
            left, right = center - 1, center
            while left >= 0 or right < len(ranks):
                if left < 0:
                    index = right
                    right += 1
                elif right >= len(ranks):
                    index = left
                    left -= 1
                elif source_rank - ranks[left] <= ranks[right] - source_rank:
                    index = left
                    left -= 1
                else:
                    index = right
                    right += 1
                candidate_rank = ranks[index]
                if candidate_rank == source_rank:
                    continue
                queue = carrier_queues[carrier][candidate_rank]
                while queue and assignment[queue[0]] in used_symbols:
                    queue.popleft()
                if queue:
                    yield queue[0]

        # Zero budget is an explicit greedy control, including error-neutral swaps.
        if self.swap_error_budget > 0.0 and self.swap_max_swaps > 0:
            for token in token_order:
                if scores[token] < mid_score or swap_count >= self.swap_max_swaps:
                    continue
                for source in sources[token]:
                    if source in used_symbols or swap_count >= self.swap_max_swaps:
                        continue
                    source_slot = slot_of_symbol[source]
                    x = source_slot % n_data
                    # A source's weighted correlation change orders carriers.
                    # Both tokens' complete changes determine actual acceptance.
                    targets = sorted(
                        (carrier for carrier in range(n_data) if carrier != x),
                        key=lambda y: (
                            vectors[token][y] - vectors[token][x] - rho[y][x] + rho[x][x],
                            abs(errors[y] - errors[x]), y,
                        ),
                    )
                    examined = 0
                    best = None
                    # First examine one partner per carrier. A severely faded
                    # best-correlation carrier must not consume the entire
                    # candidate budget before another carrier is considered.
                    partners = {y: iter(partner_slots(y, rank[token])) for y in targets}
                    active = targets
                    while active and examined < self.swap_candidates_per_symbol:
                        remaining = []
                        for y in active:
                            if examined >= self.swap_candidates_per_symbol:
                                break
                            target_slot = next(partners[y], None)
                            if target_slot is None:
                                continue
                            remaining.append(y)
                            partner = assignment[target_slot]
                            other = tokens[partner]
                            examined += 1
                            candidate_count += 1
                            delta_d = (weights[token] - weights[other]) * (errors[y] - errors[x])
                            if current_d + delta_d > budget_limit:
                                rejected_count += 1
                                continue
                            delta_a = vectors[token][y] - vectors[token][x] - rho[y][x] + rho[x][x]
                            delta_b = vectors[other][x] - vectors[other][y] - rho[x][y] + rho[y][y]
                            delta_c = (
                                (weights[token] * delta_a / pairs[token] if pairs[token] else 0.0)
                                + (weights[other] * delta_b / pairs[other] if pairs[other] else 0.0)
                            )
                            if delta_c >= -1e-12:
                                continue
                            key = (delta_c, delta_d, target_slot)
                            if best is None or key < best[0]:
                                best = (key, target_slot, partner, other, y)
                            if examined >= self.swap_candidates_per_symbol:
                                break
                        active = remaining
                    if best is None:
                        continue
                    (delta_c, delta_d, _), target_slot, partner, other, y = best
                    assignment[source_slot], assignment[target_slot] = partner, source
                    slot_of_symbol[source], slot_of_symbol[partner] = target_slot, source_slot
                    used_symbols.update((source, partner))
                    histograms[token][x] -= 1
                    histograms[token][y] += 1
                    histograms[other][y] -= 1
                    histograms[other][x] += 1
                    for carrier in range(n_data):
                        change = rho[carrier][y] - rho[carrier][x]
                        vectors[token][carrier] += change
                        vectors[other][carrier] -= change
                    current_d += delta_d
                    current_c += delta_c
                    swap_count += 1

        final_d, final_c = full_objectives(assignment)
        # The complete recomputation is the authoritative global check. A
        # rounding overrun falls back safely, rather than enlarging the budget.
        fallback = (
            not math.isfinite(final_d) or not math.isfinite(final_c)
            or final_d > budget_limit
            or final_c > initial_c + 1e-12
            or (swap_count > 0 and final_c >= initial_c)
        )
        if fallback:
            assignment = original
            final_d, final_c = initial_d, initial_c
            swap_count = 0
        relative_increase = (final_d - initial_d) / initial_d if initial_d > 0.0 else 0.0
        diagnostics = {
            "swap_count": float(swap_count),
            "swap_candidate_evaluations": float(candidate_count),
            "swap_proxy_initial": initial_d,
            "swap_proxy_final": final_d,
            "swap_proxy_budget_limit": budget_limit,
            "swap_proxy_relative_increase": relative_increase,
            "swap_budget_fraction": max(0.0, relative_increase) / self.swap_error_budget
                if self.swap_error_budget > 0.0 else 0.0,
            "swap_corr_initial": initial_c,
            "swap_corr_final": final_c,
            "swap_moved_symbol_fraction": 2.0 * swap_count / len(tokens) if tokens else 0.0,
            "swap_fallback": float(fallback),
            "swap_budget_rejected_candidates": float(rejected_count),
        }
        return torch.tensor(assignment, device=token_ids.device, dtype=torch.long), diagnostics

    def _assign_frequency_dispersed(
        self,
        token_ids: torch.Tensor,
        importance: torch.Tensor,
        quality_grid: torch.Tensor,
        correlation: torch.Tensor,
        uniform: bool,
        band_reference_quality: Optional[torch.Tensor] = None,
        risk_band_scores: Optional[torch.Tensor] = None,
        risk_eligible_bands: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_semantic_symbols = token_ids.numel()
        total_slots = quality_grid.numel()
        n_data = self.plan.n_data
        n_ofdm_symbols = quality_grid.shape[0]
        quality_values = quality_grid.detach().to(device="cpu", dtype=torch.float64).tolist()
        correlation_values = correlation.detach().to(device="cpu", dtype=torch.float64).tolist()
        token_values = token_ids.detach().to(device="cpu", dtype=torch.long)
        importance_values = importance.detach().to(device="cpu", dtype=torch.float64)
        reference_values = (
            None
            if band_reference_quality is None
            else band_reference_quality.detach().to(device="cpu", dtype=torch.float64).tolist()
        )
        eligible_bands = (
            None
            if risk_eligible_bands is None
            else [
                band
                for band, eligible in enumerate(
                    risk_eligible_bands.detach().to(device="cpu", dtype=torch.bool).tolist()
                )
                if eligible
            ]
        )
        rejected_order: List[int] = []
        if eligible_bands is not None:
            if risk_band_scores is None:
                raise ValueError("risk_band_scores are required for risk capacity spill")
            scores = risk_band_scores.detach().to(device="cpu", dtype=torch.float64).tolist()
            rejected_order = sorted(
                (band for band in range(self.plan.n_bands) if band not in eligible_bands),
                key=lambda band: (-scores[band], band),
            )
        assignments = torch.full((total_slots,), -1, dtype=torch.long)

        # Each carrier owns T independent slots. Sorting once makes subsequent
        # objective evaluations O(N_data) per semantic symbol, rather than
        # rescanning all T*N_data remaining slots.
        carrier_slots: List[List[int]] = []
        for carrier in range(n_data):
            slots = [time * n_data + carrier for time in range(n_ofdm_symbols)]
            slots.sort(
                key=lambda slot: (-quality_values[slot // n_data][carrier], slot)
            )
            carrier_slots.append(slots)
        carrier_cursor = [0] * n_data

        carrier_to_band = [-1] * n_data
        for band, positions in enumerate(self.plan.band_data_positions):
            for carrier in positions:
                carrier_to_band[carrier] = band

        n_tokens = importance_values.numel()
        if token_values.min().item() < 0 or token_values.max().item() >= n_tokens:
            raise ValueError("token_ids contains an index outside importance")
        token_order = sorted(
            range(n_tokens),
            key=lambda token: (-float(importance_values[token].item()), token),
        )
        q_high = float(torch.quantile(importance_values, self.high_quantile).item())
        q_mid = float(torch.quantile(importance_values, self.mid_quantile).item())

        def carrier_available(carrier: int) -> bool:
            return carrier_cursor[carrier] < len(carrier_slots[carrier])

        def next_slot(carrier: int) -> int:
            return carrier_slots[carrier][carrier_cursor[carrier]]

        def next_quality(carrier: int) -> float:
            slot = next_slot(carrier)
            return float(quality_values[slot // n_data][carrier])

        def best_carrier(
            allowed_bands: Sequence[int],
            correlation_sum: Sequence[float],
            history_size: int,
            correlation_weight: float,
        ) -> Optional[int]:
            allowed = set(allowed_bands)
            best: Optional[int] = None
            best_key: Optional[tuple[float, float, int, int]] = None
            for carrier in range(n_data):
                if carrier_to_band[carrier] not in allowed or not carrier_available(carrier):
                    continue
                quality_reward = self.quality_weight * next_quality(carrier)
                penalty = 0.0
                if history_size:
                    penalty = correlation_weight * correlation_sum[carrier] / history_size
                score = quality_reward - penalty
                key = (score, next_quality(carrier), -next_slot(carrier), -carrier)
                if best_key is None or key > best_key:
                    best_key = key
                    best = carrier
            return best

        def select_bands(diversity: int, correlation_weight: float) -> List[int]:
            selected: List[int] = []
            anchors: List[int] = []
            for _ in range(min(diversity, self.plan.n_bands)):
                best: Optional[int] = None
                best_key: Optional[tuple[float, float, int, int]] = None
                for carrier in range(n_data):
                    band = carrier_to_band[carrier]
                    if band in selected or not carrier_available(carrier):
                        continue
                    if eligible_bands is not None and band not in eligible_bands:
                        continue
                    penalty = 0.0
                    if anchors:
                        penalty = correlation_weight * sum(
                            correlation_values[carrier][anchor] for anchor in anchors
                        ) / len(anchors)
                    quality = (
                        next_quality(carrier)
                        if reference_values is None
                        else float(reference_values[carrier])
                    )
                    score = self.quality_weight * quality - penalty
                    key = (score, quality, -next_slot(carrier), -carrier)
                    if best_key is None or key > best_key:
                        best_key = key
                        best = carrier
                if best is None:
                    break
                selected.append(carrier_to_band[best])
                anchors.append(best)
            return selected

        minimum_importance = float(importance_values.amin().item())
        importance_span = float(
            (importance_values.amax() - importance_values.amin()).item()
        )
        for token in token_order:
            symbol_indices = torch.nonzero(
                token_values == token, as_tuple=False
            ).flatten().tolist()
            if not symbol_indices:
                continue
            score = float(importance_values[token].item())
            if uniform or score >= q_high:
                diversity = self.diversity_levels[2]
            elif score >= q_mid:
                diversity = self.diversity_levels[1]
            else:
                diversity = self.diversity_levels[0]
            relative_importance = (
                0.0
                if uniform or importance_span <= torch.finfo(torch.float64).eps
                else (score - minimum_importance) / importance_span
            )
            token_correlation_weight = self.correlation_weight * (
                1.0 + self.importance_correlation_gain * relative_importance
            )
            selected = select_bands(
                min(diversity, len(symbol_indices)), token_correlation_weight
            )
            uncovered = set(selected)
            correlation_sum = [0.0] * n_data
            history_size = 0
            for symbol_index in symbol_indices:
                candidate_bands = sorted(uncovered) if uncovered else selected
                carrier = best_carrier(
                    candidate_bands,
                    correlation_sum,
                    history_size,
                    token_correlation_weight,
                )
                if carrier is None:
                    # A token can outgrow the capacity of its requested bands
                    # when T is small. Expanding is necessary for a bijection;
                    # it never replicates a semantic symbol.
                    if eligible_bands is None:
                        # Keep the established mapper's overflow behavior exact.
                        carrier = best_carrier(
                            list(range(self.plan.n_bands)),
                            correlation_sum,
                            history_size,
                            token_correlation_weight,
                        )
                    else:
                        # Use all still-available healthy capacity before opening
                        # a rejected band. The gate is a preference, never a
                        # deletion or repetition of semantic symbols.
                        carrier = best_carrier(
                            [band for band in eligible_bands if band not in selected],
                            correlation_sum,
                            history_size,
                            token_correlation_weight,
                        )
                        if carrier is None:
                            for band in rejected_order:
                                if band in selected:
                                    continue
                                carrier = best_carrier(
                                    [band],
                                    correlation_sum,
                                    history_size,
                                    token_correlation_weight,
                                )
                                if carrier is not None:
                                    break
                    if carrier is None:
                        raise RuntimeError("Mapper ran out of OFDM slots")
                    if carrier_to_band[carrier] not in selected:
                        selected.append(carrier_to_band[carrier])
                slot = next_slot(carrier)
                carrier_cursor[carrier] += 1
                assignments[slot] = int(symbol_index)
                uncovered.discard(carrier_to_band[carrier])
                for candidate in range(n_data):
                    correlation_sum[candidate] += correlation_values[candidate][carrier]
                history_size += 1

        if int((assignments >= 0).sum().item()) != n_semantic_symbols:
            raise RuntimeError("Mapper did not assign every semantic symbol exactly once")
        return assignments.to(token_ids.device)

    def map(
        self,
        symbols: torch.Tensor,
        token_ids: torch.Tensor,
        importance: torch.Tensor,
        h_hat: Optional[torch.Tensor],
        mode: str = "proposed_ufd",
        seed: Optional[int] = None,
        correlation_matrix: Optional[torch.Tensor] = None,
        noise_variance: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, PermutationMeta]:
        mode = mode.lower()
        if mode not in self.MODES:
            raise ValueError(f"Unsupported mapper mode {mode!r}; choose from {sorted(self.MODES)}")
        if symbols.ndim != 2 or not symbols.is_complex():
            raise ValueError("symbols must be a complex tensor with shape [B,K]")
        if importance.ndim != 2 or importance.shape[0] != symbols.shape[0]:
            raise ValueError("importance must have shape [B,M]")
        batch, n_semantic_symbols = symbols.shape
        token_ids = self._token_ids_for_batch(
            token_ids, batch, n_semantic_symbols
        ).to(symbols.device)
        n_ofdm_symbols = math.ceil(n_semantic_symbols / self.plan.n_data)
        total_slots = n_ofdm_symbols * self.plan.n_data
        band_reference_quality = None
        risk_band_scores = None
        risk_eligible_bands = None
        swap_reference_error = None
        swap_rows = []
        energy_carrier_bias = None
        energy_symbol_power = None
        energy_rows = []
        if mode == "greedy_token_random":
            if h_hat is None:
                raise ValueError("greedy_token_random requires an estimated h_hat")
            if h_hat.ndim not in {2, 3}:
                raise ValueError("h_hat must have shape [B,N_fft] or [B,T,N_fft]")
            if not bool(torch.isfinite(h_hat).all().item()):
                raise ValueError("h_hat must contain only finite values for token-random mode")
            if not bool(torch.isfinite(importance).all().item()):
                raise ValueError("importance must contain only finite values for token-random mode")
            if token_ids.numel() and (
                int(token_ids.min().item()) < 0
                or int(token_ids.max().item()) >= importance.shape[1]
            ):
                raise ValueError("token_ids contains an index outside importance")
        if mode == "greedy_token_energy":
            if h_hat is None:
                raise ValueError("greedy_token_energy requires an estimated h_hat")
            if h_hat.ndim not in {2, 3}:
                raise ValueError("h_hat must have shape [B,N_fft] or [B,T,N_fft]")
            if not bool(torch.isfinite(h_hat).all().item()):
                raise ValueError("h_hat must contain only finite values for energy mode")
            if not bool(torch.isfinite(symbols).all().item()):
                raise ValueError("symbols must contain only finite values for energy mode")
            if not bool(torch.isfinite(importance).all().item()):
                raise ValueError("importance must contain only finite values for energy mode")
            if token_ids.numel() and (
                int(token_ids.min().item()) < 0
                or int(token_ids.max().item()) >= importance.shape[1]
            ):
                raise ValueError("token_ids contains an index outside importance")
            if noise_variance is None:
                raise ValueError("greedy_token_energy requires noise_variance")
            energy_noise = torch.as_tensor(
                noise_variance, device=symbols.device, dtype=torch.float64
            )
            if energy_noise.ndim == 0:
                energy_noise = energy_noise.expand(batch)
            energy_noise = energy_noise.reshape(-1)
            if energy_noise.numel() != batch or not bool(
                torch.isfinite(energy_noise).all().item()
                and (energy_noise > 0.0).all().item()
            ):
                raise ValueError("energy noise_variance must be scalar or [B], finite and positive")
        if mode == "proposed_ufd_swap":
            if not bool(torch.isfinite(symbols).all().item()):
                raise ValueError("symbols must contain only finite values for swap mode")
            if h_hat is None:
                raise ValueError("proposed_ufd_swap requires an estimated h_hat")
            if h_hat.ndim not in {2, 3}:
                raise ValueError("h_hat must have shape [B,N_fft] or [B,T,N_fft]")
            if not bool(torch.isfinite(h_hat).all().item()):
                raise ValueError("h_hat must contain only finite values for swap mode")
            if not bool(torch.isfinite(importance).all().item()):
                raise ValueError("importance must contain only finite values for swap mode")
            if noise_variance is not None:
                variance = torch.as_tensor(
                    noise_variance, device=symbols.device, dtype=symbols.real.dtype
                )
                if not bool(torch.isfinite(variance).all().item()) or not bool(
                    (variance > 0.0).all().item()
                ):
                    raise ValueError("swap noise_variance must be finite and positive")
        if mode in self.BAND_REFERENCE_MODES:
            if h_hat is None:
                raise ValueError(f"{mode} requires an estimated h_hat")
            if h_hat.ndim not in {2, 3}:
                raise ValueError("h_hat must have shape [B,N_fft] or [B,T,N_fft]")
            if not bool(torch.isfinite(h_hat).all().item()):
                raise ValueError("h_hat must contain only finite values for band-reference modes")
        quality = self._quality(
            h_hat,
            batch,
            n_ofdm_symbols,
            symbols.device,
            symbols.real.dtype,
            noise_variance=noise_variance,
        )
        if mode == "greedy_token_random" and not bool(torch.isfinite(quality).all().item()):
            raise ValueError("token-random raw quality must contain only finite values")
        if mode == "greedy_token_energy":
            if not bool(torch.isfinite(quality).all().item()):
                raise ValueError("energy raw quality must contain only finite values")
            temporal_h = h_hat[:, None, :] if h_hat.ndim == 2 else h_hat
            data_h = temporal_h.to(torch.complex128).mean(dim=1).index_select(
                -1, self.plan.tensor_indices("data", temporal_h.device)
            )
            reference_power = data_h.real.square() + data_h.imag.square()
            energy_carrier_bias = (
                energy_noise[:, None] / (reference_power + energy_noise[:, None])
            ).square()
            energy_symbol_power = (
                symbols.real.to(torch.float64).square()
                + symbols.imag.to(torch.float64).square()
            )
            if not bool(torch.isfinite(energy_carrier_bias).all().item()) or not bool(
                torch.isfinite(energy_symbol_power).all().item()
            ):
                raise ValueError("energy proxy inputs must be finite")
        if mode == "proposed_ufd_swap":
            if not bool(torch.isfinite(quality).all().item()):
                raise ValueError("swap raw quality must contain only finite values")
            if token_ids.numel() and (
                int(token_ids.min().item()) < 0
                or int(token_ids.max().item()) >= importance.shape[1]
            ):
                raise ValueError("token_ids contains an index outside importance")
            # This proxy intentionally excludes gamma: its definition is the
            # normalized estimated MMSE error 1/(1+|mean_t H_hat|^2/noise).
            temporal_h = h_hat[:, None, :] if h_hat.ndim == 2 else h_hat
            reference_h = temporal_h.mean(dim=1).index_select(
                -1, self.plan.tensor_indices("data", h_hat.device)
            ).to(symbols.device)
            variance = torch.as_tensor(
                1.0 if noise_variance is None else noise_variance,
                device=symbols.device, dtype=symbols.real.dtype,
            )
            if variance.ndim == 0:
                variance = variance.expand(batch)
            reference_quality = reference_h.abs().square() / variance.reshape(batch, 1)
            if not bool(torch.isfinite(reference_quality).all().item()) or not bool(
                (reference_quality >= 0.0).all().item()
            ):
                raise ValueError("swap reference quality must be finite and non-negative")
            swap_reference_error = 1.0 / (1.0 + reference_quality)
        if mode in self.BAND_REFERENCE_MODES:
            # Average the complex channel first, not instantaneous power. In
            # the block-static channel this suppresses independent CSI errors
            # while leaving raw per-time quality available for slot selection.
            temporal_h = h_hat[:, None, :] if h_hat.ndim == 2 else h_hat
            band_reference_quality = self._quality(
                temporal_h.mean(dim=1, keepdim=True),
                batch,
                1,
                symbols.device,
                symbols.real.dtype,
                noise_variance=noise_variance,
            ).squeeze(1)
            if not bool(torch.isfinite(quality).all().item()) or not bool(
                torch.isfinite(band_reference_quality).all().item()
            ):
                raise ValueError("band-reference quality must contain only finite values")
            risk_band_scores = torch.stack(
                [
                    torch.quantile(band_reference_quality[:, list(positions)], 0.5, dim=1)
                    for positions in self.plan.band_data_positions
                ],
                dim=1,
            )
            risk_eligible_bands = torch.ones_like(risk_band_scores, dtype=torch.bool)
            if mode == "proposed_ufd_risk":
                best_score = risk_band_scores.amax(dim=1, keepdim=True)
                risk_eligible_bands = (
                    risk_band_scores >= self.risk_band_threshold * best_score
                )
                # Stable order gives the smaller band index priority on a tie.
                minimum = min(self.risk_min_bands, self.plan.n_bands)
                for sample in range(batch):
                    strongest = sorted(
                        range(self.plan.n_bands),
                        key=lambda band: (-float(risk_band_scores[sample, band].item()), band),
                    )[:minimum]
                    risk_eligible_bands[sample, strongest] = True
        correlation = self._correlation_for_batch(
            correlation_matrix,
            batch,
            symbols.device,
            symbols.real.dtype,
        )
        if mode == "proposed_ufd_swap" and not bool(torch.isfinite(correlation).all().item()):
            raise ValueError("swap correlation must contain only finite values")
        assignment_rows = []
        for sample in range(batch):
            if mode == "sequential":
                assignment = torch.full(
                    (total_slots,), -1, device=symbols.device, dtype=torch.long
                )
                assignment[:n_semantic_symbols] = torch.arange(
                    n_semantic_symbols, device=symbols.device
                )
            elif mode == "random":
                generator = self._mapper_random_generator(mode, seed, sample)
                permutation = torch.randperm(
                    n_semantic_symbols, generator=generator
                ).to(symbols.device)
                assignment = torch.full(
                    (total_slots,), -1, device=symbols.device, dtype=torch.long
                )
                assignment[:n_semantic_symbols] = permutation
            elif mode == "greedy":
                assignment = self._assign_greedy(token_ids[sample], importance[sample], quality[sample])
            elif mode == "greedy_token_random":
                assignment = self._assign_greedy_token_random(
                    token_ids[sample], importance[sample], quality[sample],
                    self._mapper_random_generator(mode, seed, sample),
                )
            elif mode == "greedy_token_energy":
                assignment, diagnostics = self._assign_greedy_token_energy(
                    token_ids[sample], importance[sample], quality[sample],
                    energy_symbol_power[sample], energy_carrier_bias[sample],
                )
                energy_rows.append(diagnostics)
            elif mode == "proposed_ufd_swap":
                assignment, diagnostics = self._assign_quality_budgeted_swaps(
                    token_ids[sample], importance[sample], quality[sample],
                    correlation[sample], swap_reference_error[sample],
                )
                swap_rows.append(diagnostics)
            else:
                assignment = self._assign_frequency_dispersed(
                    token_ids[sample],
                    importance[sample],
                    quality[sample],
                    correlation[sample],
                    uniform=mode in self.UNIFORM_LOWCORR_MODES,
                    band_reference_quality=(
                        None if band_reference_quality is None else band_reference_quality[sample]
                    ),
                    risk_band_scores=(None if risk_band_scores is None else risk_band_scores[sample]),
                    risk_eligible_bands=(
                        risk_eligible_bands[sample] if mode == "proposed_ufd_risk" else None
                    ),
                )
            assignment_rows.append(assignment)

        symbol_index_grid = torch.stack(assignment_rows).view(
            batch, n_ofdm_symbols, self.plan.n_data
        )
        valid = symbol_index_grid >= 0
        gathered = torch.gather(symbols, 1, symbol_index_grid.reshape(batch, -1).clamp_min(0))
        grid = gathered.view_as(symbol_index_grid).to(symbols.dtype) * valid.to(symbols.dtype)
        meta = PermutationMeta(
            symbol_index_grid=symbol_index_grid,
            n_symbols=n_semantic_symbols,
            mode=mode,
            band_reference_quality=band_reference_quality,
            risk_band_scores=risk_band_scores,
            risk_eligible_bands=risk_eligible_bands,
            swap_diagnostics=(
                {
                    key: torch.tensor(
                        [row[key] for row in swap_rows],
                        device=symbols.device, dtype=torch.float64,
                    )
                    for key in swap_rows[0]
                }
                if swap_rows else None
            ),
            energy_diagnostics=(
                {
                    key: torch.tensor(
                        [row[key] for row in energy_rows],
                        device=symbols.device, dtype=torch.float64,
                    )
                    for key in energy_rows[0]
                }
                if energy_rows else None
            ),
        )
        return grid, meta

    def unmap(self, grid_hat: torch.Tensor, meta: PermutationMeta) -> torch.Tensor:
        if grid_hat.shape != meta.symbol_index_grid.shape:
            raise ValueError("grid_hat shape does not match the permutation metadata")
        batch = grid_hat.shape[0]
        flat_values = grid_hat.reshape(batch, -1)
        flat_indices = meta.symbol_index_grid.reshape(batch, -1)
        valid = flat_indices >= 0
        output = torch.zeros(
            batch,
            meta.n_symbols,
            device=grid_hat.device,
            dtype=grid_hat.dtype,
        )
        output.scatter_add_(
            1,
            flat_indices.clamp_min(0),
            flat_values * valid.to(grid_hat.dtype),
        )
        return output
