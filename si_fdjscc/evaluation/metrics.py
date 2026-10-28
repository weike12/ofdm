from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch

from ..losses.reconstruction import per_sample_psnr


SWAP_DIAGNOSTIC_FIELDS = (
    "swap_count",
    "swap_candidate_evaluations",
    "swap_proxy_initial",
    "swap_proxy_final",
    "swap_proxy_budget_limit",
    "swap_proxy_relative_increase",
    "swap_budget_fraction",
    "swap_corr_initial",
    "swap_corr_final",
    "swap_moved_symbol_fraction",
    "swap_fallback",
    "swap_budget_rejected_candidates",
)

ENERGY_DIAGNOSTIC_FIELDS = (
    "energy_proxy_initial",
    "energy_proxy_final",
    "energy_changed_symbol_fraction",
    "energy_changed_symbol_count",
)

SYMBOL_RESIDUAL_FIELDS = (
    "residual_symbol_mse",
    "residual_sds_weighted_token_mse",
    "residual_estimated_energy_proxy",
    "residual_mapper_csi_nmse",
    "residual_receiver_csi_nmse",
)


def swap_diagnostic_metrics(
    swap_diagnostics: Optional[Mapping[str, torch.Tensor]],
    batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Return one scalar per batch item with a stable CSV schema.

    Legacy modes have no swap metadata: their fields are NaN, so absence of
    the algorithm is not mistaken for a zero-swap execution. A swap execution
    must provide the complete schema and finite Tensor[B] values. Partial
    metadata is an error rather than a silently incomplete diagnostic run.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if swap_diagnostics is None:
        return {
            key: torch.full((batch_size,), float("nan"), dtype=torch.float64, device=device)
            for key in SWAP_DIAGNOSTIC_FIELDS
        }
    missing = set(SWAP_DIAGNOSTIC_FIELDS) - set(swap_diagnostics)
    if missing:
        raise ValueError("swap_diagnostics is missing fields: " + ", ".join(sorted(missing)))
    unexpected = set(swap_diagnostics) - set(SWAP_DIAGNOSTIC_FIELDS)
    if unexpected:
        raise ValueError("swap_diagnostics has unknown fields: " + ", ".join(sorted(unexpected)))
    result = {}
    for key in SWAP_DIAGNOSTIC_FIELDS:
        value = swap_diagnostics[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be a Tensor with shape [{batch_size}]")
        if tuple(value.shape) != (batch_size,):
            raise ValueError(f"{key} must have shape [{batch_size}]")
        if value.is_complex():
            raise ValueError(f"{key} must be real-valued")
        result[key] = value.detach().to(device=device, dtype=torch.float64)
    stacked = torch.stack(list(result.values()))
    if not bool(torch.isfinite(stacked).all().item()):
        raise ValueError("swap_diagnostics must contain only finite values")
    return result


def energy_diagnostic_metrics(
    energy_diagnostics: Optional[Mapping[str, torch.Tensor]],
    batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Expose mapper energy diagnostics with the same columns for all modes.

    NaN means the mapper has no energy-reordering algorithm. The energy mapper
    must supply every field, including zero changes, with finite per-image
    values. These values describe an estimated-channel proxy, not measured MSE.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if energy_diagnostics is None:
        return {
            key: torch.full((batch_size,), float("nan"), dtype=torch.float64, device=device)
            for key in ENERGY_DIAGNOSTIC_FIELDS
        }
    missing = set(ENERGY_DIAGNOSTIC_FIELDS) - set(energy_diagnostics)
    if missing:
        raise ValueError("energy_diagnostics is missing fields: " + ", ".join(sorted(missing)))
    unexpected = set(energy_diagnostics) - set(ENERGY_DIAGNOSTIC_FIELDS)
    if unexpected:
        raise ValueError("energy_diagnostics has unknown fields: " + ", ".join(sorted(unexpected)))
    result = {}
    for key in ENERGY_DIAGNOSTIC_FIELDS:
        value = energy_diagnostics[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be a Tensor with shape [{batch_size}]")
        if tuple(value.shape) != (batch_size,):
            raise ValueError(f"{key} must have shape [{batch_size}]")
        if value.is_complex():
            raise ValueError(f"{key} must be real-valued")
        result[key] = value.detach().to(device=device, dtype=torch.float64)
    if not bool(torch.isfinite(torch.stack(list(result.values()))).all().item()):
        raise ValueError("energy_diagnostics must contain only finite values")
    return result


@torch.no_grad()
def symbol_residual_metrics(
    symbols_tx: torch.Tensor,
    symbols_rx: torch.Tensor,
    token_ids: torch.Tensor,
    importance: torch.Tensor,
    symbol_index_grid: torch.Tensor,
    h_hat: torch.Tensor,
    equalizer_h_hat: torch.Tensor,
    h_true: torch.Tensor,
    noise_variance: torch.Tensor,
    data_indices: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Audit complex-symbol residuals after equalization, one value per image.

    ``symbols_tx`` is the transmitted power-normalized complex codeword;
    ``symbols_rx`` is its recovered counterpart before ComplexAdapter.inverse.
    This is valid even when the adapter discards a padded final imaginary scalar
    or does not restore the latent scale. The actual residuals use no true CSI.

    The SDS-weighted residual averages symbol errors *within* each token, then
    weights present tokens by the same softmax SDS scores passed to the mapper.
    All-zero scores give equal weight to present tokens. The estimated proxy is
    an unweighted sum, matching the token-energy mapper's objective exactly:
    sum_j |s_j|^2 [noise/(|mean_t H_hat[t, carrier(j)]|^2 + noise)]^2.
    Only the two CSI NMSE fields use the true channel response, for reporting.
    """
    if (
        not isinstance(symbols_tx, torch.Tensor)
        or not isinstance(symbols_rx, torch.Tensor)
        or symbols_tx.ndim != 2
        or symbols_rx.shape != symbols_tx.shape
        or not symbols_tx.is_complex()
        or not symbols_rx.is_complex()
    ):
        raise ValueError("symbols_tx and symbols_rx must be complex tensors with shape [B,K]")
    batch, n_symbols = symbols_tx.shape
    if batch < 1 or n_symbols < 1:
        raise ValueError("symbol batch and symbol count must be positive")
    device = symbols_tx.device
    if symbols_rx.device != device:
        raise ValueError("symbols_tx and symbols_rx must be on the same device")
    if token_ids.ndim == 1:
        token_ids = token_ids.unsqueeze(0).expand(batch, -1)
    if token_ids.shape != (batch, n_symbols) or token_ids.is_floating_point() or token_ids.is_complex():
        raise ValueError("token_ids must be integer with shape [K] or [B,K]")
    if importance.ndim != 2 or importance.shape[0] != batch or importance.shape[1] < 1:
        raise ValueError("importance must have shape [B,M] with M >= 1")
    n_tokens = importance.shape[1]
    if symbol_index_grid.ndim != 3 or symbol_index_grid.shape[0] != batch:
        raise ValueError("symbol_index_grid must have shape [B,T,N_data]")
    _, n_times, n_data = symbol_index_grid.shape
    if n_times < 1 or n_data < 1:
        raise ValueError("symbol_index_grid dimensions must be positive")
    if symbol_index_grid.is_floating_point() or symbol_index_grid.is_complex():
        raise ValueError("symbol_index_grid must contain integer source indices")
    if data_indices.ndim != 1 or data_indices.numel() != n_data:
        raise ValueError("data_indices must have one FFT-bin index per data carrier")
    if data_indices.is_floating_point() or data_indices.is_complex():
        raise ValueError("data_indices must contain integer FFT-bin indices")

    token_ids = token_ids.to(device=device, dtype=torch.long)
    grid = symbol_index_grid.to(device=device, dtype=torch.long)
    data_indices = data_indices.to(device=device, dtype=torch.long)
    importance = importance.detach().to(device=device, dtype=torch.float64)
    if (
        not bool(torch.isfinite(symbols_tx).all().item())
        or not bool(torch.isfinite(symbols_rx).all().item())
        or not bool(torch.isfinite(importance).all().item())
        or bool((importance < 0).any().item())
    ):
        raise ValueError("symbols and importance must be finite; importance must be nonnegative")
    if token_ids.numel() and (bool((token_ids < 0).any().item()) or bool((token_ids >= n_tokens).any().item())):
        raise ValueError("token_ids contains an index outside importance")
    if bool((grid < -1).any().item()) or bool((grid >= n_symbols).any().item()):
        raise ValueError("symbol_index_grid contains an invalid source index")
    flattened_grid = grid.reshape(batch, -1)
    flattened_valid = flattened_grid.ge(0)
    if not bool((flattened_valid.sum(dim=1) == n_symbols).all().item()):
        raise ValueError("symbol_index_grid must assign exactly n_symbols valid slots per image")
    assignments = torch.zeros((batch, n_symbols), device=device, dtype=torch.int32)
    assignments.scatter_add_(1, flattened_grid.clamp_min(0), flattened_valid.to(torch.int32))
    if not bool((assignments == 1).all().item()):
        raise ValueError("symbol_index_grid must contain every source symbol exactly once")

    difference = symbols_rx.detach().to(torch.complex128) - symbols_tx.detach().to(torch.complex128)
    error = difference.real.square() + difference.imag.square()
    counts = torch.zeros((batch, n_tokens), device=device, dtype=torch.float64)
    token_error_sum = torch.zeros_like(counts)
    counts.scatter_add_(1, token_ids, torch.ones_like(error))
    token_error_sum.scatter_add_(1, token_ids, error)
    present = counts > 0
    scaled_scores = importance / importance.amax(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
    scaled_scores = torch.where(present, scaled_scores, torch.zeros_like(scaled_scores))
    weight_sum = scaled_scores.sum(dim=1, keepdim=True)
    uniform = present.to(torch.float64) / present.sum(dim=1, keepdim=True)
    weights = torch.where(weight_sum > 0, scaled_scores / weight_sum.clamp_min(torch.finfo(torch.float64).tiny), uniform)
    token_mse = token_error_sum / counts.clamp_min(1.0)

    def _channel_data(name: str, response: torch.Tensor) -> torch.Tensor:
        if not isinstance(response, torch.Tensor) or not response.is_complex():
            raise ValueError(f"{name} must be a complex channel response")
        if response.ndim == 2:
            response = response.unsqueeze(1)
        if response.ndim != 3 or response.shape[0] != batch or response.shape[1] not in (1, n_times):
            raise ValueError(f"{name} must have shape [B,N_fft] or [B,T,N_fft]")
        if data_indices.numel() and (
            bool((data_indices < 0).any().item())
            or bool((data_indices >= response.shape[2]).any().item())
        ):
            raise ValueError(f"data_indices are outside {name}")
        response = response.detach().to(device=device)
        if not bool(torch.isfinite(response).all().item()):
            raise ValueError(f"{name} must be finite")
        return response.index_select(-1, data_indices).expand(batch, n_times, n_data).to(torch.complex128)

    estimated = _channel_data("h_hat", h_hat)
    receiver = _channel_data("equalizer_h_hat", equalizer_h_hat)
    truth = _channel_data("h_true", h_true)
    variance = torch.as_tensor(noise_variance, device=device, dtype=torch.float64).reshape(-1)
    if variance.numel() == 1:
        variance = variance.expand(batch)
    if variance.shape != (batch,) or not bool(torch.isfinite(variance).all().item()) or not bool((variance > 0).all().item()):
        raise ValueError("noise_variance must be finite, positive, and scalar or [B]")
    reference_h = estimated.mean(dim=1)
    reference_power = reference_h.real.square() + reference_h.imag.square()
    factor = (variance[:, None] / (reference_power + variance[:, None])).square()
    indices = grid.clamp_min(0).reshape(batch, -1)
    clean_symbols = symbols_tx.detach()
    symbol_energy = (
        clean_symbols.real.to(torch.float64).square()
        + clean_symbols.imag.to(torch.float64).square()
    ).gather(1, indices)
    valid = grid.ge(0).to(torch.float64)
    proxy = (symbol_energy.reshape_as(grid) * factor[:, None, :] * valid).sum(dim=(1, 2))

    denominator = truth.abs().square().sum(dim=(1, 2))
    if not bool((denominator > 0).all().item()):
        raise ValueError("true channel has zero energy on all data carriers")
    mapper_nmse = (estimated - truth).abs().square().sum(dim=(1, 2)) / denominator
    receiver_nmse = (receiver - truth).abs().square().sum(dim=(1, 2)) / denominator
    result = {
        "residual_symbol_mse": error.mean(dim=1),
        "residual_sds_weighted_token_mse": (weights * token_mse).sum(dim=1),
        "residual_estimated_energy_proxy": proxy,
        "residual_mapper_csi_nmse": mapper_nmse,
        "residual_receiver_csi_nmse": receiver_nmse,
    }
    if not bool(torch.isfinite(torch.stack(list(result.values()))).all().item()):
        raise ValueError("symbol residual diagnostics must be finite")
    return result


def assigned_high_sensitivity_correlation(
    symbol_index_grid: torch.Tensor,
    token_ids: torch.Tensor,
    sensitivity: torch.Tensor,
    correlation_matrix: torch.Tensor,
    high_quantile: float = 0.80,
) -> torch.Tensor:
    """Measure within-token carrier correlation for high-sensitivity units."""
    if symbol_index_grid.ndim != 3:
        raise ValueError("symbol_index_grid must have shape [B,T,N_data]")
    batch, _, n_data = symbol_index_grid.shape
    if sensitivity.ndim != 2 or sensitivity.shape[0] != batch:
        raise ValueError("sensitivity must have shape [B,M]")
    if not 0.0 <= high_quantile <= 1.0:
        raise ValueError("high_quantile must be in [0,1]")
    if token_ids.ndim == 1:
        token_ids = token_ids.unsqueeze(0).expand(batch, -1)
    if token_ids.ndim != 2 or token_ids.shape[0] != batch:
        raise ValueError("token_ids must have shape [K] or [B,K]")
    correlation = torch.as_tensor(
        correlation_matrix, device=symbol_index_grid.device
    ).abs()
    if correlation.ndim == 2:
        correlation = correlation.unsqueeze(0).expand(batch, -1, -1)
    if correlation.shape != (batch, n_data, n_data):
        raise ValueError("correlation_matrix has an incompatible shape")

    results = []
    for sample in range(batch):
        threshold = torch.quantile(sensitivity[sample].float(), high_quantile)
        high_tokens = torch.nonzero(
            sensitivity[sample] >= threshold, as_tuple=False
        ).flatten()
        flattened = symbol_index_grid[sample].reshape(-1)
        valid = flattened >= 0
        source_indices = flattened[valid]
        carrier_positions = (
            torch.arange(flattened.numel(), device=flattened.device)[valid]
            % n_data
        )
        source_tokens = token_ids[sample].index_select(0, source_indices)
        pair_values = []
        for token in high_tokens:
            positions = carrier_positions[source_tokens == token]
            if positions.numel() < 2:
                continue
            pair_indices = torch.triu_indices(
                positions.numel(), positions.numel(), offset=1, device=positions.device
            )
            pair_values.append(
                correlation[sample][
                    positions[pair_indices[0]], positions[pair_indices[1]]
                ]
            )
        if pair_values:
            results.append(torch.cat(pair_values).float().mean())
        else:
            results.append(
                torch.full((), float("nan"), device=symbol_index_grid.device)
            )
    return torch.stack(results)


def assigned_mapping_risk_metrics(
    symbol_index_grid: torch.Tensor,
    token_ids: torch.Tensor,
    sensitivity: torch.Tensor,
    band_data_positions: Sequence[Sequence[int]],
    spectral_notch: Optional[Mapping[str, Any]] = None,
    high_quantile: float = 0.80,
    mid_quantile: float = 0.50,
    risk_eligible_bands: Optional[torch.Tensor] = None,
    band_reference_quality: Optional[torch.Tensor] = None,
    risk_band_scores: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Audit actual assignments; true notch locations are used only for metrics.

    Carrier positions and notch start/width are in the frequency-ordered data
    list, not FFT bins. Multiple notch intervals are combined before computing
    exposure. Fractions count semantic symbols and ignore -1 padding.
    Token diversity averages each token's *own* occupied bands, so a group whose
    separate tokens each occupy one different band still has diversity one.
    Groups follow the mapper's >= quantile convention: ties belong to high
    first, then mid. Empty groups produce NaN fractions, never artificial zeros.
    Eligibility is optional mapper metadata; missing values remain NaN for
    legacy modes rather than claiming those modes rejected no bands.
    """
    if symbol_index_grid.ndim != 3:
        raise ValueError("symbol_index_grid must have shape [B,T,N_data]")
    batch, _, n_data = symbol_index_grid.shape
    if batch < 1 or n_data < 1:
        raise ValueError("symbol_index_grid batch and carrier dimensions must be nonempty")
    if sensitivity.ndim != 2 or sensitivity.shape[0] != batch or not sensitivity.shape[1]:
        raise ValueError("sensitivity must have nonempty shape [B,M]")
    if not 0.0 <= mid_quantile <= high_quantile <= 1.0:
        raise ValueError("Expected 0 <= mid_quantile <= high_quantile <= 1")
    if token_ids.ndim == 1:
        token_ids = token_ids.unsqueeze(0).expand(batch, -1)
    if token_ids.ndim != 2 or token_ids.shape[0] != batch:
        raise ValueError("token_ids must have shape [K] or [B,K]")
    n_tokens = sensitivity.shape[1]
    if not bool(torch.isfinite(sensitivity).all().item()):
        raise ValueError("sensitivity must contain only finite values")
    if token_ids.numel() and (
        bool((token_ids < 0).any().item()) or bool((token_ids >= n_tokens).any().item())
    ):
        raise ValueError("token_ids contains an index outside sensitivity")
    if bool((symbol_index_grid < -1).any().item()) or bool(
        (symbol_index_grid >= token_ids.shape[1]).any().item()
    ):
        raise ValueError("symbol_index_grid contains an invalid source index")

    bands = [list(positions) for positions in band_data_positions]
    n_bands = len(bands)
    if not n_bands or any(not positions for positions in bands):
        raise ValueError("band_data_positions must contain nonempty bands")
    if sorted(position for positions in bands for position in positions) != list(range(n_data)):
        raise ValueError("band_data_positions must partition all data carrier positions")
    carrier_to_band = torch.empty(n_data, dtype=torch.long)
    for band, positions in enumerate(bands):
        carrier_to_band[positions] = band

    optional = {
        "risk_eligible_bands": (risk_eligible_bands, (batch, n_bands)),
        "band_reference_quality": (band_reference_quality, (batch, n_data)),
        "risk_band_scores": (risk_band_scores, (batch, n_bands)),
    }
    for name, (value, expected_shape) in optional.items():
        if value is not None and tuple(value.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
    if risk_eligible_bands is not None and risk_eligible_bands.dtype != torch.bool:
        raise ValueError("risk_eligible_bands must be boolean")

    notch = dict(spectral_notch or {})
    nested = "notches" in notch
    if nested:
        regions = notch["notches"]
        if not isinstance(regions, (list, tuple)) or not regions:
            raise ValueError("spectral_notch.notches must be a nonempty sequence")
        if "count" in notch and int(notch["count"]) != len(regions):
            raise ValueError("spectral_notch.count must match notches")
    else:
        regions = [notch] if notch else []
    notch_mask = torch.zeros(n_data, dtype=torch.bool)
    for region in regions:
        if not isinstance(region, Mapping):
            raise ValueError("spectral_notch regions must be mappings")
        width = int(region.get("width", 0))
        start = int(region.get("start", 0)) if width else 0
        if nested and width < 1:
            raise ValueError("spectral_notch.notches widths must be positive")
        if width < 0 or start < 0 or start + width > n_data:
            raise ValueError("spectral_notch start/width is outside the data carriers")
        notch_mask[start : start + width] = True
    # One CPU transfer per input avoids repeated GPU synchronization per token.
    grids = symbol_index_grid.detach().to(device="cpu", dtype=torch.long)
    source_token_ids = token_ids.detach().to(device="cpu", dtype=torch.long)
    scores = sensitivity.detach().to(device="cpu", dtype=torch.float64)
    eligibility = None if risk_eligible_bands is None else risk_eligible_bands.detach().cpu()
    reference = None if band_reference_quality is None else band_reference_quality.detach().cpu()
    band_scores = None if risk_band_scores is None else risk_band_scores.detach().cpu()
    rows: List[Dict[str, float]] = []
    for sample in range(batch):
        flat = grids[sample].reshape(-1)
        valid = flat >= 0
        source_indices = flat[valid]
        carriers = torch.arange(flat.numel())[valid] % n_data
        assigned_tokens = source_token_ids[sample].index_select(0, source_indices)
        assigned_bands = carrier_to_band.index_select(0, carriers)
        in_notch = notch_mask[carriers]
        token_band_counts = torch.bincount(
            assigned_tokens * n_bands + assigned_bands,
            minlength=n_tokens * n_bands,
        ).reshape(n_tokens, n_bands)
        token_counts = token_band_counts.sum(dim=1)
        token_diversity = (token_band_counts > 0).sum(dim=1).double()
        token_max_fraction = token_band_counts.amax(dim=1).double() / token_counts.clamp_min(1)
        q_high = torch.quantile(scores[sample], high_quantile)
        q_mid = torch.quantile(scores[sample], mid_quantile)
        high = scores[sample] >= q_high
        mid = (scores[sample] >= q_mid) & ~high
        low = ~(high | mid)
        groups = {"": torch.ones(n_tokens, dtype=torch.bool), "high": high, "mid": mid, "low": low}
        row: Dict[str, float] = {}
        for group, tokens_in_group in groups.items():
            prefix = f"mapping_{group}_" if group else "mapping_"
            selected = tokens_in_group.index_select(0, assigned_tokens)
            count = int(selected.sum().item())
            row[f"{prefix}symbol_count"] = float(count)
            counts_by_band = torch.bincount(assigned_bands[selected], minlength=n_bands)
            row[f"{prefix}notch_fraction"] = (
                float(in_notch[selected].double().mean().item()) if count else float("nan")
            )
            for band in range(n_bands):
                row[f"{prefix}band_{band}_fraction"] = (
                    float(counts_by_band[band].item()) / count if count else float("nan")
                )
            row[f"{prefix}max_band_fraction"] = (
                float(counts_by_band.amax().item()) / count if count else float("nan")
            )
            present_tokens = tokens_in_group & (token_counts > 0)
            row[f"{prefix}token_band_count_mean"] = (
                float(token_diversity[present_tokens].mean().item())
                if bool(present_tokens.any().item()) else float("nan")
            )
            row[f"{prefix}token_max_band_fraction_mean"] = (
                float(token_max_fraction[present_tokens].mean().item())
                if bool(present_tokens.any().item()) else float("nan")
            )
            spill_key = f"risk_{group}_spill_fraction" if group else "risk_spill_fraction"
            row[spill_key] = (
                float((~eligibility[sample][assigned_bands[selected]]).double().mean().item())
                if eligibility is not None and count else float("nan")
            )
        row["risk_rejected_band_count"] = (
            float((~eligibility[sample]).sum().item()) if eligibility is not None else float("nan")
        )
        for band, positions in enumerate(bands):
            row[f"risk_band_{band}_eligible"] = (
                float(eligibility[sample, band].item()) if eligibility is not None else float("nan")
            )
            row[f"risk_band_{band}_score"] = (
                float(band_scores[sample, band].item()) if band_scores is not None else float("nan")
            )
            row[f"mapping_band_{band}_reference_quality"] = (
                float(reference[sample, positions].double().mean().item())
                if reference is not None else float("nan")
            )
        rows.append(row)
    return {
        key: torch.tensor([row[key] for row in rows], dtype=torch.float64, device=symbol_index_grid.device)
        for key in rows[0]
    }


class MetricAccumulator:
    def __init__(
        self,
        outage_threshold: float = 25.0,
        compute_msssim: bool = True,
        compute_lpips: bool = False,
        device: Optional[torch.device] = None,
        lpips_model=None,
    ) -> None:
        self.outage_threshold = float(outage_threshold)
        self.compute_msssim = bool(compute_msssim)
        self.compute_lpips = bool(compute_lpips)
        self.device = device or torch.device("cpu")
        self.rows: List[Dict[str, Any]] = []
        self.lpips_model = lpips_model

    def _msssim(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        try:
            from pytorch_msssim import ms_ssim
        except ImportError as error:
            raise RuntimeError("MS-SSIM evaluation requires pytorch-msssim") from error
        return ms_ssim(reconstruction, target, data_range=1.0, size_average=False)

    def _lpips(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.lpips_model is None:
            try:
                import lpips
            except ImportError as error:
                raise RuntimeError("LPIPS evaluation requires the lpips package") from error
            self.lpips_model = lpips.LPIPS(net="alex").to(self.device).eval()
        return self.lpips_model(
            2.0 * reconstruction.float() - 1.0,
            2.0 * target.float() - 1.0,
        ).flatten()

    @torch.no_grad()
    def add(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        identifiers: Iterable[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        reconstruction = reconstruction.clamp(0.0, 1.0)
        psnr = per_sample_psnr(reconstruction, target)
        msssim = self._msssim(reconstruction.float(), target.float()) if self.compute_msssim else None
        lpips_values = self._lpips(reconstruction, target) if self.compute_lpips else None
        metadata = dict(metadata or {})
        for index, identifier in enumerate(identifiers):
            sample_metadata = {}
            for key, value in metadata.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and value.shape[0] == psnr.shape[0]
                ):
                    item = value[index].detach().cpu()
                    sample_metadata[key] = (
                        float(item) if item.numel() == 1 else item.tolist()
                    )
                elif (
                    isinstance(value, (list, tuple))
                    and len(value) == psnr.shape[0]
                ):
                    sample_metadata[key] = value[index]
                else:
                    sample_metadata[key] = value
            row = {
                "image_id": str(identifier),
                "psnr": float(psnr[index].cpu()),
                **sample_metadata,
            }
            if msssim is not None:
                row["ms_ssim"] = float(msssim[index].cpu())
            if lpips_values is not None:
                row["lpips"] = float(lpips_values[index].cpu())
            self.rows.append(row)

    def summary(self) -> Dict[str, Any]:
        if not self.rows:
            raise RuntimeError("No metric observations were added")
        psnr = np.asarray([row["psnr"] for row in self.rows], dtype=np.float64)
        result = {
            "count": int(psnr.size),
            "psnr_mean": float(psnr.mean()),
            "psnr_std": float(psnr.std()),
            "psnr_p5": float(np.quantile(psnr, 0.05)),
            "psnr_iqr": float(np.quantile(psnr, 0.75) - np.quantile(psnr, 0.25)),
            "outage_threshold": self.outage_threshold,
            "outage_probability": float(np.mean(psnr < self.outage_threshold)),
        }
        for key in ("ms_ssim", "lpips"):
            values = [row[key] for row in self.rows if key in row]
            if values:
                result[f"{key}_mean"] = float(np.mean(values))
        for key in (
            "semantic_cbr",
            "padded_data_ratio",
            "pilot_overhead_ratio",
            "guard_dc_overhead_ratio",
            "cp_overhead_ratio",
            "air_interface_ratio",
            "high_sensitivity_carrier_correlation",
            "notch_count",
            "notch_total_width",
            "notch_width",
            "notch_attenuation_db",
        ):
            values = [row[key] for row in self.rows if key in row]
            if values:
                numeric = np.asarray(values, dtype=np.float64)
                if np.isfinite(numeric).any():
                    result[f"{key}_mean"] = float(np.nanmean(numeric))
        diagnostic_keys = {
            key for row in self.rows for key in row
            if key.startswith(("mapping_", "risk_", "swap_", "energy_", "residual_"))
        }
        for key in sorted(diagnostic_keys):
            numeric = np.asarray(
                [row.get(key, float("nan")) for row in self.rows], dtype=np.float64
            )
            finite = numeric[np.isfinite(numeric)]
            result[f"{key}_mean"] = float(finite.mean()) if finite.size else None
        return result
