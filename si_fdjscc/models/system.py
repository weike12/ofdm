from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import torch
from torch import nn

from ..comm.ofdm_channel import (
    RECEIVER_CSI_SOURCES,
    ChannelState,
    OFDMFrequencySelectiveChannel,
)
from ..comm.semantic_mapper import SemanticFrequencyMapper
from .importance_head import ImportanceHead
from .latent_adapter import ComplexAdapter
from .swinjscc_backbone import CodecBackbone


CHANNEL_CONTEXT_DIM = 11
MAPPER_CSI_SOURCES = ("estimated", "oracle", "temporal_mean")


def _batch_scalar(
    value,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    default: float,
) -> torch.Tensor:
    if value is None:
        return torch.full((batch_size,), default, device=device, dtype=dtype)
    result = torch.as_tensor(value, device=device, dtype=dtype)
    if result.ndim == 0:
        result = result.expand(batch_size)
    result = result.reshape(-1)
    if result.numel() != batch_size:
        raise ValueError(f"Expected a scalar or {batch_size} values")
    return result


def summarize_channel_context(
    h_hat: Optional[torch.Tensor],
    snr_db: torch.Tensor,
    csi_nmse_db,
    data_indices: torch.Tensor,
) -> torch.Tensor:
    """Build the lightweight channel summary required by the SDS predictor.

    The eleven features are log-power mean/variance/minimum/25th/50th/75th/
    maximum, normalised SNR, normalised CSI NMSE, a perfect-CSI flag, and a
    transmitter-CSI availability flag.  ``pilot_ls`` deliberately supplies no
    post-reception estimate to the transmitter and therefore produces zero
    frequency statistics with the availability flag cleared.
    """
    batch_size = int(snr_db.reshape(-1).numel())
    real_dtype = snr_db.dtype if snr_db.is_floating_point() else torch.float32
    device = snr_db.device
    if h_hat is None:
        statistics = torch.zeros(
            batch_size, 7, device=device, dtype=real_dtype
        )
        has_csi = torch.zeros(batch_size, 1, device=device, dtype=real_dtype)
    else:
        if h_hat.ndim == 2:
            h_hat = h_hat[:, None, :]
        if h_hat.ndim != 3 or h_hat.shape[0] != batch_size:
            raise ValueError("h_hat must have shape [B,N] or [B,T,N]")
        data_response = h_hat.index_select(-1, data_indices.to(h_hat.device))
        log_power = torch.log1p(data_response.abs().square()).float().flatten(1)
        quantiles = torch.quantile(
            log_power,
            torch.tensor([0.25, 0.50, 0.75], device=log_power.device),
            dim=1,
        ).transpose(0, 1)
        statistics = torch.cat(
            (
                log_power.mean(dim=1, keepdim=True),
                log_power.var(dim=1, unbiased=False, keepdim=True),
                log_power.amin(dim=1, keepdim=True),
                quantiles,
                log_power.amax(dim=1, keepdim=True),
            ),
            dim=1,
        ).to(device=device, dtype=real_dtype)
        has_csi = torch.ones(batch_size, 1, device=device, dtype=real_dtype)
    snr = _batch_scalar(snr_db, batch_size, device, real_dtype, 0.0)
    perfect_csi = h_hat is not None and csi_nmse_db is None
    nmse = _batch_scalar(
        csi_nmse_db,
        batch_size,
        device,
        real_dtype,
        -60.0,
    )
    return torch.cat(
        (
            statistics,
            (snr / 20.0).unsqueeze(1),
            (nmse.clamp(-60.0, 20.0) / 30.0).unsqueeze(1),
            torch.full(
                (batch_size, 1),
                float(perfect_csi),
                device=device,
                dtype=real_dtype,
            ),
            has_csi,
        ),
        dim=1,
    )


class SDSUFDJSCC(nn.Module):
    """Channel-conditioned SDS-UFD-JSCC image transmission system."""

    def __init__(
        self,
        backbone: CodecBackbone,
        importance_head: ImportanceHead,
        complex_adapter: Optional[ComplexAdapter] = None,
        mapper: Optional[SemanticFrequencyMapper] = None,
        channel: Optional[OFDMFrequencySelectiveChannel] = None,
    ) -> None:
        super().__init__()
        if backbone.latent_channels != importance_head.channels:
            raise ValueError("Backbone latent channels must match ImportanceHead channels")
        self.backbone = backbone
        self.importance_head = importance_head
        self.complex_adapter = complex_adapter or ComplexAdapter()
        self.channel = channel or OFDMFrequencySelectiveChannel()
        self.mapper = mapper or SemanticFrequencyMapper(plan=self.channel.plan)
        if self.mapper.plan != self.channel.plan:
            raise ValueError("Mapper and channel must use the same SubcarrierPlan")

    def clean_reconstruction(self, image: torch.Tensor):
        tokens, backbone_meta = self.backbone.encode(image)
        return self.backbone.decode(tokens, backbone_meta), tokens, backbone_meta

    @staticmethod
    def _grid_from_permutation(
        symbols: torch.Tensor,
        symbol_index_grid: torch.Tensor,
    ) -> torch.Tensor:
        batch = symbols.shape[0]
        if symbol_index_grid.shape[0] != batch:
            raise ValueError("Permutation batch size does not match symbols")
        valid = symbol_index_grid >= 0
        indices = symbol_index_grid.reshape(batch, -1)
        gathered = torch.gather(symbols, 1, indices.clamp_min(0))
        return (
            gathered.view_as(symbol_index_grid).to(symbols.dtype)
            * valid.to(symbols.dtype)
        )

    def reconstruct_with_fixed_transmission(
        self,
        tokens: torch.Tensor,
        backbone_meta: Mapping[str, Any],
        snr_db: torch.Tensor,
        transmission_aux: Mapping[str, Any],
    ) -> torch.Tensor:
        """Decode perturbed tokens with the exact clean mapping/channel draws.

        Reusing the permutation, channel impulse response, CSI error, and noise
        seed isolates the reconstruction damage caused by the semantic-unit
        perturbation, which is the supervision target defined in the revised
        research plan.  The returned shape is always ``[D,B,C,H,W]``.
        """
        symbols, complex_meta = self.complex_adapter(tokens)
        reconstructions = []
        for draw in transmission_aux["draws"]:
            permutation = draw["permutation"]
            if permutation.n_symbols != symbols.shape[1]:
                raise ValueError("Perturbed tokens changed the semantic symbol count")
            data_grid = self._grid_from_permutation(
                symbols, permutation.symbol_index_grid
            )
            state = draw.get("channel_state")
            if not isinstance(state, ChannelState):
                raise ValueError(
                    "transmission_aux does not contain reusable ChannelState objects"
                )
            equalized_grid, _ = self.channel(
                data_grid,
                snr_db=snr_db,
                csi_mode=state.csi_mode,
                csi_nmse_db=draw.get("csi_nmse_db"),
                channel_state=state,
                channel_seed=draw.get("channel_seed"),
                receiver_csi_source=draw.get("receiver_csi_source", "estimated"),
            )
            symbols_hat = self.mapper.unmap(equalized_grid, permutation)
            tokens_hat = self.complex_adapter.inverse(symbols_hat, complex_meta)
            reconstructions.append(self.backbone.decode(tokens_hat, backbone_meta))
        if not reconstructions:
            raise ValueError("transmission_aux contains no channel draws")
        return torch.stack(reconstructions, dim=0)

    def forward(
        self,
        image: torch.Tensor,
        snr_db: torch.Tensor,
        channel_cfg: Optional[Mapping[str, Any]] = None,
        mapper_mode: str = "proposed_ufd",
        return_aux: bool = False,
        n_channel_draws: int = 1,
        detach_importance_input: bool = False,
    ):
        if n_channel_draws < 1:
            raise ValueError("n_channel_draws must be positive")
        cfg = dict(channel_cfg or {})
        mapper_csi_source = str(cfg.get("mapper_csi_source", "estimated")).lower()
        if mapper_csi_source not in MAPPER_CSI_SOURCES:
            raise ValueError(
                "mapper_csi_source must be one of "
                + ", ".join(MAPPER_CSI_SOURCES)
            )
        if str(mapper_mode).lower() in {
            "proposed_ufd_bandmean", "proposed_ufd_risk", "proposed_ufd_swap",
            "greedy_token_random", "greedy_token_energy",
        }:
            if mapper_csi_source != "estimated":
                raise ValueError(
                    "Risk/bandmean/swap/token mapping requires mapper_csi_source='estimated' "
                    "to retain the original per-symbol CSI for slot placement"
                )
        receiver_csi_source = str(cfg.get("receiver_csi_source", "estimated")).lower()
        if receiver_csi_source not in RECEIVER_CSI_SOURCES:
            raise ValueError(
                "receiver_csi_source must be one of "
                + ", ".join(RECEIVER_CSI_SOURCES)
            )
        snr_db = _batch_scalar(
            snr_db,
            image.shape[0],
            image.device,
            image.dtype if image.is_floating_point() else torch.float32,
            0.0,
        )
        tokens, backbone_meta = self.backbone.encode(image)
        symbols, complex_meta = self.complex_adapter(tokens)
        n_ofdm_symbols = math.ceil(symbols.shape[1] / self.channel.plan.n_data)
        source_samples = int(image.shape[1] * image.shape[2] * image.shape[3])
        semantic_cbr = symbols.shape[1] / source_samples
        padded_data_ratio = (
            n_ofdm_symbols * self.channel.plan.n_data / source_samples
        )
        pilot_overhead_ratio = (
            n_ofdm_symbols * self.channel.plan.n_pilot / source_samples
        )
        guard_dc_overhead_ratio = (
            n_ofdm_symbols
            * (
                self.channel.plan.n_fft
                - self.channel.plan.n_data
                - self.channel.plan.n_pilot
            )
            / source_samples
        )
        cp_overhead_ratio = (
            n_ofdm_symbols * self.channel.modem.n_cp / source_samples
        )
        air_interface_ratio = (
            n_ofdm_symbols
            * (self.channel.plan.n_fft + self.channel.modem.n_cp)
            / source_samples
        )

        base_seed = cfg.get("channel_seed")
        if base_seed is None:
            # The sampled seed is retained in auxiliary state so clean and
            # perturbed paths can reuse the exact same multipath, CSI error,
            # and AWGN realization when constructing SDS labels.
            base_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        prepared_states = []
        draw_seeds = []
        for draw in range(n_channel_draws):
            draw_seed = int(base_seed) + draw * 10_000
            draw_seeds.append(draw_seed)
            prepared_states.append(
                self.channel.prepare_state(
                batch_size=image.shape[0],
                n_ofdm_symbols=n_ofdm_symbols,
                device=image.device,
                dtype=symbols.dtype,
                csi_mode=str(cfg.get("csi_mode", "perfect")),
                csi_nmse_db=cfg.get("csi_nmse_db"),
                channel_seed=draw_seed,
                spectral_notch=cfg.get("spectral_notch"),
            )
            )

        data_indices = self.channel.plan.tensor_indices("data", image.device)
        contexts = [
            summarize_channel_context(
                state.h_hat,
                snr_db,
                cfg.get("csi_nmse_db"),
                data_indices,
            )
            for state in prepared_states
        ]
        channel_context = torch.stack(contexts, dim=0).mean(dim=0)
        # The SDS supervision target is detached. During robust fine-tuning,
        # its loss should train the lightweight predictor rather than bend the
        # pretrained encoder features away from reconstruction.
        importance_tokens = tokens.detach() if detach_importance_input else tokens
        importance, importance_logits = self.importance_head(
            importance_tokens,
            channel_features=channel_context,
            return_logits=True,
        )
        importance_scores = self.importance_head.positive_scores(importance_logits)

        draw_importance = []
        draw_importance_logits = []
        for context in contexts:
            draw_scores, draw_logits = self.importance_head(
                importance_tokens,
                channel_features=context,
                return_logits=True,
            )
            draw_importance.append(draw_scores)
            draw_importance_logits.append(draw_logits)

        correlation_matrix = self.channel.data_subcarrier_correlation(
            device=symbols.device,
            dtype=symbols.real.dtype,
        )
        reconstructions = []
        draw_aux = []
        noise_variance = torch.pow(10.0, -snr_db / 10.0)
        for draw, (draw_seed, state, draw_scores, draw_logits, context) in enumerate(
            zip(
                draw_seeds,
                prepared_states,
                draw_importance,
                draw_importance_logits,
                contexts,
            )
        ):
            # Change only the CSI used for placement. SDS context and receiver
            # equalization continue to use the original channel estimate.
            mapper_h = state.h_hat
            if mapper_csi_source == "oracle":
                mapper_h = state.h_freq
            elif mapper_csi_source == "temporal_mean":
                if mapper_h is None or mapper_h.ndim != 3:
                    raise ValueError(
                        "temporal_mean mapper CSI requires a [B,T,N_fft] "
                        "pre-transmission channel estimate"
                    )
                mapper_h = mapper_h.mean(dim=1, keepdim=True)
            data_grid, permutation = self.mapper.map(
                symbols,
                complex_meta.token_ids,
                draw_scores,
                mapper_h,
                mode=mapper_mode,
                seed=draw_seed,
                correlation_matrix=correlation_matrix,
                noise_variance=noise_variance,
            )
            equalized_grid, channel_aux = self.channel(
                data_grid,
                snr_db=snr_db,
                csi_mode=state.csi_mode,
                csi_nmse_db=cfg.get("csi_nmse_db"),
                channel_state=state,
                channel_seed=draw_seed,
                return_waveform=bool(cfg.get("return_waveform", False)),
                receiver_csi_source=receiver_csi_source,
            )
            symbols_hat = self.mapper.unmap(equalized_grid, permutation)
            tokens_hat = self.complex_adapter.inverse(symbols_hat, complex_meta)
            reconstruction = self.backbone.decode(tokens_hat, backbone_meta)
            reconstructions.append(reconstruction)
            draw_aux.append(
                {
                    "z_hat": tokens_hat,
                    "symbols_rx": symbols_hat,
                    "permutation": permutation,
                    "channel_state": state,
                    "channel_seed": draw_seed,
                    "csi_nmse_db": cfg.get("csi_nmse_db"),
                    "mapper_csi_source": mapper_csi_source,
                    "importance": draw_scores,
                    "importance_logits": draw_logits,
                    "importance_scores": self.importance_head.positive_scores(
                        draw_logits
                    ),
                    "channel_context": context,
                    **channel_aux,
                }
            )

        stacked = torch.stack(reconstructions, dim=0)
        output = stacked[0] if n_channel_draws == 1 else stacked
        if not return_aux:
            return output
        return output, {
            "z": tokens,
            "symbols_tx": symbols,
            "importance": importance,
            "importance_logits": importance_logits,
            "importance_scores": importance_scores,
            "channel_context": channel_context,
            "complex_metadata": complex_meta,
            "backbone_metadata": backbone_meta,
            "reconstructions": stacked,
            "draws": draw_aux,
            "n_complex_symbols": symbols.shape[1],
            "n_ofdm_symbols": n_ofdm_symbols,
            "semantic_cbr": semantic_cbr,
            "padded_data_ratio": padded_data_ratio,
            "pilot_overhead_ratio": pilot_overhead_ratio,
            "guard_dc_overhead_ratio": guard_dc_overhead_ratio,
            "cp_overhead_ratio": cp_overhead_ratio,
            "air_interface_ratio": air_interface_ratio,
        }


# Backward-compatible public name for existing checkpoints and import sites.
SIFDJSCC = SDSUFDJSCC
