from __future__ import annotations

import torch.nn as nn


class OfficialSwinCodec(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


def build_official_swinjscc(
    image_size: int = 256,
    model_size: str = "base",
    latent_channels: int = 96,
    model_variant: str = "SwinJSCC_w/o_SAandRA",
) -> nn.Module:
    """Construct the official semcomm/SwinJSCC encoder and decoder.

    The official repository must already be on ``sys.path``; the external
    backbone loader does this when ``repo_path`` is configured. This factory
    mirrors the architecture settings in the official ``main.py`` while leaving
    the repository's original channel layer unused.
    """
    try:
        from net.decoder import create_decoder
        from net.encoder import create_encoder
    except ImportError as error:
        raise RuntimeError(
            "Official SwinJSCC was not found. Clone semcomm/SwinJSCC and set "
            "model.backbone.repo_path to that directory."
        ) from error

    model_size = model_size.lower()
    depths_by_size = {
        "small": [2, 2, 2, 2],
        "base": [2, 2, 6, 2],
        "large": [2, 2, 18, 2],
    }
    if model_size not in depths_by_size:
        raise ValueError("model_size must be small, base, or large")
    encoder_depths = depths_by_size[model_size]
    decoder_depths = list(reversed(encoder_depths))
    encoder_kwargs = dict(
        model=model_variant,
        img_size=(image_size, image_size),
        patch_size=2,
        in_chans=3,
        embed_dims=[128, 192, 256, 320],
        depths=encoder_depths,
        num_heads=[4, 6, 8, 10],
        C=int(latent_channels),
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    )
    decoder_kwargs = dict(
        model=model_variant,
        img_size=(image_size, image_size),
        embed_dims=[320, 256, 192, 128],
        depths=decoder_depths,
        num_heads=[10, 8, 6, 4],
        C=int(latent_channels),
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    )
    return OfficialSwinCodec(
        encoder=create_encoder(**encoder_kwargs),
        decoder=create_decoder(**decoder_kwargs),
    )
