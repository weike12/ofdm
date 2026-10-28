from __future__ import annotations

import importlib
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional


class CodecBackbone(nn.Module):
    latent_channels: int

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError

    def decode(self, tokens: torch.Tensor, metadata: Mapping[str, Any]) -> torch.Tensor:
        raise NotImplementedError


class TinyConvBackbone(CodecBackbone):
    """Small self-contained codec for unit tests and CIFAR smoke tests."""

    def __init__(self, latent_channels: int = 32, hidden_channels: int = 48) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden_channels, latent_channels, 5, stride=2, padding=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, hidden_channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_channels, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, Dict[str, Any]]:
        latent = self.encoder(image)
        batch, channels, height, width = latent.shape
        tokens = latent.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        return tokens, {
            "latent_layout": "bchw",
            "latent_hw": (height, width),
            "image_hw": tuple(image.shape[-2:]),
        }

    def decode(self, tokens: torch.Tensor, metadata: Mapping[str, Any]) -> torch.Tensor:
        height, width = metadata["latent_hw"]
        batch, _, channels = tokens.shape
        latent = tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        reconstruction = self.decoder(latent)
        image_height, image_width = metadata["image_hw"]
        return reconstruction[..., :image_height, :image_width]


def _load_object(specification: str, repo_path: Optional[str] = None):
    if repo_path:
        path = str(Path(repo_path).expanduser().resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
    if ":" not in specification:
        raise ValueError("External factory must use 'python.module:object' syntax")
    module_name, object_name = specification.split(":", 1)
    return getattr(importlib.import_module(module_name), object_name)


def _extract_checkpoint_state(payload: Any) -> Mapping[str, torch.Tensor]:
    """Accept an official raw state_dict or a common wrapped checkpoint."""
    if not isinstance(payload, Mapping):
        raise TypeError("Backbone checkpoint must contain a mapping of parameter tensors")
    for key in ("model", "state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            payload = candidate
            break
    if not payload or not all(isinstance(value, torch.Tensor) for value in payload.values()):
        raise TypeError(
            "Backbone checkpoint does not look like a state_dict; expected tensor values"
        )
    state = OrderedDict((str(key), value) for key, value in payload.items())
    if all(key.startswith("module.") for key in state):
        state = OrderedDict((key[len("module."):], value) for key, value in state.items())
    return state


class ExternalCodecBackbone(CodecBackbone):
    """Adapter for an official/vendor codec without copying its implementation.

    The external model must expose configurable encoder and decoder callables.
    Latent layouts ``bmc``, ``bchw``, and ``bhwc`` are supported.
    """

    def __init__(
        self,
        model: nn.Module,
        latent_channels: int,
        latent_layout: str = "bhwc",
        encode_method: str = "encoder",
        decode_method: str = "decoder",
        encode_kwargs: Optional[Mapping[str, Any]] = None,
        decode_kwargs: Optional[Mapping[str, Any]] = None,
        resolution_downsample: Optional[int] = None,
        resolution_alignment: Optional[int] = None,
    ) -> None:
        super().__init__()
        if latent_layout not in {"bmc", "bchw", "bhwc"}:
            raise ValueError("latent_layout must be bmc, bchw, or bhwc")
        self.model = model
        self.latent_channels = int(latent_channels)
        self.latent_layout = latent_layout
        self.encode_method = encode_method
        self.decode_method = decode_method
        self.encode_kwargs = dict(encode_kwargs or {})
        self.decode_kwargs = dict(decode_kwargs or {})
        self.resolution_downsample = resolution_downsample
        if resolution_alignment is not None and int(resolution_alignment) < 1:
            raise ValueError("resolution_alignment must be positive")
        self.resolution_alignment = (
            None if resolution_alignment is None else int(resolution_alignment)
        )
        self._current_image_hw: Optional[Tuple[int, int]] = None

    def _encoder(self):
        return getattr(self.model, self.encode_method)

    def _decoder(self):
        return getattr(self.model, self.decode_method)

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, Dict[str, Any]]:
        height, width = image.shape[-2:]
        original_hw = (height, width)
        if self.resolution_alignment is not None:
            pad_height = (-height) % self.resolution_alignment
            pad_width = (-width) % self.resolution_alignment
            if pad_height or pad_width:
                image = functional.pad(
                    image,
                    (0, pad_width, 0, pad_height),
                    mode="replicate",
                )
                height, width = image.shape[-2:]
        if (
            self.resolution_downsample is not None
            and self._current_image_hw != (height, width)
        ):
            encoder_module = getattr(self.model, self.encode_method)
            decoder_module = getattr(self.model, self.decode_method)
            if hasattr(encoder_module, "update_resolution"):
                encoder_module.update_resolution(height, width)
            if hasattr(decoder_module, "update_resolution"):
                scale = 2 ** int(self.resolution_downsample)
                decoder_module.update_resolution(height // scale, width // scale)
            self._current_image_hw = (height, width)
        latent = self._encoder()(image, **self.encode_kwargs)
        if isinstance(latent, (tuple, list)):
            latent = latent[0]
        if not isinstance(latent, torch.Tensor):
            raise TypeError("External encoder must return a Tensor or a tuple whose first item is a Tensor")
        metadata: Dict[str, Any] = {
            "latent_layout": self.latent_layout,
            "latent_shape": tuple(latent.shape),
            "image_hw": original_hw,
            "padded_image_hw": (height, width),
        }
        if self.latent_layout == "bmc":
            tokens = latent
        elif self.latent_layout == "bchw":
            batch, channels, height, width = latent.shape
            tokens = latent.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        else:
            batch, height, width, channels = latent.shape
            tokens = latent.reshape(batch, height * width, channels)
        if tokens.shape[-1] != self.latent_channels:
            raise ValueError(
                f"Configured latent_channels={self.latent_channels}, encoder returned {tokens.shape[-1]}"
            )
        return tokens, metadata

    def decode(self, tokens: torch.Tensor, metadata: Mapping[str, Any]) -> torch.Tensor:
        shape = tuple(metadata["latent_shape"])
        if self.latent_layout == "bmc":
            latent = tokens.reshape(shape)
        elif self.latent_layout == "bchw":
            batch, channels, height, width = shape
            latent = tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        else:
            latent = tokens.reshape(shape)
        reconstruction = self._decoder()(latent, **self.decode_kwargs)
        if isinstance(reconstruction, (tuple, list)):
            reconstruction = reconstruction[0]
        image_height, image_width = metadata["image_hw"]
        return reconstruction[..., :image_height, :image_width]


def build_backbone(config: Mapping[str, Any]) -> CodecBackbone:
    backbone_type = str(config.get("type", "tiny")).lower()
    if backbone_type in {"tiny", "tiny_conv"}:
        return TinyConvBackbone(
            latent_channels=int(config.get("latent_channels", 32)),
            hidden_channels=int(config.get("hidden_channels", 48)),
        )
    if backbone_type not in {"external", "swinjscc"}:
        raise ValueError(f"Unsupported backbone type: {backbone_type}")

    factory = _load_object(str(config["factory"]), config.get("repo_path"))
    model = factory(**dict(config.get("factory_kwargs", {})))
    if not isinstance(model, nn.Module):
        raise TypeError("The configured backbone factory must return torch.nn.Module")
    checkpoint_path = config.get("checkpoint")
    if checkpoint_path:
        resolved_checkpoint = Path(str(checkpoint_path)).expanduser()
        if not resolved_checkpoint.is_file():
            raise FileNotFoundError(
                f"Backbone checkpoint does not exist: {resolved_checkpoint}"
            )
        payload = torch.load(str(resolved_checkpoint), map_location="cpu")
        state = _extract_checkpoint_state(payload)
        model.load_state_dict(
            state,
            strict=bool(config.get("strict_checkpoint", True)),
        )
        print(
            f"Loaded backbone checkpoint: {resolved_checkpoint} "
            f"({len(state)} tensors)",
            flush=True,
        )
    return ExternalCodecBackbone(
        model=model,
        latent_channels=int(config["latent_channels"]),
        latent_layout=str(config.get("latent_layout", "bhwc")),
        encode_method=str(config.get("encode_method", "encoder")),
        decode_method=str(config.get("decode_method", "decoder")),
        encode_kwargs=config.get("encode_kwargs"),
        decode_kwargs=config.get("decode_kwargs"),
        resolution_downsample=config.get("resolution_downsample"),
        resolution_alignment=config.get("resolution_alignment"),
    )
