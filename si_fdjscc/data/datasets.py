from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ImageDirectoryDataset(Dataset):
    EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def __init__(
        self,
        root: str,
        crop_size: Optional[int] = None,
        training: bool = False,
    ) -> None:
        from torchvision import transforms

        self.root = Path(root).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")
        self.paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in self.EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No supported images found under {self.root}")
        operations = []
        if crop_size:
            if training:
                operations.extend(
                    [
                        transforms.RandomCrop(crop_size),
                        transforms.RandomHorizontalFlip(),
                    ]
                )
            else:
                operations.append(transforms.CenterCrop(crop_size))
        operations.append(transforms.ToTensor())
        self.transform = transforms.Compose(operations)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.stem


def build_dataset(config: Mapping[str, Any], training: bool) -> Dataset:
    name = str(config.get("name", "image_directory")).lower()
    if name in {"cifar10", "cifar-10"}:
        from torchvision.datasets import CIFAR10
        from torchvision.transforms import ToTensor

        return CIFAR10(
            root=str(config["root"]),
            train=training,
            transform=ToTensor(),
            download=bool(config.get("download", False)),
        )
    if name in {"image_directory", "div2k", "kodak", "clic21", "bsd100"}:
        return ImageDirectoryDataset(
            root=str(config["root"]),
            crop_size=config.get("crop_size"),
            training=training,
        )
    raise ValueError(f"Unsupported dataset: {name}")


def build_dataloader(config: Mapping[str, Any], training: bool) -> DataLoader:
    dataset = build_dataset(config, training=training)
    batch_size = int(config.get("batch_size", 1))
    if not training and config.get("crop_size") is None and batch_size != 1:
        raise ValueError("Full-resolution evaluation requires batch_size=1 for variable image sizes")
    generator = torch.Generator()
    generator.manual_seed(int(config.get("seed", 42)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(config.get("shuffle", training)),
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=bool(config.get("pin_memory", True)),
        drop_last=bool(config.get("drop_last", training)),
        generator=generator,
    )

