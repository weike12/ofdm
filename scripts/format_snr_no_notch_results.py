"""Print the audited no-notch five-SNR baseline sweep as terminal tables."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


SNR_VALUES = (10, 5, 0, -5, -10)
MODES = (
    "sequential",
    "random",
    "greedy",
    "greedy_token_random",
    "greedy_token_energy",
)


def print_table(path: Path) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty summary: {path}")
    datasets = {row["dataset"].lower() for row in rows}
    if len(datasets) != 1:
        raise ValueError(f"Expected one dataset in {path}; found {datasets}")
    indexed = {}
    for row in rows:
        if row["csi"] != "nmse_-10db" or row["notch"].lower() != "false":
            raise ValueError(f"Expected CSI NMSE -10 dB and no notch in {path}")
        if int(row["notch_count"]) != 0:
            raise ValueError(f"Unexpected realized notch count in {path}")
        snr = float(row["snr_db"])
        key = (snr, row["mode"])
        if key in indexed:
            raise ValueError(f"Duplicate summary group {key} in {path}")
        for field in ("psnr_mean", "ms_ssim_mean", "lpips_mean", "outage_probability"):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Non-finite {field} for {key} in {path}")
        indexed[key] = row
    expected = {(snr, mode) for snr in SNR_VALUES for mode in MODES}
    if set(indexed) != expected:
        raise ValueError(
            f"Incomplete summary in {path}: missing={expected - set(indexed)}, "
            f"unexpected={set(indexed) - expected}"
        )

    dataset = next(iter(datasets)).upper()
    for snr in SNR_VALUES:
        print(f"\n===== {dataset} | SNR {snr} dB | CSI NMSE -10 dB | no_notch =====")
        print(f"Directory: {path.parent}")
        print(
            f"{'Mode':<21} {'Samples':>7} {'PSNR':>9} "
            f"{'MS-SSIM':>9} {'LPIPS':>9} {'Outage%':>9}"
        )
        for mode in MODES:
            row = indexed[(snr, mode)]
            print(
                f"{mode:<21} {int(row['count']):>7d} "
                f"{float(row['psnr_mean']):>9.4f} "
                f"{float(row['ms_ssim_mean']):>9.6f} "
                f"{float(row['lpips_mean']):>9.6f} "
                f"{100 * float(row['outage_probability']):>9.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.summary_csv:
        print_table(path)


if __name__ == "__main__":
    main()
