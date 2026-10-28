"""Print the audited two-notch CSI sweep as a compact terminal table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


CSI_LABELS = ("nmse_-10db", "nmse_-5db")
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
        key = (row["csi"], row["notch"].lower(), row["mode"])
        if key in indexed:
            raise ValueError(f"Duplicate summary group {key} in {path}")
        indexed[key] = row
    expected = {
        (csi, notch, mode)
        for csi in CSI_LABELS
        for notch in ("false", "true")
        for mode in MODES
    }
    if set(indexed) != expected:
        raise ValueError(
            f"Incomplete summary in {path}: missing={expected - set(indexed)}, "
            f"unexpected={set(indexed) - expected}"
        )

    dataset = next(iter(datasets)).upper()
    for csi in CSI_LABELS:
        csi_db = csi.removeprefix("nmse_").removesuffix("db")
        print(f"\n===== {dataset} | CSI NMSE {csi_db} dB =====")
        print(f"Directory: {path.parent}")
        print(
            f"{'Scene':<13} {'Mode':<21} {'Samples':>7} {'PSNR':>9} "
            f"{'MS-SSIM':>9} {'LPIPS':>9} {'Outage%':>9}"
        )
        for notch, scene in (("false", "no_notch"), ("true", "two_notches")):
            for mode in MODES:
                row = indexed[(csi, notch, mode)]
                print(
                    f"{scene:<13} {mode:<21} {int(row['count']):>7d} "
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
