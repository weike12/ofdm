from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def plot_summary(summary_path: str, output_path: str, metric: str = "psnr_mean") -> None:
    import matplotlib.pyplot as plt

    with Path(summary_path).open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    groups = defaultdict(list)
    for row in rows:
        label = f"{row['mapper']} | {row['csi']} | L={row['n_taps']},tau={row['tau']}"
        groups[label].append((float(row["snr_db"]), float(row[metric])))
    figure, axis = plt.subplots(figsize=(8, 5))
    for label, points in sorted(groups.items()):
        points.sort()
        axis.plot([x for x, _ in points], [y for _, y in points], marker="o", label=label)
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel(metric)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary")
    parser.add_argument("output")
    parser.add_argument("--metric", default="psnr_mean")
    args = parser.parse_args()
    plot_summary(args.summary, args.output, args.metric)


if __name__ == "__main__":
    main()

