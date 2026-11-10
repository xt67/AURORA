"""Render README figures from the recorded feasibility run; no model execution."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/feasibility_v2/run_01"))
    parser.add_argument("--output", type=Path, default=Path("docs/images"))
    args = parser.parse_args()
    report = json.loads((args.run / "feasibility_result.json").read_text())
    assert report["status"] == "completed_feasibility_not_publication"
    assert report["source_count"] == 52 and report["epochs"] == 2
    g = np.load(args.run / "gan/g_losses.npy")
    d = np.load(args.run / "gan/d_losses.npy")
    assert g.shape == d.shape == (2,) and np.isfinite([g, d]).all()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor="#f5f8fc")
    ax.set_facecolor("#f5f8fc")
    ax.plot([1, 2], g, "o-", color="#007f82", linewidth=3, label="Generator")
    ax.plot([1, 2], d, "s-", color="#7057bb", linewidth=3, label="Discriminator")
    for values, offset in [(g, .025), (d, -.04)]:
        for epoch, value in enumerate(values, 1):
            ax.annotate(f"{value:.4f}", (epoch, value + offset), ha="center")
    ax.set(xlim=(.8, 2.2), ylim=(0, .8), xticks=[1, 2], xlabel="Epoch", ylabel="Mean BCE loss")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="upper right", frameon=False)
    fig.suptitle("AURORA | Two-epoch execution feasibility", fontsize=21, weight="bold", x=.08, ha="left")
    fig.text(.08, .88, "52 real TB training images  /  seed 42  /  DCGAN  /  RTX 4050", color="#46546b")
    fig.text(.08, .025, "Measured training losses only. Image quality and downstream utility have not been established.", fontsize=11)
    fig.subplots_adjust(top=.80, bottom=.16, left=.08, right=.96)
    fig.savefig(args.output / "feasibility-losses.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#101c30")
    ax.set(xlim=(0, 14), ylim=(0, 7))
    ax.axis("off")
    ax.text(.5, 6.35, "AURORA", fontsize=30, color="white", weight="bold")
    ax.text(.5, 5.85, "From real training images to controlled augmentation experiments", color="#bdcde3", fontsize=15)

    def box(x, y, w, h, title, body, color="#20354f"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12", facecolor=color, edgecolor="#496581"))
        ax.text(x + .18, y + h - .38, title, color="white", weight="bold", fontsize=14)
        ax.text(x + .18, y + h - .72, body, color="#d0dceb", fontsize=11, va="top", linespacing=1.5)

    box(.5, 3.65, 3.45, 1.65, "800 original images", "206 flagged candidate pairs\n210 images excluded automatically\nOriginals preserved")
    box(5, 3.65, 3.5, 1.65, "590 retained images", "413 train / 89 validation / 88 test\n289 normal / 301 TB-labelled\nNo patient identifiers")
    box(9.55, 3.65, 3.9, 1.65, "Holdouts stay real", "Validation and test kept separate\nNo generator image access\nNo clinical validation claimed")
    box(.5, .85, 5.5, 2, "COMPLETED: feasibility", "52 real TB training images / 25% budget / seed 42\nDCGAN: two epochs, four batches per epoch\nExact IDs, hashes, runtime and GPU memory recorded", "#124649")
    box(7, .85, 6.45, 2, "PROPOSED: publication comparison", "A  Real only                         B  Repeated real TB\nC  Mild TB augmentation        D  DCGAN synthetic TB\n3 budgets × 3 seeds; matched added-TB counts for B/D")
    for start, end in [((4.05, 4.45), (4.85, 4.45)), ((8.6, 4.45), (9.4, 4.45)), ((6.6, 3.5), (3.3, 3.0)), ((6.8, 3.5), (10, 3.0))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#73d9cf", "lw": 2})
    ax.text(.5, .28, "Experimental research prototype • Automated exclusions are not confirmed duplicates • Publication experiments pending", color="#bdcde3", fontsize=10)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(args.output / "aurora-workflow.png", dpi=160)
    plt.close(fig)
    print("Rendered workflow and measured feasibility-loss figures.")


if __name__ == "__main__":
    main()
