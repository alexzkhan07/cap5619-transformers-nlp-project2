"""Generate figures for the Project II report from the numeric results."""

import os
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

# ----- Task I data (from results/task1_results.txt) -----
k_vals = [1, 2, 5, 10, 20]

groups = {
    "city-in-state": {
        "total": 2467,
        "cos": [0.00, 16.34, 38.02, 60.15, 87.19],
        "l2":  [0.12, 54.72, 80.71, 90.19, 96.03],
    },
    "family": {
        "total": 506,
        "cos": [4.15, 36.36, 61.26, 73.12, 91.70],
        "l2":  [76.09, 92.89, 98.42, 99.80, 100.00],
    },
    "gram2-opposite": {
        "total": 812,
        "cos": [0.25, 8.74, 20.94, 38.42, 70.07],
        "l2":  [4.80, 34.24, 52.71, 73.65, 99.51],
    },
}


def task1_per_group_figure(name: str, data: dict, fname: str):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(k_vals, data["cos"], marker="o", color="#1f77b4", label="Cosine similarity")
    ax.plot(k_vals, data["l2"],  marker="o", color="#ff7f0e", label="L2 distance")
    ax.set_xlabel("k")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Top-k Accuracy: {name} ({data['total']} analogies)")
    ax.set_xticks(k_vals)
    ax.set_ylim(-3, 105)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=180)
    plt.close(fig)


def task1_combined_figure():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True)
    colors = {"city-in-state": "#1f77b4", "family": "#2ca02c", "gram2-opposite": "#d62728"}

    # Left: cosine
    for gname, d in groups.items():
        axes[0].plot(k_vals, d["cos"], marker="o", color=colors[gname], label=gname)
    axes[0].set_title("Cosine similarity")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xticks(k_vals)
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(loc="lower right")

    # Right: L2
    for gname, d in groups.items():
        axes[1].plot(k_vals, d["l2"], marker="o", color=colors[gname], label=gname)
    axes[1].set_title("L2 distance")
    axes[1].set_xlabel("k")
    axes[1].set_xticks(k_vals)
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(loc="lower right")

    fig.suptitle("Word Analogy Top-k Accuracy by Group and Metric")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "task1_all_groups.png"), dpi=180)
    plt.close(fig)


def task1_cos_vs_l2_bar():
    """One grouped bar per group at k=1 and k=10, cos vs L2, to emphasise the gap."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    width = 0.2
    x_positions = [0, 1, 2]
    k_show = [1, 10]

    for i, k in enumerate(k_show):
        cos_vals = [groups[g]["cos"][k_vals.index(k)] for g in groups]
        l2_vals  = [groups[g]["l2"][k_vals.index(k)]  for g in groups]
        offset_c = -1.5 * width + i * 2 * width
        offset_l = -0.5 * width + i * 2 * width
        ax.bar([p + offset_c for p in x_positions], cos_vals, width=width,
               color="#1f77b4", alpha=0.6 + 0.4 * i,
               label=f"Cosine (k={k})")
        ax.bar([p + offset_l for p in x_positions], l2_vals, width=width,
               color="#ff7f0e", alpha=0.6 + 0.4 * i,
               label=f"L2 (k={k})")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(list(groups.keys()))
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cosine vs. L2: k = 1 and k = 10")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(ncol=2, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "task1_cos_vs_l2_bars.png"), dpi=180)
    plt.close(fig)


# ----- Task II plots -----
def task2_training_figure():
    epochs = [1, 2]
    losses = [0.6406, 0.4891]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(epochs, losses, marker="o", color="#1f77b4", label="Average training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("BERT Fine-tuning: Average Training Loss per Epoch")
    ax.set_xticks(epochs)
    ax.set_ylim(0, 0.8)
    for e, L in zip(epochs, losses):
        ax.annotate(f"{L:.4f}", xy=(e, L), xytext=(6, 6), textcoords="offset points")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "task2_train_loss.png"), dpi=180)
    plt.close(fig)


def task2_accuracy_figure():
    acc = 82.81
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    bar = ax.bar(["BERT (fine-tuned, 2 epochs)"], [acc], color="#ff7f0e", width=0.5)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Task II: Final Test Accuracy")
    for r in bar:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2.0, h + 1.0,
                f"{h:.2f}%", ha="center", va="bottom", fontsize=11)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "task2_accuracy.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    for g, d in groups.items():
        task1_per_group_figure(g, d, f"task1_{g.replace('-', '_')}.png")
    task1_combined_figure()
    task1_cos_vs_l2_bar()
    task2_training_figure()
    task2_accuracy_figure()
    print("Figures written to", OUT)
