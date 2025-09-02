import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("Performance\\PRA.txt", sep="\s+", header=None)


df.columns = [
    "Samples",
    "Precision_ST_SEE", "Precision_HRM", "Precision_TwoStage",
    "Recall_ST_SEE", "Recall_HRM", "Recall_TwoStage",
    "Accuracy_ST_SEE", "Accuracy_HRM", "Accuracy_TwoStage"
]

samples = df["Samples"]


colors = ['#1f77b4', '#d62728', '#ff7f0e']
bar_width = 0.25
x = np.arange(len(samples))

def plot_metric(metric, labels, filename):
    plt.figure(figsize=(12, 6))
    plt.bar(x - bar_width, df[f"{metric}_ST_SEE"], width=bar_width, color=colors[0], label=labels[0])
    plt.bar(x, df[f"{metric}_HRM"], width=bar_width, color=colors[1], label=labels[1])
    plt.bar(x + bar_width, df[f"{metric}_TwoStage"], width=bar_width, color=colors[2], label=labels[2])

    plt.xlabel("Samples", fontsize=12, fontweight='bold')
    plt.ylabel(metric, fontsize=12, fontweight='bold')
    plt.title(f"{metric} Comparison", fontsize=14, fontweight='bold')
    plt.xticks(x, samples, rotation=45, fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


plot_metric("Precision", ["ST-SEE", "Autocorrelation-based HRM [1]", "Two-stage framework [2]"], "precision_comparison.png")
plot_metric("Recall", ["ST-SEE", "Autocorrelation-based HRM [1]", "Two-stage framework [2]"], "recall_comparison.png")
plot_metric("Accuracy", ["ST-SEE", "Autocorrelation-based HRM [1]", "Two-stage framework [2]"], "accuracy_comparison.png")
