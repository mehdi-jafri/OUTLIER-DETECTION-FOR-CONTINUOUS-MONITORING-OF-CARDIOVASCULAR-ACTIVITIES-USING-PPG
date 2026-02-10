import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read text file without header
df = pd.read_csv("Performance\\TT.txt", sep="\s+", header=None)
df.columns = ["Samples", "ST-SEE", "Autocorrelation_HRM", "Two_stage_framework"]


samples = df["Samples"]
st_see = df["ST-SEE"]
autocorr_hrm = df["Autocorrelation_HRM"]
two_stage = df["Two_stage_framework"]

# Bar chart setup
bar_width = 0.25
x = np.arange(len(samples))

plt.figure(figsize=(12, 6))

# Bars with similar colors to uploaded image
plt.bar(x - bar_width, st_see, width=bar_width, color='#1f77b4', label="ST-SEE")  # Blue
plt.bar(x, autocorr_hrm, width=bar_width, color='#d62728', label="Autocorrelation-based HRM [1]")  # Red
plt.bar(x + bar_width, two_stage, width=bar_width, color='#ff7f0e', label="Two-stage framework [2]")  # Orange

# Styling
plt.xlabel("Samples", fontsize=12, fontweight='bold')
plt.ylabel("Training time (sec)", fontsize=12, fontweight='bold')
plt.title("Performance of Training time", fontsize=14, fontweight='bold')
plt.xticks(x, samples, rotation=45, fontweight='bold')
plt.yticks(fontweight='bold')
plt.legend(fontsize=10, loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("training_time_comparison_styled.png", dpi=300)
plt.show()
