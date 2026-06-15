import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("assets", exist_ok=True)

def run_eda(df: pd.DataFrame):
    df = df.copy()
    numeric_cols = ["engine_temp_C","vibration_ms2","battery_voltage_V",
                    "engine_load_pct","rpm","oil_pressure_psi","coolant_temp_C"]

    # 1. Distribution plots
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle("Sensor Data Distributions by Failure Status", fontsize=14, fontweight="bold")
    for i, col in enumerate(numeric_cols):
        ax = axes[i // 3][i % 3]
        for label, color in [(0, "#2ECC71"), (1, "#E74C3C")]:
            subset = df[df["failure"] == label][col].dropna()
            ax.hist(subset, bins=30, alpha=0.6, color=color,
                    label="Normal" if label == 0 else "Failure", density=True)
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for j in range(len(numeric_cols), 9):
        axes[j // 3][j % 3].set_visible(False)
    plt.tight_layout()
    plt.savefig("assets/eda_distributions.png", dpi=120)
    plt.close()

    # 2. Correlation heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[numeric_cols + ["failure"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/eda_correlation.png", dpi=120)
    plt.close()

    # 3. Failure rate by temp bins
    df["temp_bin"] = pd.cut(df["engine_temp_C"], bins=[60,80,100,120,160],
                            labels=["60-80","80-100","100-120","120-160"])
    fail_rate = df.groupby("temp_bin", observed=True)["failure"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(fail_rate["temp_bin"].astype(str), fail_rate["failure"],
                  color=["#2ECC71","#F39C12","#E67E22","#E74C3C"])
    ax.set_title("Failure Rate by Engine Temperature Range", fontsize=12, fontweight="bold")
    ax.set_xlabel("Temperature Range (°C)")
    ax.set_ylabel("Failure Rate")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, fail_rate["failure"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.0%}", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("assets/eda_temp_failure.png", dpi=120)
    plt.close()

    # 4. Box plots
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle("Sensor Value Ranges: Normal vs Failure", fontsize=13, fontweight="bold")
    for i, col in enumerate(numeric_cols):
        ax = axes[i // 4][i % 4]
        df.boxplot(column=col, by="failure", ax=ax,
                   boxprops=dict(color="#333"),
                   medianprops=dict(color="#E74C3C", linewidth=2))
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("Failure")
        ax.grid(alpha=0.3)
    for j in range(len(numeric_cols), 8):
        axes[j // 4][j % 4].set_visible(False)
    plt.suptitle("Sensor Value Ranges: Normal vs Failure", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/eda_boxplots.png", dpi=120)
    plt.close()

    print("EDA charts saved to assets/")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from preprocess import load_and_clean
    df = load_and_clean("data/vehicle_sensor_data.csv")
    run_eda(df)
