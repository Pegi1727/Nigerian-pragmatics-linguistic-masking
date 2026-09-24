#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_and_visualizations.py
Replication package: "You Look Fresh Today!" — Nigerian English Pragmatics Survey (N = 100)

This script:
  1) Loads pragmatics_reconstructed_item_level.csv
  2) Produces descriptive frequency tables
  3) Produces 4 figures (PNG, 300 dpi)
  4) Produces a cross-tabulation workbook (xlsx) and a summary CSV

Usage:
    pip install -r requirements.txt
    python analysis_and_visualizations.py

Inputs  : pragmatics_reconstructed_item_level.csv (same directory, or edit DATA_PATH)
Outputs : Figure_1_General_Use_and_Intent.png
          Figure_2_Affective_Material_Gap.png
          Figure_3_Hunger_Reactions_Q4.png
          Figure_4_Response_Strategies_Comparison.png
          pragmatics_crosstabs_analysis.xlsx
          pragmatics_descriptive_summary.csv
"""

import os
import sys
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DATA_PATH = os.environ.get("DATA_PATH", "pragmatics_reconstructed_item_level.csv")
OUT_DIR = os.environ.get("OUT_DIR", ".")
FIG_DPI = 300

BLUE = "#1f4e79"
ORANGE = "#e07b39"
GREEN = "#3a7d44"
RED = "#b23a48"
GREY = "#8c8c8c"

plt.rcParams.update({
    "figure.figsize": (9, 5.5),
    "figure.dpi": 100,
    "savefig.dpi": FIG_DPI,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fail(msg):
    sys.exit(f"[ERROR] {msg}")


def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        fail(f"Dataset not found at '{path}'. Place the CSV next to this script "
             "or set the DATA_PATH environment variable.")
    df = pd.read_csv(path)
    expected = ["ID", "Q1_frequency", "Q2_intent", "Q3_affection", "Q4_reaction",
                "Q5", "Q6_economy", "Age", "Location", "Ethnicity"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        fail(f"Missing expected columns: {missing}")
    print(f"[OK] Loaded {len(df)} respondents, {df.shape[1]} variables from {path}")
    return df


def value_table(series, label):
    vc = series.value_counts(dropna=False)
    tbl = pd.DataFrame({label: vc.index, "n": vc.values})
    tbl["%"] = (tbl["n"] / len(series) * 100).round(1)
    return tbl


def bar_plot(tbl, label_col, title, fname, color=BLUE, xlabel="Respondents (n)"):
    """Vertical bar chart of a categorical frequency table."""
    fig, ax = plt.subplots()
    x = np.arange(len(tbl))
    ax.bar(x, tbl["n"], color=color, edgecolor="white", width=0.65)
    for xi, (n, pct) in enumerate(zip(tbl["n"], tbl["%"])):
        ax.text(xi, n + max(tbl["n"]) * 0.02, f"{n}\n({pct}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(textwrap.fill(str(v), 16) for v in tbl[label_col])
    ax.set_ylabel(xlabel)
    ax.set_title(title, fontweight="bold", pad=14)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, max(tbl["n"]) * 1.22)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path)
    plt.close(fig)
    print(f"[OK] Saved {path}")


def grouped_plot(cross, title, fname, colors, xlabel):
    """Grouped bar chart from a crosstab DataFrame (rows=categories, cols=groups)."""
    fig, ax = plt.subplots()
    n_groups = cross.shape[0]
    n_series = cross.shape[1]
    x = np.arange(n_groups)
    width = 0.8 / n_series
    for j, col in enumerate(cross.columns):
        vals = cross[col].values
        ax.bar(x + (j - (n_series - 1) / 2) * width, vals, width,
               label=str(col), color=colors[j % len(colors)], edgecolor="white")
        for xi, v in zip(x + (j - (n_series - 1) / 2) * width, vals):
            if v > 0:
                ax.text(xi, v + cross.values.max() * 0.015, int(v),
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(textwrap.fill(str(v), 18) for v in cross.index)
    ax.set_ylabel("Respondents (n)")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold", pad=14)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path)
    plt.close(fig)
    print(f"[OK] Saved {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()
    n = len(df)

    # ------------------------------------------------------------------
    # Descriptive tables
    # ------------------------------------------------------------------
    tables = {
        "Q1_frequency": value_table(df["Q1_frequency"], "Q1_frequency"),
        "Q2_intent": value_table(df["Q2_intent"], "Q2_intent"),
        "Q3_affection": value_table(df["Q3_affection"], "Q3_affection"),
        "Q4_reaction": value_table(df["Q4_reaction"], "Q4_reaction"),
        "Q5": value_table(df["Q5"], "Q5"),
        "Q6_economy": value_table(df["Q6_economy"], "Q6_economy"),
        "Location": value_table(df["Location"], "Location"),
    }
    print("\n=== Descriptive frequencies (N = %d) ===" % n)
    for name, t in tables.items():
        print(f"\n{name}\n{t.to_string(index=False)}")

    summary_path = os.path.join(OUT_DIR, "pragmatics_descriptive_summary.csv")
    with open(summary_path, "w", encoding="utf-8") as fh:
        for name, t in tables.items():
            fh.write(f"# {name}\n")
            t.to_csv(fh, index=False)
            fh.write("\n")
    print(f"[OK] Saved {summary_path}")

    # ------------------------------------------------------------------
    # Figure 1 — General use and perceived intent (Q1 + Q2)
    # ------------------------------------------------------------------
    bar_plot(tables["Q1_frequency"], "Q1_frequency",
             "Figure 1a. How often respondents hear/give 'You look fresh today!'",
             "Figure_1a_Q1_Frequency.png", color=BLUE)
    bar_plot(tables["Q2_intent"], "Q2_intent",
             "Figure 1b. Perceived intent behind the greeting (Q2)",
             "Figure_1b_Q2_Intent.png", color=BLUE)

    # Composite Figure 1 (as used in the report)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, color in zip(axes, ["Q1_frequency", "Q2_intent"], [BLUE, ORANGE]):
        t = tables[key]
        ax.bar(np.arange(len(t)), t["n"], color=color, edgecolor="white")
        for xi, (nv, pv) in enumerate(zip(t["n"], t["%"])):
            ax.text(xi, nv + max(t["n"]) * 0.02, f"{nv}\n({pv}%)", ha="center", fontsize=9)
        ax.set_xticks(np.arange(len(t)))
        ax.set_xticklabels(textwrap.fill(str(v), 14) for v in t[t.columns[0]], fontsize=9)
        ax.set_ylabel("Respondents (n)")
        ax.set_ylim(0, max(t["n"]) * 1.25)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_title("Q1. Frequency of use/hearing", fontweight="bold")
    axes[1].set_title("Q2. Perceived intent", fontweight="bold")
    fig.suptitle("Figure 1. General use and perceived intent of the compliment",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "Figure_1_General_Use_and_Intent.png")
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {p1}")

    # ------------------------------------------------------------------
    # Figure 2 — Affective vs. material reading (Q3 vs Q6) — the "gap"
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    t3 = tables["Q3_affection"]
    axes[0].bar(np.arange(len(t3)), t3["n"], color=GREEN, edgecolor="white")
    for xi, (nv, pv) in enumerate(zip(t3["n"], t3["%"])):
        axes[0].text(xi, nv + 1, f"{nv} ({pv}%)", ha="center", fontsize=10)
    axes[0].set_xticks(np.arange(len(t3)))
    axes[0].set_xticklabels([str(v) for v in t3["Q3_affection"]], fontsize=9)
    axes[0].set_ylabel("Respondents (n)")
    axes[0].set_ylim(0, 115)
    axes[0].set_title("Q3. Compliment signals affection\n(universal agreement)", fontweight="bold")

    t6 = tables["Q6_economy"]
    axes[1].bar(np.arange(len(t6)), t6["n"], color=RED, edgecolor="white")
    for xi, (nv, pv) in enumerate(zip(t6["n"], t6["%"])):
        axes[1].text(xi, nv + 1.2, f"{nv} ({pv}%)", ha="center", fontsize=10)
    axes[1].set_xticks(np.arange(len(t6)))
    axes[1].set_xticklabels(textwrap.fill(str(v), 18) for v in t6["Q6_economy"], fontsize=9)
    axes[1].set_ylim(0, 85)
    axes[1].set_title("Q6. 'Fresh' also a signal of\neconomic caution?", fontweight="bold")
    fig.suptitle("Figure 2. Affective agreement vs. economic-material gap",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "Figure_2_Affective_Material_Gap.png")
   ] Saved {p2}")

    # ------------------------------------------------------------------
    plt.close(fig)
    print(f"[OK] Saved {p2}")

    # ------------------------------------------------------------------
    # Figure 3 — Reactions to being told to 'eat well' (Q4), by Location
    # ------------------------------------------------------------------
    cross_q4loc = pd.crosstab(df["Q4_reaction"], df["Location"])
    print("\n=== Q4 reaction x Location ===\n", cross_q4loc)
    grouped_plot(cross_q4loc,
                 "Figure 3. Reactions to 'You must be eating well!' (Q4), by Location",
                 "Figure_3_Hunger_Reactions_Q4.png",
                 colors=[BLUE, ORANGE], xlabel="Reaction type")

    # ------------------------------------------------------------------
    # Figure 4 — Response strategies (Q5), by Location
    # ------------------------------------------------------------------
    cross_q5loc = pd.crosstab(df["Q5"], df["Location"])
    print("\n=== Q5 strategy x Location ===\n", cross_q5loc)
    grouped_plot(cross_q5loc,
                 "Figure 4. Response strategies to the compliment (Q5), by Location",
                 "Figure_4_Response_Strategies_Comparison.png",
                 colors=[BLUE, GREEN], xlabel="Response strategy")

    # ------------------------------------------------------------------
    # Cross-tab workbook
    # ------------------------------------------------------------------
    xlsx_path = os.path.join(OUT_DIR, "pragmatics_crosstabs_analysis.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        for name, t in tables.items():
            t.to_excel(xw, sheet_name=name[:31], index=False)
        cross_q4loc.to_excel(xw, sheet_name="Q4_by_Location")
        cross_q5loc.to_excel(xw, sheet_name="Q5_by_Location")
        pd.crosstab(df["Q2_intent"], df["Location"]).to_excel(xw, sheet_name="Q2_by_Location")
        pd.crosstab(df["Q6_economy"], df["Location"]).to_excel(xw, sheet_name="Q6_by_Location")
    print(f"[OK] Saved {xlsx_path}")

    print("\n[DONE] All outputs written to:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
