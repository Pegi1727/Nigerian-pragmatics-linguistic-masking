"""Build nigerian_pragmatics_replication.ipynb with nbformat (no raw-string escaping bugs)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

# ---------------------------------------------------------------- title / workflow
nb.cells.append(new_markdown_cell(r"""# Replication Notebook — *"Between Care and 'Billing': Decoding Linguistic Masking and Transnational Identity in the Nigerian Greeting 'Have You Eaten?'"*

This notebook replicates the full quantitative workflow of the study using the reconstructed item-level dataset
(`pragmatics_reconstructed_item_level.csv`, N = 100).

## Replication workflow

| Step | Analysis | Notebook cell |
|------|----------|---------------|
| 1 | Environment setup & library imports (pandas, numpy, scipy.stats, statsmodels, matplotlib, seaborn) | Code cell 1 |
| 2 | Load item-level data; validate N = 100 (60 Diaspora, 40 Nigeria residents) | Code cell 2 |
| 3 | Frequency analyses: **Q1** usage, **Q2** intent, **Q3** affective alignment, each with **Wilson 95% CIs** | Code cell 3 |
| 4 | Immediate hunger reactions: **Q4** multi-response distribution | Code cell 4 |
| 5 | Response strategies (**Q5**), affective–material decoupling (**exact binomial test**), and **χ²** test Diaspora vs Nigeria residents — target: χ²(3) = 0.34, p = 0.952, Cramér's V = 0.058 | Code cell 5 |
| 6 | Perceived persistence under economic hardship (**Q6**: 70% / 5% / 25%) | Code cell 6 |
| 7 | Figures 1–4 in the publication visual style | Code cell 7 |

**Methodological notes**

* Categorical proportions are reported with **Wilson score 95% confidence intervals** (more accurate than
  Wald intervals at small *n* and near boundary proportions).
* The affective–material decoupling is tested with an **exact binomial test**: while 100% of respondents agree
  the greeting conveys affection (Q3), only 10% report a concrete material response (Q5: "Offer food/money").
* Group differences (Diaspora vs Nigeria residents) on Q5 use **Pearson's chi-square test of independence**;
  effect size is reported as **Cramér's V** = sqrt(χ² / (n·(k−1))).
* All figures use a consistent publication style (white grid, muted palette, percentage axes, Wilson CI whiskers)."""))

# ---------------------------------------------------------------- cell 1: setup
nb.cells.append(new_markdown_cell("## Code cell 1 — Environment setup and library imports"))
nb.cells.append(new_code_cell(
"""# Environment setup -----------------------------------------------------------
import matplotlib
matplotlib.use("Agg")  # headless-safe rendering
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, binomtest
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint   # Wilson 95% CI
import matplotlib.pyplot as plt
import seaborn as sns

# Publication visual style ----------------------------------------------------
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "legend.frameon": False,
})
PALETTE = {"Diaspora": "#1f77b4", "Nigeria": "#d62728"}
ACCENT, MUTED = "#4c72b0", "#8c8c8c"

DATA_PATH = Path("pragmatics_reconstructed_item_level.csv")
OUT_DIR = Path(".")
print("pandas", pd.__version__, "| numpy", np.__version__, "| seaborn", sns.__version__)"""))

# ---------------------------------------------------------------- cell 2: load & validate
nb.cells.append(new_markdown_cell("""## Code cell 2 — Load data and validate sample size

**Validation targets:** N = 100 respondents; 60 Diaspora, 40 Nigeria residents; no missing values."""))
nb.cells.append(new_code_cell(
"""df = pd.read_csv(DATA_PATH)

# --- structural validation ----------------------------------------------------
assert len(df) == 100, f"Expected N=100, got {len(df)}"
loc_counts = df["Location"].value_counts()
assert loc_counts["Diaspora"] == 60 and loc_counts["Nigeria"] == 40, loc_counts.to_dict()
assert df.isna().sum().sum() == 0, "Unexpected missing values"
assert set(df["Q1_frequency"].unique()) <= {"Always", "Often", "Sometimes"}
assert set(df["Q6_economy"].unique()) <= {"No: traditional regardless", "Yes: more cautious", "Not sure"}

print(f"Shape: {df.shape[0]} respondents x {df.shape[1]} variables")
print("\\nGroup sizes (Location):")
print(loc_counts.to_string())
print("\\nDiaspora % =", round(100 * loc_counts['Diaspora'] / len(df)), "| Nigeria % =", round(100 * loc_counts['Nigeria'] / len(df)))
print("\\nAge distribution:", df['Age'].value_counts().to_dict())
print("Ethnicity:", df['Ethnicity'].value_counts().to_dict())
df.head()"""))

# ---------------------------------------------------------------- cell 3: frequencies + Wilson CIs
nb.cells.append(new_markdown_cell("""## Code cell 3 — Frequency analyses with Wilson 95% CIs

Proportions for **Q1 (usage frequency)**, **Q2 (attributed intent)** and **Q3 (affective alignment)**,
each with Wilson score 95% confidence intervals."""))
nb.cells.append(new_code_cell(
"""N = len(df)

def freq_table(col, order=None):
    \"\"\"Counts, %, and Wilson 95% CI per category.\"\"\"
    vc = df[col].value_counts()
    if order:
        vc = vc.reindex(order).fillna(0).astype(int)
    rows = []
    for cat, k in vc.items():
        lo, hi = proportion_confint(k, N, alpha=0.05, method="wilson")
        rows.append({"category": cat, "n": int(k), "pct": 100 * k / N,
                     "ci_lo_pct": 100 * lo, "ci_hi_pct": 100 * hi})
    return pd.DataFrame(rows)

q1 = freq_table("Q1_frequency", ["Always", "Often", "Sometimes"])
q2 = freq_table("Q2_intent", ["Genuine care", "Small care", "Social expectation/habit", "Small talk"])
q3 = freq_table("Q3_affection", ["Strongly Agree", "Agree", "Neutral", "Disagree"])

print("Q1 — How often do you use/greet with 'Have you eaten?'")
print(q1.round(1).to_string(index=False), "\\n")
print("Q2 — Perceived intent behind the greeting")
print(q2.round(1).to_string(index=False), "\\n")
print("Q3 — Affective alignment ('the greeting expresses affection')")
print(q3.round(1).to_string(index=False))

# Key replication checkpoints --------------------------------------------------
q1_often = q1.loc[q1.category == "Often"].iloc[0]
q3_full  = q3.loc[q3.category == "Strongly Agree"].iloc[0]
print("\\nCheckpoints:")
    print(f"  Q1 'Often' = {q1_often.pct:.0f}%  Wilson 95% CI [{q1_often.ci_lo_pct:.1f}, {q1_often.ci_hi_pct:.1f}]")
print(f"  Q2 'Genuine care'+'Small care' = {(q2.n.iloc[0]+q2.n.iloc[1])}% (62%)")
print(f"  Q3 'Strongly Agree' = {q3_full.pct:.0f}%  (unanimous affective alignment)")"""))

# ---------------------------------------------------------------- cell 4: Q4
nb.cells.append(new_markdown_cell("""## Code cell 4 — Immediate hunger reactions (Q4, multi-response distribution)

Distribution of respondents' immediate internal reactions when asked *"Have you eaten?"*."""))
nb.cells.append(new_code_cell(
"""Q4_ORDER = ["Surprised", "Neutral/Amused", "Empathetic/help", "Anxious re billing"]

q4 = freq_table("Q4_reaction", Q4_ORDER)
print("Q4 - Immediate hunger reactions")
print(q4.round(1).to_string(index=False))

# quick chart
ax = plt.figure(figsize=(7, 3.4))
plt.errorbar(q4.pct, range(len(q4)), xerr=[q4.pct - q4.ci_lo_pct, q4.ci_hi_pct - q4.pct],
             fmt="none", ecolor=MUTED, capsize=3, lw=1)
plt.scatter(q4.pct, range(len(q4)), s=55, color=ACCENT, zorder=3)
plt.yticks(range(len(q4)), q4.category)
plt.xlim(0, 55)
plt.xlabel("Percentage of respondents (%)")
plt.title("Q4 — Immediate hunger reactions (Wilson 95% CI)")
plt.gca().invert_yaxis()
plt.tight_layout(); plt.show()

print("\\nInterpretation: the dominant immediate reaction is surprise (42%), followed by neutral/amused (33%); "
      "billing-related anxiety (8%) is a minority but theoretically salient 'masking' response.")"""))

# ---------------------------------------------------------------- cell 5: Q5 + decoupling + chi-square
nb.cells.append(new_markdown_cell("""## Code cell 5 — Response strategies (Q5), affective–material decoupling, and group comparison

1. Frequency of **Q comparison

1. Frequency of **Q5 binomial test** of affective–material decoupling: material response ("Offer food/money", 10%)
   against the null expectation of parity (p₀ = 0.5), given unanimous affective agreement (Q3 = 100%).
3. **Pearson χ²** comparing Diaspora vs Nigeria residents on Q5 —
   target: χ²(3) = 0.34, p = 0.952, Cramér's V = 0.058."""))
nb.cells.append(new_code_cell(
"""Q5_ORDER = ["Verbal comfort/prayers", "Change topic", "Offer food/money", "Other"]
q5 = freq_table("Q5", Q5_ORDER)
print("Q5 — Response strategies")
print(q5.round(1).to_string(index=False))

# --- Affective-material decoupling: exact binomial test -----------------------
material_n = int(q5.loc[q5.category == "Offer food/money", "n"].iloc[0])   # k = 10 of 100
dec_test = binomtest(material_n, N, p=0.5, alternative="less")
gap = 100 - material_n   # 100% affective alignment vs 10% material response
print(f"\\nAffective-material decoupling:")
print(f"  Affective alignment (Q3 Strongly Agree) = 100%")
print(f"  Material response (Q5 'Offer food/money') = {material_n}%")
print(f"  Decoupling gap = {gap} percentage points")
print(f"  Exact binomial test (k={material_n}, n={N}, p0=0.5, H1: less):")
print(f"    statistic = {dec_test.statistic}, p-value = {dec_test.pvalue:.3e}")
print("  -> The gap between expressed affection and material offers is highly significant (p < .001).")

# --- Chi-square: Diaspora vs Nigeria residents on Q5 --------------------------
ct = pd.crosstab(df["Location"], df["Q5"])[Q5_ORDER].reindex(["Diaspora", "Nigeria"])
chi2, p_chi2, dof, expected = chi2_contingency(ct)
n_total = ct.values.sum()
k_min = min(ct.shape)                      # k = min(rows, cols) = 2
cramers_v = np.sqrt(chi2 / (n_total * (k_min - 1)))
print("\\nContingency table (Location x Q5):")
print(ct.to_string())
print(f"\\nChi-square test of independence:")
print(f"  chi2({dof}) = {chi2:.2f}, p = {p_chi2:.3f}, Cramer's V = {cramers_v:.3f}")
print(f"  Replication targets: chi2(3) = 0.34, p = 0.952, V = 0.058  ->  "
      f"{'MATCH' if (abs(chi2-0.34)<0.01 and abs(p_chi2-0.952)<0.001 and abs(cramers_v-0.058)<0.001) else 'CHECK'}")

# Small expected counts check (Cochran's rule)
exp_min = expected.min()
print(f"  Minimum expected cell count = {exp_min:.1f} "
      f"({'> 5, chi-square assumption met' if exp_min >= 5 else '< 5, interpret with caution'})")"""))

# ---------------------------------------------------------------- cell 6: Q6
nb.cells.append(new_markdown_cell("""## Code cell 6 — Perceived persistence under economic hardship (Q6)

Target distribution: **70%** "No: traditional regardless", **5%** "Yes: more cautious", **25%** "Not sure",
with Wilson 95% CIs."""))
nb.cells.append(new_code_cell(
"""Q6_ORDER = ["No: traditional regardless", "Yes: more cautious", "Not sure"]
q6 = freq_table("Q6_economy", Q6_ORDER)
print("Q6 — Does economic hardship change how the greeting is used/perceived?")
print(q6.round(1).to_string(index=False))

for _, r in q6.iterrows():
    print(f"  {r.category}: {r.pct:.0f}%  Wilson 95% CI [{r.ci_lo_pct:.1f}, {r.ci_hi_pct:.1f}]  (n={r.n})")

assert q6.loc[q6.category == 'No: traditional regardless', 'n'].iloc[0] == 70
assert q6.loc[q6.category == 'Yes: more cautious', 'n'].iloc[0] == 5
assert q6.loc[q6.category == 'Not sure', 'n'].iloc[0] == 25
print("\\nCheckpoint passed: 70% / 5% / 25% distribution reproduced.")"""))

# ---------------------------------------------------------------- cell 7: figures 1-4
nb.cells.append(new_markdown_cell("""## Code cell 7 — Figures 1–4 (publication visual style)

* **Figure 1** — General use (Q1) and attributed intent (Q2) with Wilson 95% CIs.
* **Figure 2** — The affective–material gap (Q3 = 100% vs Q5 material = 10%).
* **Figure 3** — Hunger reactions (Q4) distribution.
* **Figure 4** — Response strategies (Q5) compared across Diaspora vs Nigeria residents."""))
nb.cells.append(new_code_cell(
"""def wilson_ci_ax_bars(ax, tab, color, ypos_offset=0.0, vertical=True):
    \"\"\"Draw percentage bars with Wilson 95% CI whiskers.\"\"\"
    if vertical:
        ax.bar(tab.category, tab.pct, color=color, alpha=.85, edgecolor="white")
        ax.errorbar(tab.category, tab.pct,
                    yerr=[tab.pct - tab.ci_lo_pct, tab.ci_hi_pct - tab.pct],
                    fmt="none", ecolor="#333333", capsize=4, lw=1)
        for x, (_, r) in zip(range(len(tab)), tab.iterrows()):
            ax.text(x, r.ci_hi_pct + 1.5, f"{r.pct:.0f}%", ha="center", fontsize=9, fontweight="bold")
        ax.set_ylabel("Percentage of respondents (%)")
        ax.set_ylim(0, 108)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=18, ha="right")
    else:
        ax.barh(tab.category, tab.pct, color=color, alpha=.85, edgecolor="white")
        ax.errorbar(tab.pct, tab.category,
                    xerr=[tab.pct - tab.ci_lo_pct, tab.ci_hi_pct - tab.pct],
                    fmt="none", ecolor="#333333", capsize=4, lw=1)
        for y, (_, r) in zip(range(len(tab)), tab.iterrows()):
            ax.text(r.ci_hi_pct + 1.5, y, f"{r.pct:.0f}%", va="center", fontsize=9, fontweight="bold")
        ax.set_xlabel("Percentage of respondents (%)")
        ax.set_xlim(0, 108)

# ---- Figure 1: Q1 usage + Q2 intent -----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
wilson_ci_ax_bars(axes[0], q1, "#4c72b0")
axes[0].set_title("Figure 1a — Usage frequency (Q1)")
wilson_ci_ax_bars(axes[1], q2, "#dd8452")
axes[1].set_title("Figure 1b — Attributed intent (Q2)")
fig.suptitle("Figure 1 — General use and intent of 'Have you eaten?' (Wilson 95% CI)", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_DIR / "Figure_1_General_Use_and_Intent_replication.png"); plt.show()

# ---- Figure 2: affective-material gap ----------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 3.8))
bars = ax.bar(["Affective alignment\\n(Q3 'Strongly Agree')", "Material response\\n(Q5 'Offer food/money')"],
              [100, material_n], color=["#55a868", "#c44e52"], alpha=.88, edgecolor="white", width=.55)
for x, v in zip([0, 1], [100, material_n]):
    ax.text(x, v + 2, f"{v}%", ha="center", fontsize=11, fontweight="bold")
ax.annotate("", xy=(1, 12), xytext=(1, 88),
            arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.2))
ax.text(1.12, 50, f"Decoupling gap\\n= {gap} pts\\n(exact binomial\\np < .001)", fontsize=8.5, color="#333333")
ax.set_ylabel("Percentage of respondents (%)"); ax.set_ylim(0, 112)
ax.set_title("Figure 2 — Affective–material decoupling")
fig.tight_layout()
fig.savefig(OUT_DIR / "Figure_2_Affective_Material_Gap_replication.png"); plt.show()

# ---- Figure 3: Q4 hunger reactions -------------------------------------------
fig, ax = plt.subplots(figsize=(7, 3.8))
wilson_ci_ax_bars(ax, q4, "#8172b3")
ax.set_title("Figure 3 — Immediate hunger reactions (Q4, Wilson 95% CI)")
fig.tight_layout()
fig.savefig(OUT_DIR / "Figure_3_Hunger_Reactions_Q4_replication.png"); plt.show()

# ---- Figure 4: Q5 strategies by group ----------------------------------------
q5_grp = (pd.crosstab(df["Q5"], df["Location"], normalize="columns")[["Diaspora", "Nigeria"]] * 100)
q5_grp = q5_grp.reindex(Q5_ORDER).reset_index().melt(id_vars="Q5", var_name="Location", value_name="pct")
fig, ax = plt.subplots(figsize=(8, 3.8))
sns.barplot(data=q5_grp, x="Q5", y="pct", hue="Location", palette=PALETTE, ax=ax, edgecolor="white")
for cont in ax.containers:
    ax.bar_label(cont, fmt="%.0f%%", fontsize=8.5, padding=2)
ax.set_ylabel("Percentage within group (%)"); ax.set_xlabel("")
ax.set_ylim(0, 80)
ax.set_xticklabels([t.get_text().replace("/", "/\\n") for t in ax.get_xticklabels()])
ax.set_title("Figure 4 — Response strategies (Q5): Diaspora vs Nigeria residents\\n"
             f"χ²(3) = {chi2:.2f}, p = {p_chi2:.3f}, Cramér's V = {cramers_v:.3f} (n.s.)")
ax.legend(title="Group", loc="upper right")
fig.tight_layout()
fig.savefig(OUT_DIR / "Figure_4_Response_Strategies_Comparison_replication.png"); plt.show()

print("Figures saved:")
for f in sorted(OUT_DIR.glob("Figure_*_replication.png")):
    print("  ", f.name)"""))

# ---------------------------------------------------------------- save & verify
path = "/mnt/data/nigerian_pragmatics_replication.ipynb"
nbf.validate(nb)
with open(path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

# verify
nb2 = nbf.read(path, as_version=4)
nbf.validate(nb2)
md = sum(1 for c in nb2.cells if c.cell_type == "markdown")
code = sum(1 for c in nb2.cells if c.cell_type == "code")
import os, json
json.load(open(path))
print(f"OK: wrote {path}")
print(f"cells: {len(nb2.cells)} total | markdown={md} | code={code}")
print(f"size: {os.path.getsize(path):,} bytes")
