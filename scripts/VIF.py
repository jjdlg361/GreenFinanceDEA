import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

# ========= File path =========
PATH = r"Tobit regression.xlsx"
SHEET = 0  

if __name__ == "__main__":
    df = pd.read_excel(PATH, sheet_name=SHEET)
    cols = ["GF","ATR","LDR","RPC","OPM"]
    miss = set(cols) - set(df.columns)
    if miss:
        raise ValueError(f"missing columns：{miss}")

    Xdf = df[cols].copy()

    # ========= VIF =========
    Xv = sm.add_constant(Xdf, has_constant="add")
    vif_vals = [variance_inflation_factor(Xv.values, i) for i in range(1, Xv.shape[1])]  # 排除常数项
    vif_tbl = pd.DataFrame({"variable": Xv.columns[1:], "VIF": vif_vals})
    print("\n=== VIF analysis ===")
    print(vif_tbl.round(3).to_string())

    # ========= Correlation heatmap =========
    corr = Xdf.corr().values
    n = len(cols)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="viridis")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(cols); ax.set_yticklabels(cols)

    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle=":", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color=("white" if abs(val) > 0.6 else "black"),
                    fontsize=9)

    plt.tight_layout()
    plt.show()
