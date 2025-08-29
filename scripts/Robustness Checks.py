import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy.stats import norm

# ========= File path =========
PATH = r"Robustness checks.xlsx"
SHEET = 0

# ========= Two-sided Tobit (censored at 0 and 1) =========
class TobitTwoSided(GenericLikelihoodModel):
    def __init__(self, endog, exog, left=0.0, right=1.0, **kwds):
        self.left = left
        self.right = right
        super().__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        beta, sigma = params[:-1], params[-1]
        if sigma <= 0:
            return np.inf * np.ones_like(self.endog)

        xb = np.dot(self.exog, beta)
        y = self.endog

        z_left  = (self.left  - xb) / sigma
        z_right = (self.right - xb) / sigma

        uncensored = (y > self.left + 1e-12) & (y < self.right - 1e-12)
        left_c  = (y <= self.left + 1e-12)
        right_c = (y >= self.right - 1e-12)

        ll = np.zeros_like(y, dtype=float)
        if np.any(uncensored):
            z = (y[uncensored] - xb[uncensored]) / sigma
            ll[uncensored] = np.log(norm.pdf(z)) - np.log(sigma)
        if np.any(left_c):
            ll[left_c]  = np.log(norm.cdf(z_left[left_c]))
        if np.any(right_c):
            ll[right_c] = np.log(1 - norm.cdf(z_right[right_c]))
        return -ll

    def fit(self, start_params=None, maxiter=2000, maxfun=2000, disp=0, **kwds):
        if start_params is None:
            ols_res = sm.OLS(self.endog, self.exog).fit()
            sigma0 = np.sqrt(np.mean(ols_res.resid**2))
            start_params = np.r_[ols_res.params, sigma0]
        return super().fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, disp=disp, **kwds)

def stars(p): 
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))

if __name__ == "__main__":
    df = pd.read_excel(PATH, sheet_name=SHEET)
    need = {"SE","GF","ATR","LDR","RPC","OPM"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"missing columns：{miss}")

    # ========= GF with controls =========
    y = df["SE"].to_numpy()
    Xdf = df[["GF","ATR","LDR","RPC","OPM"]]
    X = sm.add_constant(Xdf, has_constant="add").to_numpy()

    mod2 = TobitTwoSided(y, X, left=0.0, right=1.0)
    res2 = mod2.fit(disp=0)

    index2 = ["const","GF","ATR","LDR","RPC","OPM","sigma"]
    params2 = pd.Series(res2.params, index=index2)
    bse2    = pd.Series(res2.bse,    index=index2)
    zvals2  = params2 / bse2
    pvals2  = pd.Series(res2.pvalues, index=index2)
    tbl2 = pd.DataFrame({
        "coef": params2, 
        "std_err": bse2, 
        "z": zvals2, 
        "p_value": pvals2,
        "stars": [stars(p) for p in pvals2]
    })
    print("\n=== GF with controls ===")
    print(tbl2.round(4).to_string())