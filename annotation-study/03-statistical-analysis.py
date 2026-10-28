import os
import itertools
import numpy as np
import pandas as pd

from scipy.stats import kruskal, mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests


# file paths
INPUT_FILE = "results/reannotation-per-image-with-domain.csv"
OUT_DIR = "results"

ERROR_COL = "correction_rate"

N_BOOTSTRAPS = 1000
BOOTSTRAP_FRAC = 0.8
RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)


DOMAIN_COLS = [
    "visibility_category",
    "illumination_category",
    "color_category",
    "layout_category",
    "scale_category",
    "background_category",
    "orientation_category",
    "perspective_category",
]


# ordered properties (these are the ones where Spearman makes sense)
ORDINAL_MAPPINGS = {
    "visibility_category": {
        "low": 1,
        "moderate": 2,
        "high": 3,
    },
    "illumination_category": {
        "dark": 1,
        "medium": 2,
        "bright": 3,
    },
    "layout_category": {
        "sparse": 1,
        "moderate": 2,
        "crowded": 3,
    },
    "scale_category": {
        "small": 1,
        "medium": 2,
        "large": 3,
    },
    "orientation_category": {
        "upright": 1,
        "tilted": 2,
        "rotated": 3,
    },
}


# not tested with Spearman because these are not safely monotonic
NON_MONOTONIC_REASON = {
    "color_category": "nominal category; no natural ordering",
    "background_category": "not assumed monotonic; simple/textured/complex may not form a linear difficulty order",
    "perspective_category": "not assumed monotonic; front/oblique/nadir are geometric modes rather than increasing difficulty levels",
}



# helpers
def clean_label(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")


def epsilon_squared_kruskal(h_stat, n, k):
    if pd.isna(h_stat) or n <= k:
        return np.nan

    eps = (h_stat - k + 1) / (n - k)

    # clip negative sampling estimates to zero
    return max(0, eps)


def cliffs_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    nx = len(x)
    ny = len(y)

    if nx == 0 or ny == 0:
        return np.nan

    y_sorted = np.sort(y)

    greater = 0
    less = 0

    for value in x:
        greater += np.searchsorted(y_sorted, value, side="left")
        less += ny - np.searchsorted(y_sorted, value, side="right")

    return (greater - less) / (nx * ny)


def bootstrap_ci(values, lower=2.5, upper=97.5):
    values = np.asarray(values)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    return (
        np.nanmedian(values),
        np.nanpercentile(values, lower),
        np.nanpercentile(values, upper),
    )


# load files

df = pd.read_csv(INPUT_FILE)

df = df.dropna(subset=[ERROR_COL]).copy()

for col in DOMAIN_COLS:
    if col in df.columns:
        df[col] = df[col].apply(clean_label)

print(f"Loaded {len(df)} rows with non-missing {ERROR_COL}.")




# 1. kruskal wallis test

kruskal_rows = []

for col in DOMAIN_COLS:
    if col not in df.columns:
        continue

    sub = df.dropna(subset=[col, ERROR_COL]).copy()
    categories = sorted(sub[col].unique())
    groups = [g.to_numpy() for _, g in sub.groupby(col)[ERROR_COL] if len(g) > 0]
    n, k = len(sub), len(groups)

    if k >= 2 and len(np.unique(sub[ERROR_COL])) > 1:
        h_stat, p_value = kruskal(*groups)
        eps2 = epsilon_squared_kruskal(h_stat, n, k)
    else:
        h_stat, p_value, eps2 = np.nan, np.nan, np.nan

    kruskal_rows.append({
        "domain_property": col,
        "categories": "; ".join(categories),
        "n_images": n,
        "n_categories": k,
        "kruskal_h": h_stat,
        "kruskal_pvalue": p_value,
        "epsilon_squared": eps2,
    })

kruskal_df = pd.DataFrame(kruskal_rows)
kruskal_df.to_csv(os.path.join(OUT_DIR, "kruskal.csv"), index=False)



# 2. spearman rank correlation

spearman_rows = []

for col, mapping in ORDINAL_MAPPINGS.items():
    if col not in df.columns:
        continue

    sub = df.dropna(subset=[col, ERROR_COL]).copy()
    unmapped = sorted(set(sub[col].unique()) - set(mapping))
    sub["domain_rank"] = sub[col].map(mapping)
    sp_data = sub.dropna(subset=["domain_rank", ERROR_COL])

    if (len(sp_data) >= 2
            and sp_data["domain_rank"].nunique() >= 2
            and sp_data[ERROR_COL].nunique() >= 2):
        rho, p_value = spearmanr(sp_data["domain_rank"], sp_data[ERROR_COL])
    else:
        rho, p_value = np.nan, np.nan

    spearman_rows.append({
        "domain_property": col,
        "categories": "; ".join(sorted(sp_data[col].unique())),
        "n_images": len(sp_data),
        "spearman_rho": rho,
        "spearman_pvalue": p_value,
        "unmapped_categories": "; ".join(unmapped),
    })

spearman_df = pd.DataFrame(spearman_rows)
spearman_df.to_csv(os.path.join(OUT_DIR, "spearman.csv"), index=False)


# 3. pairwise mann-whitney u test

pairwise_rows = []

for col in DOMAIN_COLS:

    if col not in df.columns:
        continue

    sub = df.dropna(subset=[col, ERROR_COL]).copy()
    categories = sorted(sub[col].unique())

    raw_pairwise = []

    for a, b in itertools.combinations(categories, 2):

        x = sub.loc[sub[col] == a, ERROR_COL].dropna().values
        y = sub.loc[sub[col] == b, ERROR_COL].dropna().values

        if len(x) == 0 or len(y) == 0:
            continue

        u_stat, raw_p = mannwhitneyu(
            x,
            y,
            alternative="two-sided"
        )

        delta = cliffs_delta(x, y)

        raw_pairwise.append({
            "domain_property": col,
            "category_a": a,
            "category_b": b,
            "n_a": len(x),
            "n_b": len(y),

            "mean_a": np.mean(x),
            "mean_b": np.mean(y),
            "mean_difference_a_minus_b": np.mean(x) - np.mean(y),

            "median_a": np.median(x),
            "median_b": np.median(y),
            "median_difference_a_minus_b": np.median(x) - np.median(y),

            "mannwhitney_u": u_stat,
            "raw_pvalue": raw_p,

            "cliffs_delta_a_vs_b": delta,
        })

    if raw_pairwise:
        adjusted_pvalues = multipletests(
            [r["raw_pvalue"] for r in raw_pairwise],
            method="fdr_bh"
        )[1]

        for r, adj_p in zip(raw_pairwise, adjusted_pvalues):
            r["fdr_pvalue"] = adj_p
            pairwise_rows.append(r)


pairwise_df = pd.DataFrame(pairwise_rows)

pairwise_df.to_csv(
    os.path.join(OUT_DIR, "pairwise-mannwhitney.csv"),
    index=False
)


# 4. bootstrap robustness

bootstrap_overview_rows = []
bootstrap_pairwise_rows = []

for col in DOMAIN_COLS:

    if col not in df.columns:
        continue

    sub = df.dropna(subset=[col, ERROR_COL]).copy()
    categories = sorted(sub[col].unique())

    if len(categories) < 2:
        continue

    boot_h = []
    boot_eps = []
    boot_kw_p = []

    boot_rho = []
    boot_sp_p = []

    pairwise_delta_tracker = {
        (a, b): [] for a, b in itertools.combinations(categories, 2)
    }

    pairwise_mean_diff_tracker = {
        (a, b): [] for a, b in itertools.combinations(categories, 2)
    }

    for i in range(N_BOOTSTRAPS):

        boot = sub.sample(
            frac=BOOTSTRAP_FRAC,
            replace=True,
            random_state=RANDOM_SEED + i
        )

        boot_groups = [
            g[ERROR_COL].dropna().values
            for _, g in boot.groupby(col)
            if len(g[ERROR_COL].dropna()) > 0
        ]

        boot_k = len(boot_groups)
        boot_n = len(boot)

        if boot_k >= 2 and len(np.unique(boot[ERROR_COL])) > 1:
            h, p = kruskal(*boot_groups)
            eps = epsilon_squared_kruskal(h, boot_n, boot_k)

            boot_h.append(h)
            boot_kw_p.append(p)
            boot_eps.append(eps)

        # spearman only for ordered properties
        if col in ORDINAL_MAPPINGS:
            mapping = ORDINAL_MAPPINGS[col]
            boot = boot.copy()
            boot["domain_rank"] = boot[col].map(mapping)

            sp_data = boot.dropna(subset=["domain_rank", ERROR_COL])

            if (len(sp_data) >= 2 and sp_data["domain_rank"].nunique() >= 2
                    and sp_data[ERROR_COL].nunique() >= 2):
                rho, sp_p = spearmanr(
                    sp_data["domain_rank"],
                    sp_data[ERROR_COL]
                )

                boot_rho.append(rho)
                boot_sp_p.append(sp_p)

        # pairwise bootstrap
        for a, b in itertools.combinations(categories, 2):

            x = boot.loc[boot[col] == a, ERROR_COL].dropna().values
            y = boot.loc[boot[col] == b, ERROR_COL].dropna().values

            if len(x) == 0 or len(y) == 0:
                continue

            delta = cliffs_delta(x, y)
            mean_diff = np.mean(x) - np.mean(y)

            pairwise_delta_tracker[(a, b)].append(delta)
            pairwise_mean_diff_tracker[(a, b)].append(mean_diff)

    h_med, h_low, h_high = bootstrap_ci(boot_h)
    eps_med, eps_low, eps_high = bootstrap_ci(boot_eps)
    rho_med, rho_low, rho_high = bootstrap_ci(boot_rho)

    kw_sig_rate = np.mean(np.asarray(boot_kw_p) < 0.05) if len(boot_kw_p) > 0 else np.nan
    sp_sig_rate = np.mean(np.asarray(boot_sp_p) < 0.05) if len(boot_sp_p) > 0 else np.nan

    bootstrap_overview_rows.append({
        "domain_property": col,
        "n_bootstraps": N_BOOTSTRAPS,
        "bootstrap_fraction": BOOTSTRAP_FRAC,

        "kruskal_h_median": h_med,
        "kruskal_h_ci_low": h_low,
        "kruskal_h_ci_high": h_high,

        "epsilon_squared_median": eps_med,
        "epsilon_squared_ci_low": eps_low,
        "epsilon_squared_ci_high": eps_high,

        "kruskal_significant_fraction_p_lt_0_05": kw_sig_rate,

        "spearman_used": col in ORDINAL_MAPPINGS,
        "spearman_rho_median": rho_med,
        "spearman_rho_ci_low": rho_low,
        "spearman_rho_ci_high": rho_high,
        "spearman_significant_fraction_p_lt_0_05": sp_sig_rate,
    })

    for (a, b), deltas in pairwise_delta_tracker.items():

        d_med, d_low, d_high = bootstrap_ci(deltas)

        md_med, md_low, md_high = bootstrap_ci(
            pairwise_mean_diff_tracker[(a, b)]
        )

        # direction stability (how often does category_a have higher correction rate than category_b)
        mean_diffs = np.asarray(pairwise_mean_diff_tracker[(a, b)])
        direction_fraction_a_gt_b = np.mean(mean_diffs > 0) if len(mean_diffs) else np.nan

        bootstrap_pairwise_rows.append({
            "domain_property": col,
            "category_a": a,
            "category_b": b,

            "cliffs_delta_median": d_med,
            "cliffs_delta_ci_low": d_low,
            "cliffs_delta_ci_high": d_high,

            "mean_difference_a_minus_b_median": md_med,
            "mean_difference_a_minus_b_ci_low": md_low,
            "mean_difference_a_minus_b_ci_high": md_high,

            "direction_fraction_a_higher_than_b": direction_fraction_a_gt_b,
        })


bootstrap_overview_df = pd.DataFrame(bootstrap_overview_rows)
bootstrap_pairwise_df = pd.DataFrame(bootstrap_pairwise_rows)

bootstrap_overview_df.to_csv(
    os.path.join(OUT_DIR, "bootstrap-kruskal-spearman.csv"),
    index=False
)

bootstrap_pairwise_df.to_csv(
    os.path.join(OUT_DIR, "bootstrap-mannwhitney.csv"),
    index=False
)



# summary
print()
print("Saved results to:", OUT_DIR)
print()
print("Main files:")
print(" - kruskal.csv")
print(" - spearman.csv")
print(" - pairwise-mannwhitney.csv")
print(" - bootstrap-kruskal-spearman.csv")
print(" - bootstrap-mannwhitney.csv")