import pandas as pd
import os

# ---------------- CONFIG ----------------

INPUT_FILE = "results/reannotation-per-image-with-domain.csv"
OUTPUT_FILE = "results/correction-rates-per-domain.csv"


DOMAIN_COLS = [
    "visibility_category",
    "illumination_category",
    "color_category",
    "layout_category",
    "scale_category",
    "background_category",
    "orientation_category",
    "perspective_category"
]

METRIC_COLS = [
    "correction_rate",
    "added_rate",
    "removed_rate"
]

# ---------------- LOAD ----------------

df = pd.read_csv(INPUT_FILE)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ---------------- COMPUTE ERROR-TYPE RATES ----------------

denom = df["original_count"].replace(0, pd.NA)

df["added_rate"] = df["added"] / denom
df["removed_rate"] = df["removed"] / denom

# ---------------- SUMMARISE PER DOMAIN CATEGORY ----------------

summary = {
    "images": {}
}

for metric in METRIC_COLS:
    summary[f"mean_{metric}"] = {}
    summary[f"median_{metric}"] = {}
    summary[f"std_{metric}"] = {}

for col in DOMAIN_COLS:

    if col not in df.columns:
        continue

    for category_name, group in df.groupby(col):

        header = (
            f"{col.replace('_category', '')}_"
            f"{str(category_name).lower().replace(' ', '_')}"
        )

        summary["images"][header] = len(group)

        for metric in METRIC_COLS:
            values = group[metric]

            summary[f"mean_{metric}"][header] = values.mean()
            summary[f"median_{metric}"][header] = values.median()
            summary[f"std_{metric}"][header] = values.std()

# ---------------- SAVE ----------------

summary_df = pd.DataFrame(summary).T
summary_df.to_csv(OUTPUT_FILE)

print(summary_df)
print()
print("Saved:", OUTPUT_FILE)