# TO ASSESS RE-ANNOTATION PER IMAGE - LINKED TO DOMAIN PROPERTIES
# SAVE RESULTS UNDER: "results/reannotation-per-domain.csv"

import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

# ---------------- CONFIG ----------------

RUOD_FILE = "datasets/comparison-instances/comparison-ruod-instances.json"
RUOD_R_FILE = "datasets/comparison-instances/comparison-ruod-r-instances.json"

DOMAIN_FILE = "domain-labels/comparison_domain_labels_ruod-r.csv"

OUT_CSV = "results/reannotation-per-image-with-domain.csv"

Path("results").mkdir(parents=True, exist_ok=True)

IOU_THRESH = 0.5

# ---------------- LOAD ----------------

with open(RUOD_FILE) as f:
    data_r = json.load(f)

with open(RUOD_R_FILE) as f:
    data_u = json.load(f)

domain = pd.read_csv(DOMAIN_FILE)

# ---------------- HELPERS ----------------

def norm(name):
    return Path(str(name)).name.strip().lower()


# normalize filenames
domain["image_file"] = domain["image_file"].apply(norm)


# ---------------- map filename -> image_id ----------------

def build_map(data):
    return {
        norm(img["file_name"]): img["id"]
        for img in data["images"]
    }

map_r = build_map(data_r)
map_u = build_map(data_u)

shared_files = sorted(set(map_r) & set(map_u))

# ---------------- annotations ----------------

def build_ann(data):
    d = defaultdict(list)
    for ann in data["annotations"]:
        d[ann["image_id"]].append(ann)
    return d

ann_r = build_ann(data_r)
ann_u = build_ann(data_u)

# ---------------- IoU ----------------

def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter

    return inter / (union + 1e-9)

# ---------------- match annotations ----------------

def match(r_list, u_list):
    used_u = set()
    matched = 0

    for ar in r_list:
        best = -1
        best_j = -1

        for j, au in enumerate(u_list):
            if j in used_u:
                continue

            if ar["category_id"] != au["category_id"]:
                continue

            score = iou(ar["bbox"], au["bbox"])

            if score > best:
                best = score
                best_j = j

        if best >= IOU_THRESH:
            used_u.add(best_j)
            matched += 1

    removed = len(r_list) - matched
    added = len(u_list) - len(used_u)

    return added, removed, matched

# ---------------- build instance evaluation ----------------

rows = []
missing_in_ruod = []
missing_in_ruod_r = []

for f in shared_files:
    if f not in map_r:
        missing_in_ruod.append(f)
        continue

    if f not in map_u:
        missing_in_ruod_r.append(f)
        continue

    id_r = map_r[f]
    id_u = map_u[f]

    added, removed, matched = match(
        ann_r[id_r],
        ann_u[id_u]
    )

    original_count = matched + removed
    revised_count = matched + added
    correction_count = added + removed

    if original_count > 0:
        correction_rate = correction_count / original_count
    else:
        correction_rate = 0

    if added > 0 and removed == 0:
        category = "only_added"
    elif removed > 0 and added == 0:
        category = "only_removed"
    elif added > 0 and removed > 0:
        category = "both"
    else:
        category = "unchanged"

    rows.append({
        "image_file": f,
        "ruod_image_id": id_r,
        "ruod_r_image_id": id_u,

        "added": added,
        "removed": removed,
        "matched": matched,

        "original_count": original_count,
        "revised_count": revised_count,
        "correction_count": correction_count,
        "correction_rate": round(correction_rate, 2),

        "category": category
    })

df = pd.DataFrame(rows)

# ---------------- merge domain labels ----------------

domain_cols = [
    "image_file",
    "visibility_category",
    "color_category",
    "illumination_category",
    "layout_category",
    "scale_category",
    "background_category",
    "orientation_category",
    "perspective_category"
]

available_domain_cols = [
    col for col in domain_cols
    if col in domain.columns
]

missing_domain_cols = [
    col for col in domain_cols
    if col not in domain.columns
]

if missing_domain_cols:
    print("WARNING missing domain columns:")
    for col in missing_domain_cols:
        print(" -", col)

df = df.merge(
    domain[available_domain_cols],
    on="image_file",
    how="left"
)

# ---------------- save ----------------

df.to_csv(OUT_CSV, index=False)

# ---------------- report ----------------

print("Saved:")
print(" -", OUT_CSV)

print()
print("Shared images:", len(shared_files))
print("Rows evaluated:", len(df))

print()
print("Missing from RUOD JSON:", len(missing_in_ruod))
print("Missing from RUOD-R JSON:", len(missing_in_ruod_r))

print()
print("Category distribution:")
print(df["category"].value_counts())

print()
print("Domain labels missing after merge:")
print(df[available_domain_cols].isna().sum())