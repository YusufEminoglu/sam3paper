"""Instance-level visual audit material for the SAM 3 output (desk-based, single rater; disclosed as such).

usage: python 50_audit_sheets.py <seg_dir> <manifest_run.csv> <out_dir> <n_images> [seed]
- stratified sample of images (source x has-vehicle x has-furniture), fixed seed
- one PNG per image: original | overlay with numbered instances (score >= 0.6, obstacle classes + person)
- audit_template.csv: one row per instance (judgement to be filled: TP / FP) and one row per image for missed instances
"""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

seg_dir, manifest, out_dir, n = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42
out_dir.mkdir(parents=True, exist_ok=True)
SHOW = ["person", "car", "truck", "motorcycle", "chair", "table", "potted plant", "trash can", "dumpster", "utility pole",
        "street light", "utility box", "electrical cabinet"]
COL = {c: tuple(int(x) for x in np.random.default_rng(i + 3).integers(60, 255, 3)) for i, c in enumerate(SHOW)}
man = pd.read_csv(manifest)
info = []
for r in man.itertuples():
    jf = seg_dir / (r.key + ".json")
    if jf.exists():
        d = json.load(open(jf))["dets"]
        d = [x for x in d if x["score"] >= 0.6 and x["prompt"] in SHOW]
        info.append(dict(key=r.key, path=r.path, src=r.src, n=len(d), veh=any(x["prompt"] in ("car", "truck", "motorcycle") for x in d),
                         furn=any(x["prompt"] in ("chair", "table", "potted plant") for x in d)))
df = pd.DataFrame(info)
df = df[df.n > 0]
rng = np.random.default_rng(seed)
df["stratum"] = df.src + "_" + df.veh.astype(str) + "_" + df.furn.astype(str)
pick = []
per = max(1, n // df.stratum.nunique())
for s, g in df.groupby("stratum"):
    pick += list(g.sample(min(len(g), per), random_state=int(rng.integers(1e6))).key)
rest = df[~df.key.isin(pick)].sample(frac=1, random_state=seed).key.tolist()
pick += rest[: max(0, n - len(pick))]
pick = pick[:n]
rows, irows = [], []
font = ImageFont.load_default()
for key in pick:
    r = man[man.key == key].iloc[0]
    j = json.load(open(seg_dir / (key + ".json")))
    npz = np.load(seg_dir / (key + ".npz")); shape = tuple(npz["shape"])
    img = Image.open(r.path).convert("RGB")
    ov = np.array(img).copy()
    idx = 0
    labels = []
    for i, d in enumerate(j["dets"]):
        if d["score"] < 0.6 or d["prompt"] not in SHOW:
            continue
        idx += 1
        m = np.unpackbits(npz["masks"][i])[: shape[0] * shape[1]].reshape(shape).astype(bool)
        ov[m] = (0.5 * ov[m] + 0.5 * np.array(COL[d["prompt"]])).astype(np.uint8)
        ys, xs = np.where(m)
        labels.append((idx, int(xs.mean()), int(ys.mean()), d["prompt"], d["score"]))
        irows.append(dict(key=key, inst=idx, cls=d["prompt"], score=round(d["score"], 3), judgement=""))
    ovi = Image.fromarray(ov); dr = ImageDraw.Draw(ovi)
    for k, x, y, c, s in labels:
        dr.rectangle([x - 8, y - 6, x + 8 + 6 * len(str(k)), y + 8], fill=(0, 0, 0)); dr.text((x - 6, y - 5), str(k), fill=(255, 255, 0), font=font)
    sheet = Image.new("RGB", (1280 + 260, 640), (255, 255, 255))
    sheet.paste(img, (0, 0)); sheet.paste(ovi, (640, 0))
    d2 = ImageDraw.Draw(sheet)
    for t, (k, x, y, c, s) in enumerate(labels[:34]):
        d2.text((1290, 6 + 18 * t), f"{k}: {c} {s:.2f}", fill=(0, 0, 0), font=font)
    sheet.save(out_dir / (key + ".png"))
    rows.append(dict(key=key, src=r.src, n_inst=idx, missed_vehicle="", missed_furniture="", missed_infrastructure="", missed_person="", note=""))
pd.DataFrame(irows).to_csv(out_dir / "audit_instances_template.csv", index=False)
pd.DataFrame(rows).to_csv(out_dir / "audit_images_template.csv", index=False)
print("audit images:", len(pick), " instances:", len(irows))
