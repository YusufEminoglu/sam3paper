"""SAM 3 vs SegFormer-B5 (ADE20K) on identical images.

usage: python 45_compare_models.py <seg_sam> <seg_sf> <manifest_run.csv> <ecw_sam.csv> <ecw_sf.csv> <out_dir>
(1) per-class pixel agreement: IoU of the union mask per class and image, over images where at least one model detects the class
(2) detection agreement: share of images in which both / only one model detects the class
(3) pipeline-level agreement: correlation and mean absolute difference of best-side ECW, and agreement on the impassable tier
Agreement between two models is NOT accuracy; it shows where the models diverge (fixed vocabulary vs open vocabulary).
"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path

sam, sf, man, ecw_a, ecw_b, out = Path(sys.argv[1]), Path(sys.argv[2]), pd.read_csv(sys.argv[3]), sys.argv[4], sys.argv[5], Path(sys.argv[6])
out.mkdir(exist_ok=True, parents=True)
CLS = ["person", "car", "truck", "motorcycle", "chair", "table", "potted plant", "trash can", "utility pole", "street light", "sidewalk", "road", "building"]
THR = 0.6


def union(dir_, key, shape_hint=None):
    j = json.load(open(dir_ / (key + ".json"))); z = np.load(dir_ / (key + ".npz")); shape = tuple(z["shape"])
    u = {}
    for i, d in enumerate(j["dets"]):
        if d["score"] < THR or d["prompt"] not in CLS:
            continue
        m = np.unpackbits(z["masks"][i])[: shape[0] * shape[1]].reshape(shape).astype(bool)
        u[d["prompt"]] = u.get(d["prompt"], np.zeros(shape, bool)) | m
    return u, shape


rows = []
for r in man.itertuples():
    if not ((sam / (r.key + ".json")).exists() and (sf / (r.key + ".json")).exists()):
        continue
    a, shp = union(sam, r.key); b, _ = union(sf, r.key)
    for c in CLS:
        ma, mb = a.get(c), b.get(c)
        pa, pb = ma is not None and ma.any(), mb is not None and mb.any()
        if not (pa or pb):
            continue
        iou = np.nan
        if pa and pb:
            iou = (ma & mb).sum() / (ma | mb).sum()
        rows.append(dict(key=r.key, cls=c, sam=int(pa), sf=int(pb), iou=iou, area_sam=int(ma.sum()) if pa else 0, area_sf=int(mb.sum()) if pb else 0))
d = pd.DataFrame(rows); d.to_csv(out / "model_agreement_images.csv", index=False)
s = d.groupby("cls").agg(n_images=("key", "size"), both=("iou", lambda x: x.notna().sum()), sam_only=("sam", lambda x: 0),
                         mean_iou_when_both=("iou", "mean"), median_iou_when_both=("iou", "median")).reset_index()
s["sam_only"] = d[(d.sam == 1) & (d.sf == 0)].groupby("cls").size().reindex(s.cls).fillna(0).values.astype(int)
s["sf_only"] = d[(d.sam == 0) & (d.sf == 1)].groupby("cls").size().reindex(s.cls).fillna(0).values.astype(int)
s.round(3).to_csv(out / "model_agreement_by_class.csv", index=False)
print(s.round(2).to_string(index=False))

ea, eb = pd.read_csv(ecw_a), pd.read_csv(ecw_b)
ok = lambda e: e[(e.coverage >= 0.5) & e.ecw_min.notna()].groupby("node").ecw_min.max()
m = pd.concat([ok(ea).rename("sam"), ok(eb).rename("sf")], axis=1).dropna()
if len(m) > 3:
    res = dict(n=len(m), pearson=m.sam.corr(m.sf), spearman=m.sam.corr(m.sf, method="spearman"), mad=(m.sam - m.sf).abs().mean(),
               bias_sf_minus_sam=(m.sf - m.sam).mean(), agree_impassable=((m.sam < 0.9) == (m.sf < 0.9)).mean())
    pd.Series(res).to_csv(out / "pipeline_agreement.csv"); print(pd.Series(res).round(3).to_string())
