"""Established urban-scene baseline: SegFormer-B5 (ADE20K, 640x640) on the SAME images, exported in the SAM 3 file format
(json + packed masks) so that 30_ecw_v2.py can run unchanged on it (seg_dir = seg_sf).
Semantic classes are split into instances with connected components (area >= MIN_PX).  ADE20K has no dumpster / utility box /
electrical cabinet class: these obstacle classes are simply unavailable to the baseline (a fixed-vocabulary limitation).
usage: python 25_segformer_baseline.py manifest.csv out_dir [threads]
"""
import sys, os, json, numpy as np, pandas as pd, torch, cv2
from PIL import Image
from pathlib import Path
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL = "nvidia/segformer-b5-finetuned-ade-640-640"
MIN_PX = 250
# SAM-prompt name -> substrings of ADE20K label names (first label of the comma list)
MAP = {"person": ["person"], "car": ["car"], "truck": ["truck"], "motorcycle": ["minibike", "motorbike"], "chair": ["chair"],
       "table": ["table"], "potted plant": ["plant"], "trash can": ["ashcan", "trash can"], "utility pole": ["pole"],
       "street light": ["streetlight", "street lamp"], "sidewalk": ["sidewalk"], "road": ["road"], "building": ["building"]}

man = pd.read_csv(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(exist_ok=True, parents=True)
torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 6)
proc = SegformerImageProcessor.from_pretrained(MODEL)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL).eval()
id2 = {int(k): v.lower() for k, v in model.config.id2label.items()}
cls_ids = {}
for name, keys in MAP.items():
    ids = [i for i, l in id2.items() if any(l.split(",")[0].strip() == k or l.startswith(k) for k in keys)]
    cls_ids[name] = ids
print({k: [id2[i] for i in v] for k, v in cls_ids.items()})
for r in man.itertuples():
    if (out / (r.key + ".json")).exists():
        continue
    img = Image.open(r.path).convert("RGB")
    inp = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        lg = model(**inp).logits
    up = torch.nn.functional.interpolate(lg, size=img.size[::-1], mode="bilinear", align_corners=False)
    prob = up.softmax(1)[0]
    pred = prob.argmax(0).numpy()
    dets, masks = [], []
    for name, ids in cls_ids.items():
        if not ids:
            continue
        m = np.isin(pred, ids).astype(np.uint8)
        if name in ("sidewalk", "road", "building"):
            comps = [m.astype(bool)]
        else:
            n, lab = cv2.connectedComponents(m, connectivity=8)
            comps = [lab == k for k in range(1, n) if (lab == k).sum() >= MIN_PX]
        for c in comps:
            sc = float(prob[ids][:, torch.from_numpy(c)].max(0).values.mean()) if c.any() else 0.0
            dets.append(dict(prompt=name, score=sc, box=[0, 0, 0, 0], area_px=int(c.sum())))
            masks.append(np.packbits(c.reshape(-1)))
    np.savez_compressed(out / (r.key + ".npz"), masks=np.array(masks) if masks else np.zeros((0, 1), np.uint8), shape=np.array(img.size[::-1]))
    json.dump(dict(key=r.key, path=r.path, size=img.size, dets=dets), open(out / (r.key + ".json"), "w"))
print("done")
