"""Batch SAM 3 text-prompted instance segmentation with resumable, mask-preserving output.

usage: python 20_sam3_segment.py manifest.csv out_dir [--limit N] [--fp16]
manifest columns: key, path         (key = unique image id, used as file stem)
For each image and each prompt (concept) we keep every instance with score >= SCORE_MIN (0.5); the
downstream analysis can raise the threshold (0.5 / 0.6 / 0.7 sensitivity).
Output per image:  out_dir/<key>.npz   masks (packbits, N x H*W/8 bytes), plus  <key>.json  detections.
The Hugging Face token is read from the environment (HF_TOKEN) and never written to disk.
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd, torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model
from imgpath import image_path

PROMPTS = ["person", "car", "truck", "motorcycle", "chair", "table", "potted plant", "trash can", "dumpster",
           "utility pole", "street light", "utility box", "electrical cabinet", "sidewalk", "road", "building"]
SCORE_MIN = 0.5

ap = argparse.ArgumentParser()
ap.add_argument("manifest"); ap.add_argument("out_dir")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--verify", action="store_true")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
tok = os.environ.get("HF_TOKEN")
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
proc = Sam3Processor.from_pretrained("facebook/sam3", token=tok)
model = Sam3Model.from_pretrained("facebook/sam3", token=tok).to(dev).eval()

man = pd.read_csv(a.manifest)
if a.limit:
    man = man.head(a.limit)
t0 = time.time(); done = 0
for i, r in man.iterrows():
    outj = os.path.join(a.out_dir, r.key + ".json")
    if os.path.exists(outj):
        continue
    img = Image.open(image_path(r)).convert("RGB")
    dets, masks = [], []
    with torch.no_grad():
        pv = proc(images=img, text=PROMPTS[0], return_tensors="pt").to(dev)
        ve = model.get_vision_features(pixel_values=pv["pixel_values"])      # image embedding computed ONCE per image
    for p in PROMPTS:
        inp = proc(images=img, text=p, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(vision_embeds=ve, input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
        res = proc.post_process_instance_segmentation(out, threshold=SCORE_MIN, mask_threshold=0.5,
                                                      target_sizes=inp["original_sizes"].tolist())[0]
        m = res["masks"].cpu().numpy().astype(bool); b = res["boxes"].cpu().numpy(); s = res["scores"].cpu().numpy()
        for k in range(len(m)):
            dets.append(dict(prompt=p, score=float(s[k]), box=[float(x) for x in b[k]], area_px=int(m[k].sum())))
            masks.append(np.packbits(m[k].reshape(-1)))
        if a.verify and len(m):
            with torch.no_grad():
                o2 = model(**inp)
            r2 = proc.post_process_instance_segmentation(o2, threshold=SCORE_MIN, mask_threshold=0.5, target_sizes=inp["original_sizes"].tolist())[0]
            print("VERIFY", p, "n", len(s), len(r2["scores"]), "max|dscore|", float(np.abs(np.sort(s) - np.sort(r2["scores"].cpu().numpy())).max()) if len(s) == len(r2["scores"]) else "n/a", flush=True)
    np.savez_compressed(os.path.join(a.out_dir, r.key + ".npz"), masks=np.array(masks) if masks else np.zeros((0, 1), np.uint8),
                        shape=np.array(img.size[::-1]))
    json.dump(dict(key=r.key, path=r.path, size=img.size, dets=dets), open(outj, "w"))
    done += 1
    if done % 10 == 0 or done == 1:
        el = time.time() - t0
        print(f"[{done}] {r.key}  {el/done:.1f}s/img  ({len(dets)} dets)", flush=True)
print("finished", done, "images in", round(time.time() - t0), "s")
