"""Resolve the image file named in a manifest row.

`manifest_run.csv` stores a relative path such as `scs/node1398_E.jpg` or
`tg/Axis2_Dist40_Left.jpg`. Point PC_SCS_IMAGES and PC_LEGACY_IMAGES at the two
folders that hold the frames: either the parent of the `scs/` and `tg/`
subfolders, or the folder that holds the files directly (both work).
An absolute path in the manifest is used as is.
"""
import os
from pathlib import Path


def image_path(row):
    try:
        src, raw = row["src"], row["path"]
    except (KeyError, TypeError, IndexError):
        src, raw = row.src, row.path
    p = Path(str(raw))
    if p.is_absolute():
        return p
    root = os.environ.get("PC_LEGACY_IMAGES" if str(src) == "TG" else "PC_SCS_IMAGES")
    if not root:
        return p
    q = Path(root) / p
    return q if q.exists() else Path(root) / p.name
