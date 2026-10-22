"""source dataset: read every JPEG straight from the deposit zip, record its EXIF, cache a small copy.

Outputs
  data/bld/img/<label>/<stem>.jpg   long side 768 px, pixel orientation exactly as stored
                                    (the authors' loader does not apply the EXIF orientation
                                    tag, so neither does the cache; the tag is kept in meta)
  data/bld/meta.csv                 one row per image: label, device, capture time, size, ...

The 24-megapixel originals take ~0.4 s each to decode; PIL's draft mode asks libjpeg for a
DCT-domain downscale and cuts that several-fold.
"""
import io
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from PIL import Image, ExifTags

from common import BASE
ZIP = f"{BASE}/data/bld_zip/BLD_dataset.zip"
OUT = f"{BASE}/data/bld"
LONG = 768
TAGS = {v: k for k, v in ExifTags.TAGS.items()}


def one(name):
    label = name.split("/")[-2]
    stem = os.path.splitext(os.path.basename(name))[0]
    dst = f"{OUT}/img/{label}/{stem}.jpg"
    with zipfile.ZipFile(ZIP) as z:
        raw = z.read(name)
    im = Image.open(io.BytesIO(raw))
    w, h = im.size
    ex = im.getexif()
    sub = ex.get_ifd(0x8769) if ex else {}
    row = dict(file=f"{label}/{stem}.jpg", label=label, stem=stem, width=w, height=h,
               bytes=len(raw), make=str(ex.get(TAGS["Make"], "")).strip(),
               model=str(ex.get(TAGS["Model"], "")).strip(),
               orientation=ex.get(TAGS["Orientation"], 1),
               datetime=str(sub.get(TAGS["DateTimeOriginal"], ex.get(TAGS["DateTime"], ""))),
               focal=float(sub.get(TAGS["FocalLength"], 0) or 0),
               focal35=sub.get(TAGS["FocalLengthIn35mmFilm"], ""),
               fnumber=float(sub.get(TAGS["FNumber"], 0) or 0),
               exposure=float(sub.get(TAGS["ExposureTime"], 0) or 0),
               iso=sub.get(TAGS["ISOSpeedRatings"], ""),
               flash=sub.get(TAGS["Flash"], ""),
               lens=str(sub.get(TAGS["LensModel"], "")).strip(),
               has_gps=bool(ex.get_ifd(0x8825)) if ex else False)
    if not os.path.exists(dst):
        im.draft("RGB", (LONG * 2, LONG * 2))
        im = im.convert("RGB")
        s = LONG / max(im.size)
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
        im.save(dst + ".tmp", "JPEG", quality=93)
        os.replace(dst + ".tmp", dst)
    return row


def main():
    with zipfile.ZipFile(ZIP) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    for lab in {n.split("/")[-2] for n in names}:
        os.makedirs(f"{OUT}/img/{lab}", exist_ok=True)
    rows = []
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex:
        for i, r in enumerate(ex.map(one, names, chunksize=8)):
            rows.append(r)
            if (i + 1) % 200 == 0:
                print(i + 1, "/", len(names), flush=True)
    df = pd.DataFrame(rows).sort_values("file")
    df.to_csv(f"{OUT}/meta.csv", index=False)
    print(df.label.value_counts().to_dict())
    print(df.groupby(["label", "model"]).size().to_string())
    print("DONE prep", flush=True)


if __name__ == "__main__":
    main()
