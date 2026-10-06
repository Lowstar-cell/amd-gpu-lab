"""
Inference: das auf SYNTHETISCHEN Szenen trainierte Tiefennetz auf ECHTE Fotos
loslassen -> Domain-Gap-Test. Kein Ground Truth vorhanden, also rein qualitativ
("sieht die vorhergesagte Tiefe plausibel aus?").

    ../.venv-rocm6.4/bin/python infer.py --photos ../echte-fotos
"""

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image, ImageOps

from datagen import depth_to_color
from model import DepthUNet


def load_image(path, res):
    """EXIF-Rotation korrigieren, quadratisch center-croppen, auf res skalieren."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)   # Handy-Fotos sind oft EXIF-gedreht
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((res, res), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0   # (res, res, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", default="../echte-fotos")
    ap.add_argument("--weights", default="depth_unet.pt")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--out", default="real_photos_depth.png")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--scale", type=int, default=1, help="Kacheln fuer die Anzeige hochskalieren")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthUNet(coord=True).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    paths = sorted(p for p in glob.glob(os.path.join(args.photos, "*"))
                   if p.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"[infer] {len(paths)} Fotos in {args.photos}")

    origs, depths = [], []
    for p in paths:
        arr = load_image(p, args.res)
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            d = model(x)[0, 0].cpu().numpy()
        origs.append((arr * 255).astype(np.uint8))
        depths.append(depth_to_color(d))
        print(f"  {os.path.basename(p):32s} depth min/max {d.min():.2f}/{d.max():.2f}")

    # Grid: pro Zelle [Original | vorhergesagte Tiefe], optional hochskaliert
    R, gap, pad = args.res * args.scale, 2, 8
    cell_w = 2 * R + gap
    cols = args.cols
    rows = (len(paths) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * (cell_w + pad) + pad, rows * (R + pad) + pad),
                       (255, 255, 255))
    for i, (o, dd) in enumerate(zip(origs, depths)):
        r, c = divmod(i, cols)
        x0 = pad + c * (cell_w + pad)
        y0 = pad + r * (R + pad)
        canvas.paste(Image.fromarray(o).resize((R, R), Image.BILINEAR), (x0, y0))
        canvas.paste(Image.fromarray(dd).resize((R, R), Image.NEAREST), (x0 + R + gap, y0))
    canvas.save(args.out)
    print(f"[infer] gespeichert: {args.out}  (links Original, rechts Tiefe; rot=nah, blau=fern)")


if __name__ == "__main__":
    main()
