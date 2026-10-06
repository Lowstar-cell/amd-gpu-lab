"""
Training: CNN lernt aus selbst-generierten (Bild, perfekte Tiefe)-Paaren die
Rueckwaerts-Richtung Bild -> Tiefe.

Am Ende vergleichen wir gegen die NAIVE Baseline von gestern ("Helligkeit = Tiefe",
sogar mit optimaler affiner Skalierung). Erwartung: Das CNN schlaegt sie deutlich,
weil es Position/Groesse/Verdeckung nutzt statt nur Helligkeit.

    ../.venv-rocm6.4/bin/python train.py --epochs 30
"""

import argparse
import os
import time

import torch
import torch.nn as nn

from datagen import generate_dataset, depth_to_color
from model import DepthUNet


def affine_fit(pred, target):
    """Beste a*pred+b (global, least squares) -> gibt der Baseline die faire Chance."""
    x = pred.reshape(-1)
    y = target.reshape(-1)
    A = torch.stack([x, torch.ones_like(x)], dim=1)
    sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)
    a, b = sol[0], sol[1]
    return (a * pred + b).clamp(0, 1)


def mae(a, b):
    return (a - b).abs().mean().item()


def save_eval_grid(rgb, gt, pred, naive, out, n=6, pad=4):
    """4 Reihen: RGB | Ground-Truth | CNN-Vorhersage | naive Baseline."""
    import numpy as np
    from PIL import Image
    n = min(n, rgb.shape[0])
    H, W = rgb.shape[2], rgb.shape[3]
    rows = 4
    canvas = Image.new("RGB", (n * (W + pad) + pad, rows * (H + pad) + pad), (240, 240, 240))
    rgb_np = (rgb[:n].clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    for i in range(n):
        x = pad + i * (W + pad)
        canvas.paste(Image.fromarray(rgb_np[i]), (x, pad))
        for r, t in enumerate([gt, pred, naive], start=1):
            col = depth_to_color(t[i, 0].cpu().numpy())
            canvas.paste(Image.fromarray(col), (x, pad + r * (H + pad)))
    canvas.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-coord", action="store_true", help="ohne y-Koordinaten-Kanal")
    ap.add_argument("--regen", action="store_true", help="Daten-Cache ignorieren")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  res={args.res}  coord={not args.no_coord}")

    # ---- Daten (mit Cache) ----
    cache = f"dataset_{args.res}_{args.n_train}_{args.n_val}.pt"
    if os.path.exists(cache) and not args.regen:
        print(f"[train] lade Daten-Cache {cache}")
        d = torch.load(cache)
        rgb_tr, dep_tr, rgb_va, dep_va = d["rgb_tr"], d["dep_tr"], d["rgb_va"], d["dep_va"]
    else:
        print(f"[train] generiere {args.n_train}+{args.n_val} Szenen ...")
        t = time.time()
        rgb_tr, dep_tr = generate_dataset(args.n_train, args.res, args.res, device, seed=1)
        rgb_va, dep_va = generate_dataset(args.n_val, args.res, args.res, device, seed=999)
        print(f"[train] Daten erzeugt in {time.time()-t:.1f}s")
        torch.save({"rgb_tr": rgb_tr, "dep_tr": dep_tr, "rgb_va": rgb_va, "dep_va": dep_va}, cache)

    rgb_va_d, dep_va_d = rgb_va.to(device), dep_va.to(device)

    # ---- Modell ----
    model = DepthUNet(coord=not args.no_coord).to(device)
    print(f"[train] U-Net Parameter: {model.num_params()/1e6:.2f} M")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossfn = nn.L1Loss()

    n = rgb_tr.shape[0]
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        running = 0.0
        t = time.time()
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            x = rgb_tr[idx].to(device)
            y = dep_tr[idx].to(device)
            pred = model(x)
            loss = lossfn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * idx.numel()
        sched.step()

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                vpred = torch.cat([model(rgb_va_d[i:i+args.batch])
                                   for i in range(0, rgb_va_d.shape[0], args.batch)])
                vmae = mae(vpred, dep_va_d)
            print(f"[train] epoch {epoch:3d}  train_L1={running/n:.4f}  "
                  f"val_MAE={vmae:.4f}  ({time.time()-t:.1f}s)")

    # ---- Finale Auswertung + Baseline-Vergleich ----
    model.eval()
    with torch.no_grad():
        vpred = torch.cat([model(rgb_va_d[i:i+args.batch])
                           for i in range(0, rgb_va_d.shape[0], args.batch)])
    cnn_mae = mae(vpred, dep_va_d)

    # Naive Baseline: Helligkeit -> Tiefe, mit global optimaler affiner Skalierung
    gray = rgb_va_d.mean(1, keepdim=True)
    naive = affine_fit(gray, dep_va_d)
    naive_mae = mae(naive, dep_va_d)

    print("\n=== Ergebnis (val MAE, kleiner = besser) ===")
    print(f"  Naive Helligkeit->Tiefe (best. affine): {naive_mae:.4f}")
    print(f"  Gelerntes CNN:                          {cnn_mae:.4f}")
    print(f"  -> CNN ist {naive_mae/max(cnn_mae,1e-6):.1f}x besser")

    torch.save(model.state_dict(), "depth_unet.pt")
    save_eval_grid(rgb_va_d, dep_va_d, vpred, naive, "eval_result.png", n=6)
    print("\n[train] Modell -> depth_unet.pt   Visualisierung -> eval_result.png")


if __name__ == "__main__":
    main()
