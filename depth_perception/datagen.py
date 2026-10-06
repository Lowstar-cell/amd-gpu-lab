"""
Datengenerator fuer das Tiefen-Lernen (Bild -> Tiefe).

Kernidee (Option 3 "Rasterizer als Datengenerator"):
Wir kennen die VORWAERTS-Richtung perfekt (3D-Szene -> Bild + exakte Tiefe).
Also erzeugen wir massenhaft (RGB, perfekte Tiefe)-Paare und trainieren spaeter
ein CNN, die RUECKWAERTS-Richtung (Bild -> Tiefe) zu lernen - ganz ohne manuelles
Labeln.

Damit das ueberhaupt lernbar ist (siehe Notiz TIEFENWAHRNEHMUNG_NAECHSTE_SCHRITTE.md),
bauen wir bewusst echte monokulare Tiefen-Cues ein:
  * Bodenebene mit Tiefengradient (unten = nah, Horizont = fern)
  * Objekte stehen auf dem Boden -> Bild-y-Position korreliert mit Tiefe
  * perspektivische Groesse: naeher = groesser
  * Verdeckung: naehere Objekte ueberdecken fernere (echter Z-Buffer)
ABER: Objektfarben sind ZUFAELLIG. Damit scheitert die naive Regel "hell = nah"
(die gestrige Lektion), waehrend ein CNN aus Position/Groesse/Verdeckung lernen kann.

Reines PyTorch, laeuft auf der ROCm-GPU (.venv-rocm6.4).
"""

import math
import torch


# --------------------------------------------------------------------------
# Rasterizer (schlank, voll vektorisiert ueber Pixel, per-vertex Interpolation)
# --------------------------------------------------------------------------
def rasterize(tris, cols, H, W, bg_rgb, bg_depth):
    """
    Rendert Dreiecke mit Z-Buffer.

    tris : (N, 3, 3)  Screen-Space Vertices (x_px, y_px, z) - kleineres z = naeher
    cols : (N, 3, 3)  per-vertex RGB in [0,1]
    bg_rgb   : (3,)   Hintergrundfarbe (Himmel)
    bg_depth : float  Tiefe der Hintergrund-Pixel

    Returns: rgb (3,H,W), depth (H,W)  -- beide auf tris.device
    """
    device = tris.device
    N = tris.shape[0]

    # Pixelzentren
    ys = (torch.arange(H, device=device).float() + 0.5).view(H, 1).expand(H, W)
    xs = (torch.arange(W, device=device).float() + 0.5).view(1, W).expand(H, W)

    depth = torch.full((H, W), float(bg_depth), device=device)
    rgb = bg_rgb.view(3, 1, 1).expand(3, H, W).clone()

    for t in range(N):
        v0, v1, v2 = tris[t, 0], tris[t, 1], tris[t, 2]

        # Kantenfunktionen (signierte Flaeche der Teildreiecke)
        def edge(a, b):
            return (xs - a[0]) * (b[1] - a[1]) - (ys - a[1]) * (b[0] - a[0])

        w0 = edge(v1, v2)
        w1 = edge(v2, v0)
        w2 = edge(v0, v1)

        # WICHTIG: gleiche Orientierung wie die edge-Funktionen, sonst summieren
        # die baryzentrischen Gewichte zu -1 (area == w0+w1+w2 sein muss).
        area = (v2[0] - v0[0]) * (v1[1] - v0[1]) - (v2[1] - v0[1]) * (v1[0] - v0[0])
        if area.abs() < 1e-6:
            continue  # degeneriertes Dreieck

        # Innen-Test unabhaengig von der Vertex-Reihenfolge (beide Wicklungen)
        inside = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | \
                 ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))

        b0 = w0 / area
        b1 = w1 / area
        b2 = w2 / area

        z = b0 * v0[2] + b1 * v1[2] + b2 * v2[2]
        mask = inside & (z < depth)

        depth = torch.where(mask, z, depth)
        col = b0.unsqueeze(0) * cols[t, 0].view(3, 1, 1) + \
              b1.unsqueeze(0) * cols[t, 1].view(3, 1, 1) + \
              b2.unsqueeze(0) * cols[t, 2].view(3, 1, 1)
        rgb = torch.where(mask.unsqueeze(0), col, rgb)

    return rgb, depth


# --------------------------------------------------------------------------
# Szenengenerator: einfache Lochkamera, die auf eine Bodenebene blickt
# --------------------------------------------------------------------------
class SceneConfig:
    z_near = 2.0
    z_far = 30.0
    cam_height = 1.5            # Kamerahoehe ueber dem Boden (Welt-Einheiten)
    focal = 0.75               # Brennweite als Anteil der Bildbreite
    horizon_frac = 0.42        # Horizont-Hoehe im Bild (Anteil von oben)
    n_obj_range = (3, 7)


def _quad(x0, y0, x1, y1, z, device):
    """Achsenparalleles Rechteck (2 Dreiecke) bei konstanter Tiefe z."""
    return torch.tensor([
        [[x0, y0, z], [x1, y0, z], [x1, y1, z]],
        [[x0, y0, z], [x1, y1, z], [x0, y1, z]],
    ], device=device, dtype=torch.float32)


def make_scene(H, W, device, cfg=SceneConfig, generator=None):
    """
    Baut eine Szene und rendert sie.
    Returns: rgb (3,H,W) in [0,1], depth_norm (1,H,W) in [0,1] (1=nah, 0=fern)
    """
    def rand(*shape, lo=0.0, hi=1.0):
        scalar = len(shape) == 0
        size = (1,) if scalar else shape
        r = torch.rand(*size, generator=generator, device=device) * (hi - lo) + lo
        return r[0] if scalar else r

    f = cfg.focal * W
    horizon = cfg.horizon_frac * H
    cx = W / 2.0

    tris_list = []
    cols_list = []

    # --- Bodenebene: unten (nah) bis Horizont (fern), Tiefengradient ueber z ---
    ground_z_near = cfg.z_near
    ground_z_far = cfg.z_far
    base = rand(lo=0.30, hi=0.45).item()   # graeulicher Boden, leichte Variation
    # 2 Dreiecke; z interpoliert von unten (nah) nach Horizont (fern)
    g = torch.tensor([
        [[0, H, ground_z_near], [W, H, ground_z_near], [W, horizon, ground_z_far]],
        [[0, H, ground_z_near], [W, horizon, ground_z_far], [0, horizon, ground_z_far]],
    ], device=device, dtype=torch.float32)
    # leichter Fog-Cue in der Bodenfarbe: vorne (nah) heller, hinten dunkler
    c_near = torch.tensor([base, base * 0.95, base * 0.85], device=device)
    c_far = c_near * 0.6
    gc = torch.stack([
        torch.stack([c_near, c_near, c_far]),
        torch.stack([c_near, c_far, c_far]),
    ])
    tris_list.append(g)
    cols_list.append(gc)

    # --- Objekte: stehen auf dem Boden, Groesse ~ 1/Z, zufaellige Farbe ---
    n_obj = int(rand(lo=cfg.n_obj_range[0], hi=cfg.n_obj_range[1] + 1).item())
    for _ in range(n_obj):
        # Tiefe naeher gewichten (u^2), damit auch grosse nahe Objekte vorkommen
        u = rand().item()
        Z = cfg.z_near + 0.5 + (cfg.z_far - 2.5) * (u ** 2)
        # horizontale Position an Tiefe koppeln -> bleibt im Bild, auch wenn nah
        Xw = rand(lo=-0.62, hi=0.62).item() * Z   # horizontale Weltposition
        obj_h = rand(lo=0.6, hi=2.2).item()       # Objekthoehe (Welt)
        obj_w = rand(lo=0.4, hi=1.6).item()       # Objektbreite (Welt)

        # Projektion (Lochkamera, Boden = Welt-Y 0, Kamera blickt leicht nach unten)
        sx = cx + f * Xw / Z
        sy_foot = horizon + f * cfg.cam_height / Z          # Bodenkontakt
        sy_top = horizon + f * (cfg.cam_height - obj_h) / Z  # Oberkante
        half_w = 0.5 * f * obj_w / Z

        x0, x1 = sx - half_w, sx + half_w
        y0, y1 = sy_top, sy_foot

        # zufaellige Grundfarbe -> "hell = nah" ist hier WERTLOS
        c = rand(3, lo=0.15, hi=1.0)
        # dezentes vertikales Shading (oben heller) als Textur, nicht tiefeneindeutig
        c_top = (c * 1.15).clamp(0, 1)
        c_bot = (c * 0.65).clamp(0, 1)

        quad = torch.tensor([
            [[x0, y0, Z], [x1, y0, Z], [x1, y1, Z]],
            [[x0, y0, Z], [x1, y1, Z], [x0, y1, Z]],
        ], device=device, dtype=torch.float32)
        qc = torch.stack([
            torch.stack([c_top, c_top, c_bot]),
            torch.stack([c_top, c_bot, c_bot]),
        ])
        tris_list.append(quad)
        cols_list.append(qc)

    tris = torch.cat(tris_list, dim=0)
    cols = torch.cat(cols_list, dim=0)

    # Himmel: hellblau, sehr fern
    sky = torch.tensor([0.55, 0.70, 0.90], device=device)
    rgb, depth = rasterize(tris, cols, H, W, bg_rgb=sky, bg_depth=cfg.z_far)

    # Tiefe -> normalisierte inverse Tiefe (Disparity), 1 = nah, 0 = fern
    inv = 1.0 / depth.clamp(min=cfg.z_near)
    inv_near = 1.0 / cfg.z_near
    inv_far = 1.0 / cfg.z_far
    depth_norm = ((inv - inv_far) / (inv_near - inv_far)).clamp(0, 1)

    return rgb.clamp(0, 1), depth_norm.unsqueeze(0)


def generate_dataset(n, H, W, device, seed=0):
    """Erzeugt n Paare. Returns rgb (n,3,H,W), depth (n,1,H,W) auf CPU."""
    g = torch.Generator(device=device).manual_seed(seed)
    rgb = torch.empty(n, 3, H, W)
    depth = torch.empty(n, 1, H, W)
    for i in range(n):
        r, d = make_scene(H, W, device, generator=g)
        rgb[i] = r.cpu()
        depth[i] = d.cpu()
    return rgb, depth


def depth_to_color(d):
    """Tiefe (H,W in [0,1], 1=nah) -> Jet-artige RGB-uint8 (H,W,3). Nur numpy."""
    import numpy as np
    anchors = np.array([
        [0.0, 0.0, 0.5],   # fern  -> dunkelblau
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],   # nah   -> rot
    ])
    pos = np.linspace(0, 1, len(anchors))
    d = np.clip(d, 0, 1)
    rgb = np.stack([np.interp(d, pos, anchors[:, k]) for k in range(3)], axis=-1)
    return (rgb * 255).astype(np.uint8)


def save_preview(rgb, depth, out, pad=4):
    """Grid-PNG: obere Reihe RGB, untere Reihe farbcodierte Tiefe."""
    import numpy as np
    from PIL import Image
    n, _, H, W = rgb.shape
    rgb_np = (rgb.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    canvas = Image.new("RGB", (n * (W + pad) + pad, 2 * (H + pad) + pad), (240, 240, 240))
    for i in range(n):
        x = pad + i * (W + pad)
        canvas.paste(Image.fromarray(rgb_np[i]), (x, pad))
        canvas.paste(Image.fromarray(depth_to_color(depth[i, 0].numpy())), (x, 2 * pad + H))
    canvas.save(out)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", type=int, default=6, help="Anzahl Vorschau-Szenen")
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--out", default="preview_scenes.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[datagen] device={device}, res={args.res}")

    import time
    t = time.time()
    rgb, depth = generate_dataset(args.preview, args.res, args.res, device, seed=1)
    dt = time.time() - t
    print(f"[datagen] {args.preview} Szenen in {dt:.2f}s ({dt/args.preview*1000:.0f} ms/Szene)")

    save_preview(rgb, depth, args.out)
    print(f"[datagen] Vorschau gespeichert: {args.out}")
