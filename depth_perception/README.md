# Tiefenwahrnehmung — Bild → Tiefe, selbst gelernt

**Option 3 aus `TIEFENWAHRNEHMUNG_NAECHSTE_SCHRITTE.md`: "Rasterizer als Datengenerator".**
Wir nehmen kein fertiges Modell (Depth Anything o.ä.), sondern erzeugen die
Trainingsdaten selbst und trainieren ein eigenes Tiefennetz.

## Die Idee in einem Satz

Die **Vorwärts**-Richtung (3D-Szene → Bild + perfekte Tiefe) ist deterministisch und
leicht — also lassen wir den Rasterizer massenhaft `(RGB, Tiefe)`-Paare erzeugen und
trainieren ein CNN, die schwere **Rückwärts**-Richtung (Bild → Tiefe) zu lernen. Kein
manuelles Labeln.

## Warum das die gestrige Lektion auflöst

Gestern: naives "hell = nah" stirbt bei normalen Szenen. Deshalb bekommen die
generierten Szenen **echte, aber lern­bare** Tiefen-Cues — und *zufällige* Objektfarben,
damit Helligkeit allein wertlos ist:

| Cue | Quelle |
|---|---|
| Boden-Gradient (unten = nah, Horizont = fern) | geneigte Bodenebene mit Z-Gradient |
| Größe (näher = größer) | perspektivische Projektion |
| Verdeckung | echter Z-Buffer |
| ~~Helligkeit~~ | **bewusst zufällig** → naive Methode scheitert |

## Ergebnis (val MAE, kleiner = besser)

| Methode | MAE |
|---|---|
| Naive Helligkeit→Tiefe (mit optimaler affiner Skalierung) | **0.168** |
| Gelerntes CNN (U-Net, 1.93 M Params) | **0.0025** |

→ **CNN ~67× besser.** Visuell in `eval_result.png`: das CNN rekonstruiert die Tiefe
nahezu perfekt, die naive Methode sieht die Objekte gar nicht. Selbst ein *weißes*
Objekt wird korrekt als *nah* erkannt.

## Dateien

| Datei | Zweck |
|---|---|
| `datagen.py` | Schlanker Z-Buffer-Rasterizer + Szenengenerator → `(RGB, Tiefe)` |
| `model.py`   | Kleines U-Net (RGB→Tiefe), optionaler y-Koordinaten-Kanal |
| `train.py`   | Training + Baseline-Vergleich + Visualisierung |

## Benutzung

```bash
# venv mit ROCm-PyTorch
cd depth_perception

# 1) Datenqualität ansehen (Vorschau-Grid: RGB oben, Tiefe unten)
../.venv-rocm6.4/bin/python datagen.py --preview 6

# 2) Trainieren (erzeugt Daten in ~8s, trainiert ~2.5 min auf RX 7900 XTX)
../.venv-rocm6.4/bin/python train.py --epochs 40

# Ergebnis: depth_unet.pt + eval_result.png (4 Reihen: RGB | GT | CNN | naiv)
```

Nützliche Flags: `--no-coord` (Ablation ohne y-Kanal), `--regen` (Cache neu),
`--res`, `--n-train`, `--epochs`.

## Mögliche nächste Schritte

- **Ablation** `--no-coord`: zeigt, wie viel der y-Koordinaten-Kanal beim Boden bringt.
- **Echte 3D-Objekte** statt Billboards (Boxen mit 3 sichtbaren, unterschiedlich
  geshadeten Seiten) → stärkeres Shape-from-Shading-Signal.
- **Generalisierung testen:** das auf synthetischen Szenen trainierte Netz auf ein
  echtes Foto loslassen — wo bricht es? (Domain-Gap.)
- **Gradient-/SSIM-Loss** für schärfere Objektkanten.
- Brücke zur GPU-Daten-Idee: statt Rasterizer den **echten Z-Buffer aus Spielen** als
  Ground Truth (perfekte Tiefe pro Frame).
