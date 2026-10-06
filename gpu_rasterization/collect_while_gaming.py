"""
GPU Datensammlung im Hintergrund
Läuft während du zockst und sammelt kontinuierlich Trainingsdaten!
"""

from gpu_rasterize import AMDRasterizer
from data_collection.collector import RasterizationDataCollector
import numpy as np
import time
from datetime import datetime
import os

# ========== EINSTELLUNGEN ==========
SCENES_PER_SAVE = 100        # Alle 100 Scenes speichern
TRIANGLES_PER_SCENE = 15     # Anzahl Triangles pro Scene
RESOLUTION = 256             # 256x256 (niedriger = schneller)
DELAY_BETWEEN_SCENES = 0.1   # Sekunden zwischen Scenes (weniger = mehr Daten, aber mehr Last)
TOTAL_DURATION_MINUTES = 60  # 1 Stunde

print("=" * 60)
print("GPU DATENSAMMLUNG IM HINTERGRUND")
print("=" * 60)
print(f"""
EINSTELLUNGEN:
  Resolution:         {RESOLUTION}x{RESOLUTION}
  Triangles/Scene:    {TRIANGLES_PER_SCENE}
  Speichern alle:     {SCENES_PER_SAVE} Scenes
  Delay:              {DELAY_BETWEEN_SCENES}s
  Laufzeit:           {TOTAL_DURATION_MINUTES} Minuten

WICHTIG: Dieses Script läuft im Hintergrund!
         Lass es einfach laufen während du zockst.

STARTEN: Drücke Enter...
STOPPEN: Strg+C
""")

input()

# Setup
rasterizer = AMDRasterizer(width=RESOLUTION, height=RESOLUTION)
collector = RasterizationDataCollector(save_dir='./data/gaming')
os.makedirs('./data/gaming', exist_ok=True)

# Stats
start_time = time.time()
total_scenes = 0
total_samples = 0
save_counter = 0

print("\n[GESTARTET] Sammle Daten...")
print("=" * 60)

try:
    end_time = start_time + (TOTAL_DURATION_MINUTES * 60)

    while time.time() < end_time:
        # Generiere zufällige Scene
        vertices = []
        for _ in range(TRIANGLES_PER_SCENE):
            # Random Triangle (NDC space)
            tri = np.random.rand(3, 3).astype(np.float32)
            tri[:, :2] = tri[:, :2] * 2 - 1  # [-1, 1] range
            tri[:, 2] = tri[:, 2]             # depth [0, 1]
            vertices.append(tri)

        vertices = np.array(vertices)
        colors = np.random.rand(TRIANGLES_PER_SCENE, 3).astype(np.float32)
        mvp = np.eye(4, dtype=np.float32)

        # Rasterize (silent mode)
        _, features = rasterizer.rasterize_triangles(
            vertices, colors, mvp, collect_data=True
        )

        # Sammeln
        if features is not None:
            collector.collect_from_rasterization(
                features['barycentric_coords'],
                features['depth'],
                features['coverage']
            )

        total_scenes += 1

        # Periodisch speichern
        if total_scenes % SCENES_PER_SAVE == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'gaming_data_{timestamp}.h5'

            if len(collector.features) > 0:
                filepath = collector.save_dataset(filename)
                save_counter += 1
                total_samples += len(collector.features)

                # Stats
                elapsed = (time.time() - start_time) / 60
                samples_per_min = total_samples / elapsed if elapsed > 0 else 0
                remaining = TOTAL_DURATION_MINUTES - elapsed

                print(f"\n[{int(elapsed):02d}m] Scene {total_scenes}")
                print(f"  Gespeichert: {len(collector.features)} samples -> {filename}")
                print(f"  Total:       {total_samples} samples ({samples_per_min:.0f}/min)")
                print(f"  Verbleibend: {remaining:.1f} Minuten")

                collector.clear()

        # Kurze Pause (damit Zocken nicht laggt)
        time.sleep(DELAY_BETWEEN_SCENES)

    print("\n" + "=" * 60)
    print("FERTIG!")
    print("=" * 60)

except KeyboardInterrupt:
    print("\n\n[GESTOPPT] Durch Benutzer...")

# Finale Stats
elapsed_total = (time.time() - start_time) / 60

# Letzte Daten speichern
if len(collector.features) > 0:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    collector.save_dataset(f'gaming_data_final_{timestamp}.h5')
    total_samples += len(collector.features)

print(f"""
ZUSAMMENFASSUNG:
  Laufzeit:        {elapsed_total:.1f} Minuten
  Scenes:          {total_scenes}
  Total Samples:   {total_samples:,}
  Gespeichert:     {save_counter + 1} Dateien

Daten in: ./data/gaming/*.h5

Nächste Schritte:
  1. Alle .h5 Dateien kombinieren
  2. In Training laden
  3. Model trainieren!
""")
