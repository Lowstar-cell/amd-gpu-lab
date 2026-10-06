"""
Kombiniere alle gesammelten Gaming-Daten in ein großes Dataset
"""

import h5py
import numpy as np
import glob
import os

print("=" * 60)
print("GAMING DATA COMBINER")
print("=" * 60)

# Alle .h5 Dateien finden
data_files = glob.glob('./data/gaming/*.h5')

if len(data_files) == 0:
    print("\n[ERROR] Keine Daten gefunden in ./data/gaming/")
    print("        Starte zuerst: START_DATENSAMMLUNG.bat")
    exit(1)

print(f"\nGefunden: {len(data_files)} Dateien")

all_features = []
all_labels = []

# Alle laden
for i, filepath in enumerate(sorted(data_files), 1):
    print(f"  [{i}/{len(data_files)}] Lade {os.path.basename(filepath)}...", end='')

    try:
        with h5py.File(filepath, 'r') as f:
            features = f['features'][:]
            labels = f['labels'][:]

            all_features.append(features)
            all_labels.append(labels)

            print(f" {features.shape[0]:,} samples")
    except Exception as e:
        print(f" FEHLER: {e}")

# Kombinieren
print("\nKombiniere...")
combined_features = np.concatenate(all_features, axis=0)
combined_labels = np.concatenate(all_labels, axis=0)

print(f"  Total Samples: {combined_features.shape[0]:,}")
print(f"  Feature Dim:   {combined_features.shape[1]}")

# Speichern
output_file = './data/gaming_combined.h5'
print(f"\nSpeichere: {output_file}")

with h5py.File(output_file, 'w') as f:
    f.create_dataset('features', data=combined_features, compression='gzip')
    f.create_dataset('labels', data=combined_labels, compression='gzip')

    f.attrs['num_samples'] = combined_features.shape[0]
    f.attrs['feature_dim'] = combined_features.shape[1]
    f.attrs['num_files'] = len(data_files)
    f.attrs['description'] = 'Combined Gaming Data Collection'

print(f"\n[OK] Kombiniertes Dataset: {output_file}")
print(f"     {combined_features.shape[0]:,} Samples ready für Training!")

# Stats
print("\n" + "=" * 60)
print("DATASET STATISTIKEN:")
print("=" * 60)

print(f"\nFeatures (per sample):")
feature_names = ['Barycentric_W0', 'Barycentric_W1', 'Barycentric_W2',
                 'Depth', 'Screen_X', 'Screen_Y', 'Coverage']

for i, name in enumerate(feature_names):
    if i < combined_features.shape[1]:
        values = combined_features[:, i]
        print(f"  {name:20s}: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}")

print(f"\nLabels (Triangle IDs):")
unique_labels = len(np.unique(combined_labels))
print(f"  Unique Triangles: {unique_labels}")
print(f"  Label Range:      {combined_labels.min()} - {combined_labels.max()}")

print("\n[FERTIG] Bereit für Training!")
