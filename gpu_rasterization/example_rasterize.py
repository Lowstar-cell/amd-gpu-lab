"""
AMD GPU Rasterization - Komplettes Beispiel
Zeigt alle Features: HIP/ROCm, Datensammlung, Embedding Training
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from gpu_rasterize import AMDRasterizer, RasterizationEmbedding
from data_collection.collector import RasterizationDataCollector
import time

def create_test_scene(num_triangles: int = 100):
    """Erstelle zufällige 3D Scene"""
    vertices = []
    colors = []

    for _ in range(num_triangles):
        # Random triangle in normalized device coordinates
        tri = np.random.rand(3, 3).astype(np.float32)
        tri[:, :2] = tri[:, :2] * 512  # Screen space

        # Random depth
        tri[:, 2] = np.random.rand() * 100

        vertices.append(tri)

        # Random color
        color = np.random.rand(3).astype(np.float32)
        colors.append(color)

    return np.array(vertices), np.array(colors)


def main():
    print("=" * 60)
    print("AMD GPU 3D RASTERIZER - FULL DEMO")
    print("=" * 60)

    # 1. Initialize Rasterizer
    width, height = 512, 512
    rasterizer = AMDRasterizer(width=width, height=height, backend='auto')

    # 2. Create scene
    print("\n[1] Creating test scene...")
    num_triangles = 50
    vertices, colors = create_test_scene(num_triangles)
    print(f"    Scene: {num_triangles} triangles")

    # 3. Setup MVP matrix
    mvp = np.eye(4, dtype=np.float32)
    mvp[0, 0] = 2.0 / width
    mvp[1, 1] = 2.0 / height
    mvp[0, 3] = -1.0
    mvp[1, 3] = -1.0

    # 4. Rasterize with data collection
    print("\n[2] Rasterizing on AMD GPU...")
    start = time.time()

    color_img, training_data = rasterizer.rasterize_triangles(
        vertices,
        colors,
        mvp,
        collect_data=True
    )

    elapsed = time.time() - start
    print(f"    Rasterization time: {elapsed*1000:.2f} ms")

    # 5. Data Collection
    if training_data:
        print("\n[3] Collecting training data...")
        collector = RasterizationDataCollector(save_dir='./data')

        collector.collect_from_rasterization(
            training_data['barycentric_coords'],
            training_data['depth'],
            training_data['coverage']
        )

        dataset_path = collector.save_dataset('demo_data.h5')
        print(f"    Saved to: {dataset_path}")

        # 6. Train embedding model
        print("\n[4] Training embedding model...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"    Training device: {device}")

        embedding_model = RasterizationEmbedding(input_dim=7, embedding_dim=128)
        embedding_model = embedding_model.to(device)

        # Get dataset
        dataset = collector.get_pytorch_dataset()
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=256,
            shuffle=True
        )

        # Simple training loop
        optimizer = torch.optim.Adam(embedding_model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        # Classification head für Demo
        classifier = torch.nn.Linear(128, num_triangles).to(device)

        for epoch in range(5):
            total_loss = 0
            for features, labels in dataloader:
                features = features.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Embedding
                embeddings = embedding_model(features)

                # Classify which triangle
                logits = classifier(embeddings)

                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"    Epoch {epoch+1}/5 - Loss: {avg_loss:.4f}")

        print(f"    [OK] Model trained!")

    # 7. Visualize
    print("\n[5] Visualizing results...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Color output
    axes[0, 0].imshow(color_img)
    axes[0, 0].set_title('Rasterized Image')
    axes[0, 0].axis('off')

    # Depth map
    if training_data:
        depth = training_data['depth']
        depth_vis = np.clip(depth, 0, 100) / 100
        axes[0, 1].imshow(depth_vis, cmap='viridis')
        axes[0, 1].set_title('Depth Buffer')
        axes[0, 1].axis('off')

        # Coverage map
        coverage = training_data['coverage'].astype(float)
        axes[1, 0].imshow(coverage, cmap='gray')
        axes[1, 0].set_title('Coverage Map')
        axes[1, 0].axis('off')

        # Barycentric visualization (erste Triangle)
        bary = training_data['barycentric_coords'][:, :, 0, :]
        bary_vis = np.clip(bary, 0, 1)
        axes[1, 1].imshow(bary_vis)
        axes[1, 1].set_title('Barycentric Coords (Triangle 0)')
        axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('rasterizer_output.png', dpi=150)
    print(f"    Saved visualization to: rasterizer_output.png")

    # 8. Performance stats
    print("\n[6] Performance Statistics:")
    total_pixels = width * height
    rendered_pixels = (color_img.sum(axis=2) > 0).sum()
    fill_rate = rendered_pixels / total_pixels * 100

    print(f"    Resolution: {width}x{height} ({total_pixels:,} pixels)")
    print(f"    Rendered pixels: {rendered_pixels:,} ({fill_rate:.1f}%)")
    print(f"    Triangles: {num_triangles}")
    print(f"    Throughput: {num_triangles / elapsed:.0f} triangles/sec")

    if training_data:
        samples = len(collector.features)
        print(f"    Training samples: {samples:,}")

    print("\n" + "=" * 60)
    print("SUCCESS! AMD GPU Rasterizer is working!")
    print("=" * 60)


if __name__ == '__main__':
    main()
