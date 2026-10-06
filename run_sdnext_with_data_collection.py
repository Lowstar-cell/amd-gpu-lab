#!/usr/bin/env python3
"""
Stable Diffusion Next - GPU Data Collection Runner

Startet SD.Next und generiert Bilder während GPU Metriken gesammelt werden.
Perfekt für GPU Performance Daten sammeln!

Usage:
    python run_sdnext_with_data_collection.py --images 10 --output sd_gpu_data
"""

import subprocess
import time
import json
import threading
import requests
from pathlib import Path
from datetime import datetime
import argparse


class SDNextGPUCollector:
    """Runs SD.Next and collects GPU metrics."""

    def __init__(self, sdnext_path, output_dir, interval=0.5):
        self.sdnext_path = Path(sdnext_path)
        self.output_dir = Path(output_dir)
        self.interval = interval
        self.collecting = False
        self.metrics = []
        self.sdnext_process = None

        # Create run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"sdnext_run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Image output directory
        self.img_dir = self.run_dir / "generated_images"
        self.img_dir.mkdir(exist_ok=True)

        print(f"📁 Data directory: {self.run_dir}")
        print(f"🖼️  Images will be saved to: {self.img_dir}")

    def collect_gpu_metrics(self):
        """Collect GPU metrics via rocm-smi."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showuse", "--showmeminfo", "vram",
                 "--showtemp", "--showpower", "--csv"],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    # Parse CSV
                    header = lines[0].split(',')
                    data = lines[1].split(',')

                    return {
                        'timestamp': time.time(),
                        'header': header,
                        'data': data,
                        'raw': result.stdout.strip()
                    }
        except Exception as e:
            pass

        return None

    def start_collection(self):
        """Start background GPU metrics collection."""
        self.collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        print("✓ GPU metrics collection started")

    def _collection_loop(self):
        """Background loop collecting metrics."""
        while self.collecting:
            metric = self.collect_gpu_metrics()
            if metric:
                self.metrics.append(metric)
            time.sleep(self.interval)

    def stop_collection(self):
        """Stop collection and save data."""
        self.collecting = False
        if hasattr(self, 'collection_thread'):
            self.collection_thread.join(timeout=2)

        # Save metrics
        if self.metrics:
            metrics_file = self.run_dir / "gpu_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            print(f"✓ Saved {len(self.metrics)} GPU metric snapshots")

    def start_sdnext(self, port=7860):
        """Start SD.Next in API mode."""
        print(f"\n🚀 Starting SD.Next on port {port}...")

        launch_script = self.sdnext_path / "webui.py"
        venv_python = self.sdnext_path / "venv" / "bin" / "python3"

        if not venv_python.exists():
            venv_python = "python3"

        cmd = [
            str(venv_python),
            str(launch_script),
            "--api",
            "--port", str(port),
            "--listen",
            "--skip-torch-cuda-test",  # We know ROCm works
            "--no-gradio-queue"
        ]

        # Start SD.Next
        self.sdnext_process = subprocess.Popen(
            cmd,
            cwd=str(self.sdnext_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print(f"✓ SD.Next starting (PID: {self.sdnext_process.pid})")

        # Wait for API to be ready
        api_url = f"http://localhost:{port}/sdapi/v1/sd-models"
        print(f"⏳ Waiting for API to be ready at {api_url}...")

        for attempt in range(60):  # 60 seconds timeout
            try:
                response = requests.get(api_url, timeout=2)
                if response.status_code == 200:
                    print(f"✓ SD.Next API ready!")
                    return True
            except:
                pass

            time.sleep(1)
            if attempt % 10 == 0 and attempt > 0:
                print(f"  Still waiting... ({attempt}s)")

        print("✗ Timeout waiting for SD.Next API")
        return False

    def stop_sdnext(self):
        """Stop SD.Next."""
        if self.sdnext_process:
            print("\n🛑 Stopping SD.Next...")
            self.sdnext_process.terminate()
            try:
                self.sdnext_process.wait(timeout=10)
            except:
                self.sdnext_process.kill()
            print("✓ SD.Next stopped")

    def generate_images(self, num_images=5, port=7860):
        """Generate images via SD.Next API."""
        api_url = f"http://localhost:{port}/sdapi/v1/txt2img"

        prompts = [
            "a beautiful landscape with mountains and lakes, photorealistic, 8k",
            "cyberpunk city at night, neon lights, futuristic, detailed",
            "portrait of a wise old wizard, fantasy art, intricate details",
            "abstract geometric patterns, colorful, modern art",
            "cute cat sitting on a windowsill, sunlight, cozy atmosphere",
            "space station orbiting a planet, sci-fi, epic scale",
            "medieval castle on a hilltop, dramatic clouds, fantasy",
            "underwater scene with coral reefs and fish, vibrant colors",
            "steampunk airship flying through clouds, detailed machinery",
            "enchanted forest with glowing mushrooms, magical atmosphere",
        ]

        results = []

        print(f"\n🎨 Generating {num_images} images...")

        for i in range(num_images):
            prompt = prompts[i % len(prompts)]
            print(f"\n[{i+1}/{num_images}] Generating: '{prompt[:50]}...'")

            payload = {
                "prompt": prompt,
                "negative_prompt": "blurry, low quality, ugly",
                "steps": 20,
                "width": 512,
                "height": 512,
                "cfg_scale": 7,
                "sampler_name": "Euler a",
            }

            start_time = time.time()

            try:
                response = requests.post(api_url, json=payload, timeout=300)

                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    data = response.json()

                    # Save image
                    if 'images' in data and len(data['images']) > 0:
                        import base64
                        img_data = base64.b64decode(data['images'][0])
                        img_path = self.img_dir / f"image_{i+1:03d}.png"

                        with open(img_path, 'wb') as f:
                            f.write(img_data)

                        print(f"  ✓ Generated in {elapsed:.2f}s")
                        print(f"  💾 Saved to: {img_path.name}")

                        results.append({
                            'image_id': i + 1,
                            'prompt': prompt,
                            'generation_time': elapsed,
                            'image_path': str(img_path),
                            'timestamp': time.time()
                        })
                    else:
                        print(f"  ✗ No image in response")
                else:
                    print(f"  ✗ API error: {response.status_code}")

            except Exception as e:
                print(f"  ✗ Error: {e}")

        # Save generation results
        results_file = self.run_dir / "generation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Saved generation results")

        return results


def main():
    parser = argparse.ArgumentParser(description='SD.Next GPU Data Collector')
    parser.add_argument('--sdnext', default='/home/user/sdnext',
                        help='Path to SD.Next installation')
    parser.add_argument('--images', type=int, default=10,
                        help='Number of images to generate')
    parser.add_argument('--output', default='sd_gpu_data',
                        help='Output directory for data')
    parser.add_argument('--interval', type=float, default=0.2,
                        help='GPU metric collection interval (seconds)')
    parser.add_argument('--port', type=int, default=7860,
                        help='SD.Next API port')
    args = parser.parse_args()

    print("=" * 70)
    print("STABLE DIFFUSION NEXT - GPU DATA COLLECTOR")
    print("=" * 70)
    print()

    # Initialize
    collector = SDNextGPUCollector(
        sdnext_path=args.sdnext,
        output_dir=args.output,
        interval=args.interval
    )

    try:
        # Start GPU metrics collection
        collector.start_collection()

        # Start SD.Next
        if not collector.start_sdnext(port=args.port):
            print("Failed to start SD.Next API")
            return

        # Generate images
        results = collector.generate_images(num_images=args.images, port=args.port)

        # Summary
        print("\n" + "=" * 70)
        print("✓ GENERATION COMPLETE!")
        print("=" * 70)
        print(f"Generated images: {len(results)}")
        print(f"GPU metrics collected: {len(collector.metrics)} snapshots")
        print(f"Data saved to: {collector.run_dir}")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")

    finally:
        # Cleanup
        collector.stop_sdnext()
        collector.stop_collection()


if __name__ == '__main__':
    main()
