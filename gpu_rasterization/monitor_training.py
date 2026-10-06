#!/usr/bin/env python3
"""Monitor training checkpoints and produce an append-only summary and artifacts archive.

Usage:
  python monitor_training.py --checkpoint-dir gpu_runs/long --summary-file gpu_runs/long/training_summary.txt --artifact-dir gpu_runs/long/artifacts --interval 30

Optional: set UPLOAD_DEST env var (scp target like user@host:/path/) to upload artifacts when created.
"""

import argparse
import os
import time
import hashlib
import shutil
import subprocess
from datetime import datetime

try:
    import torch
except Exception:
    torch = None


def md5_of_file(path, block_size=2**20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(block_size), b''):
            h.update(b)
    return h.hexdigest()


def run_rocm_smi():
    try:
        out = subprocess.check_output(['/usr/bin/rocm-smi', '-a'], stderr=subprocess.DEVNULL, text=True)
        return out.splitlines()
    except Exception:
        return []


def upload_file(local_path, upload_dest):
    # upload_dest should be something like user@host:/path/
    try:
        subprocess.check_call(['scp', local_path, upload_dest])
        return True
    except Exception as e:
        print('Upload failed:', e)
        return False


def parse_epoch_from_name(name):
    # tries to find "epoch_#" or "_epoch_#" in the filename
    import re
    m = re.search(r'epoch[_-]?(\d+)', name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def inspect_checkpoint(path):
    info = {'path': path}
    try:
        st = os.stat(path)
        info['size_bytes'] = st.st_size
        info['mtime'] = datetime.fromtimestamp(st.st_mtime).isoformat()
        info['md5'] = md5_of_file(path)
        info['epoch_name'] = parse_epoch_from_name(os.path.basename(path))
        if torch is not None:
            try:
                d = torch.load(path, map_location='cpu')
                if isinstance(d, dict):
                    if 'epoch' in d:
                        info['epoch'] = int(d['epoch'])
                    if 'best_val_loss' in d:
                        info['best_val_loss'] = d.get('best_val_loss')
                else:
                    info['epoch'] = info.get('epoch_name')
            except Exception:
                # skip heavy or failing loads
                pass
    except Exception as e:
        info['error'] = str(e)
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--summary-file', required=True)
    parser.add_argument('--artifact-dir', required=True)
    parser.add_argument('--interval', type=int, default=30)
    args = parser.parse_args()

    seen = set()
    os.makedirs(args.artifact_dir, exist_ok=True)
    # write a header to summary file if it doesn't exist
    if not os.path.exists(args.summary_file):
        with open(args.summary_file, 'a') as f:
            f.write('# Training summary started: ' + datetime.now().isoformat() + '\n')

    upload_dest = os.environ.get('UPLOAD_DEST')
    print('Monitor started. Watching:', args.checkpoint_dir)
    print('Summary file:', args.summary_file)
    if upload_dest:
        print('Upload destination configured:', upload_dest)

    # initial rocm-smi snapshot
    lines = run_rocm_smi()
    with open(args.summary_file, 'a') as f:
        f.write('\n# ROCm-SMI snapshot at ' + datetime.now().isoformat() + '\n')
        for ln in lines:
            f.write('# ' + ln + '\n')

    while True:
        try:
            files = sorted([os.path.join(args.checkpoint_dir, p) for p in os.listdir(args.checkpoint_dir)], key=lambda p: os.path.getmtime(p))
        except Exception:
            files = []
        new = [p for p in files if p not in seen and os.path.isfile(p)]
        for p in new:
            info = inspect_checkpoint(p)
            entry = {
                'time': datetime.now().isoformat(),
                'file': os.path.relpath(p),
                'size_MB': round(info.get('size_bytes', 0) / 1024.0 / 1024.0, 3),
                'md5': info.get('md5'),
                'epoch_field': info.get('epoch') or info.get('epoch_name') or 'unknown',
                'details': {}
            }
            if 'best_val_loss' in info:
                entry['details']['best_val_loss'] = info['best_val_loss']

            # append to summary
            with open(args.summary_file, 'a') as f:
                f.write(f"[{entry['time']}] CHECKPOINT: {entry['file']} size={entry['size_MB']}MB epoch={entry['epoch_field']} md5={entry['md5']}\n")

            # create artifact tar for this checkpoint
            base = os.path.basename(p)
            tarname = os.path.join(args.artifact_dir, base + '.tar.gz')
            try:
                shutil.make_archive(tarname[:-7], 'gztar', root_dir=os.path.dirname(p), base_dir=base)
                with open(args.summary_file, 'a') as f:
                    f.write(f"[{datetime.now().isoformat()}] ARCHIVE: {tarname}\n")
                # optionally upload
                if upload_dest:
                    ok = upload_file(tarname, upload_dest)
                    with open(args.summary_file, 'a') as f:
                        f.write(f"[{datetime.now().isoformat()}] UPLOAD: {'OK' if ok else 'FAILED'} {upload_dest}\n")
            except Exception as e:
                with open(args.summary_file, 'a') as f:
                    f.write(f"[{datetime.now().isoformat()}] ARCHIVE FAILED: {e}\n")

            seen.add(p)

        time.sleep(args.interval)


if __name__ == '__main__':
    main()
