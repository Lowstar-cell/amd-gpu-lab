#!/usr/bin/env python3
"""
Build RenderPipelineShaders examples for Windows with optional GPA instrumentation flag.

Usage (Windows/Visual Studio):
  python scripts/build_rps_example_win.py --build-dir build/win --config Release --generator "Visual Studio 16 2019"

If you previously patched a sample using `scripts/instrument_rps_example_for_windows.py`, set `--enable-gpa` to pass a define to the sample build.
This script is intended for use on Windows (or with a Windows build environment).
"""
import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data/RenderPipelineShaders-main')
    parser.add_argument('--build-dir', default='data/RenderPipelineShaders-main/build')
    parser.add_argument('--config', default='Release')
    parser.add_argument('--generator', default='Visual Studio 16 2019')
    parser.add_argument('--enable-gpa', action='store_true')
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    build_dir = os.path.abspath(args.build_dir)
    os.makedirs(build_dir, exist_ok=True)

    cmake_cmd = ['cmake', '-S', root, '-B', build_dir, '-G', args.generator]
    if args.enable_gpa:
        cmake_cmd += ['-DENABLE_GPA=1']
    print('Running:', ' '.join(cmake_cmd))
    subprocess.check_call(cmake_cmd)

    build_cmd = ['cmake', '--build', build_dir, '--config', args.config]
    print('Building', ' '.join(build_cmd))
    subprocess.check_call(build_cmd)

if __name__ == '__main__':
    main()
