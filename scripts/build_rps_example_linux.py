#!/usr/bin/env python3
"""
Build a RenderPipelineShaders example or test on Linux using CMake.
"""
import argparse
import os
import subprocess
import sys


def run(cmd, cwd=None):
    print("Running:", ' '.join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rps-root', default='data/RenderPipelineShaders-main')
    parser.add_argument('--build-dir', default='build/rps_build')
    parser.add_argument('--target', default='test_triangle_vk')
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--cmake-args', default='', help='Extra CMake args to pass')
    args = parser.parse_args()

    rps_root = os.path.abspath(args.rps_root)
    build_dir = os.path.abspath(args.build_dir)
    os.makedirs(build_dir, exist_ok=True)

    cmake_cmd = ['cmake', rps_root, '-B', build_dir, '-DRpsEnableVulkan=ON']
    if args.cmake_args:
        cmake_cmd.extend(args.cmake_args.split())

    run(cmake_cmd)
    run(['cmake', '--build', build_dir, '--target', args.target, '--', '-j'], cwd=build_dir)
    if args.install:
        run(['cmake', '--install', build_dir])

    print('Built target', args.target)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

