#!/usr/bin/env python3
"""
Run a collection job across multiple RenderPipelineShaders samples/tests.

This script scans a set of candidate source directories for files that likely
contain _vk.cpp test sources and runs the existing pipeline script on each.

It's intended for automation and supports dry-run mode so it can be used in CI
without GPU/library dependencies.
"""
import argparse
import os
import subprocess
import sys
import glob


def run(cmd):
    print('>',' '.join(cmd))
    subprocess.check_call(cmd)


def find_vk_sources(root_dir):
    # Look under the tests/gui and tests/console folders for *_vk.cpp and *_vk_dyn.cpp
    patterns = [os.path.join(root_dir, 'tests', '**', '*_vk.cpp'), os.path.join(root_dir, 'tests', '**', '*_vk_dyn.cpp')]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p, recursive=True))
    # deduplicate
    seen = set()
    out = []
    for p in sorted(paths):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rps-root', default='data/RenderPipelineShaders-main', help='Root of the RPS SDK checkout')
    parser.add_argument('--examples', nargs='*', help='Explicit list of example/test files to patch/run (overrides scan)')
    parser.add_argument('--build-dir', default='build/rps_build', help='CMake build directory')
    parser.add_argument('--target', default='test_triangle_vk', help='Default build target')
    parser.add_argument('--libgpa', default='data/4_2/bin', help='Path to GPA libs')
    parser.add_argument('--dry-run', action='store_true', help='Dry-run; do not build or run instruments. Still patches and converts with fake data')
    parser.add_argument('--iterations', type=int, default=1, help='Number of runs to collect per example')
    parser.add_argument('--examples-pattern', default=None, help='Optional glob pattern (relative to rps-root) to filter discovered files')
    args = parser.parse_args()

    root = os.path.abspath(args.rps_root)
    if args.examples:
        candidates = [os.path.abspath(e) for e in args.examples]
    else:
        candidates = find_vk_sources(root)
        if args.examples_pattern:
            candidates = [c for c in candidates if args.examples_pattern in c]

    if not candidates:
        print('No candidate VK sources found under', root)
        return 1

    print('Found', len(candidates), 'candidates; first few:')
    for c in candidates[:5]:
        print('  -', c)

    for src in candidates:
        print('\n=== Processing', src, '===')
        cmd = [sys.executable, 'scripts/run_instrumented_rps_pipeline.py', '--example', src, '--build-dir', args.build_dir, '--target', args.target, '--libgpa', args.libgpa]
        if args.dry_run:
            cmd += ['--no-build', '--no-run', '--dry-run']
        cmd += ['--iterations', str(args.iterations), '--no-train']
        # keep backups around (pipeline supports --restore) — we do NOT auto-restore here
        run(cmd)

    print('\nBatch collection finished for', len(candidates), 'sources')


if __name__ == '__main__':
    raise SystemExit(main())
