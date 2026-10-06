#!/usr/bin/env python3
"""
Orchestrate patch/build/run/convert/train for a RenderPipelineShaders example.
This script operates in stages and supports a dry-run mode where build/run are skipped.
"""
import argparse
import os
import subprocess
import sys
import shutil
import tempfile


def run(cmd, cwd=None, env=None):
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd, cwd=cwd, env=env)


def find_executable(build_dir, target):
    # Look for an executable named target in build_dir tree
    for root, _, files in os.walk(build_dir):
        for f in files:
            if target in f:
                fp = os.path.join(root, f)
                # basic heuristic: executable file
                if os.path.isfile(fp):
                    return fp
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', default='data/RenderPipelineShaders-main/tests/gui/test_triangle_vk.cpp')
    parser.add_argument('--include', action='append', default=['data/RenderPipelineShaders-main/include'])
    parser.add_argument('--libgpa', default='data/4_2/bin')
    parser.add_argument('--build-dir', default='build/rps_build')
    parser.add_argument('--target', default='test_triangle_vk')
    parser.add_argument('--iterations', type=int, default=1, help='Number of times to run the instrumented binary')
    parser.add_argument('--collected-dir', default='data/rps_collected')
    parser.add_argument('--restore', action='store_true', help='Restore backups after running')
    parser.add_argument('--no-build', action='store_true')
    parser.add_argument('--no-run', action='store_true')
    parser.add_argument('--no-patch', action='store_true')
    parser.add_argument('--no-train', action='store_true')
    parser.add_argument('--train-args', default='--epochs 2 --batch-size 8 --device cpu')
    parser.add_argument('--dataset-out', default='data/rps_dataset/gpa_dataset.csv')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false', help='Disable numeric normalization and numeric dataset generation')
    parser.add_argument('--derived', default=None, help='Path to derived.json to use; if not provided and --discover-derived enabled the pipeline will generate one in the collected dir')
    parser.add_argument('--discover-derived', dest='discover_derived', action='store_true', help='Automatically discover derived counters in collected dir and emit derived.json')
    parser.add_argument('--no-discover-derived', dest='discover_derived', action='store_false', help='Disable automatic discovery of derived counters')
    parser.set_defaults(discover_derived=True)
    args = parser.parse_args()

    example = os.path.abspath(args.example)
    if not os.path.exists(example):
        print('Example source not found:', example)
        sys.exit(1)

    # 1) Patch
    if not args.no_patch:
        # Prefer AST patcher
        patched = False
        try:
            run([sys.executable, 'scripts/instrument_rps_example_ast.py', '--example', example, '--libgpa', args.libgpa] + [x for inc in args.include for x in ('--include', inc)])
            patched = True
        except subprocess.CalledProcessError:
            print('AST patcher failed or not available; falling back to regex patcher')
        if not patched:
            run([sys.executable, 'scripts/instrument_rps_example_for_linux.py', '--example', example])

    # 2) Build
    build_dir = os.path.abspath(args.build_dir)
    if not args.no_build and not args.dry_run:
        run([sys.executable, 'scripts/build_rps_example_linux.py', '--rps-root', 'data/RenderPipelineShaders-main', '--build-dir', build_dir, '--target', args.target])

    # 3) Run (multiple iterations)
    binary = find_executable(build_dir, args.target)
    if not binary:
        print('Warning: target executable not found in', build_dir)
    collected_csvs = []
    if not args.no_run and binary and not args.dry_run:
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = os.path.abspath(args.libgpa) + ':' + env.get('LD_LIBRARY_PATH', '')
        os.makedirs(args.collected_dir, exist_ok=True)
        collected_csvs = []
        for it in range(args.iterations):
            print(f"Run iteration {it+1}/{args.iterations}")
            run([binary], cwd=os.path.dirname(binary), env=env)
            # move generated CSV to collected dir with unique name
            src_csv = os.path.join(os.path.dirname(binary), 'gpa_rps_output.csv')
            if os.path.exists(src_csv):
                import time
                ts = int(time.time())
                dst = os.path.join(os.path.abspath(args.collected_dir), f'gpa_rps_output_{ts}_{it}.csv')
                shutil.move(src_csv, dst)
                collected_csvs.append(dst)
                # move meta file too
                src_meta = os.path.join(os.path.dirname(binary), 'gpa_rps_output.csv.meta.json')
                if os.path.exists(src_meta):
                    dst_meta = os.path.join(os.path.abspath(args.collected_dir), f'gpa_rps_output_{ts}_{it}.csv.meta.json')
                    shutil.move(src_meta, dst_meta)
                # don't convert immediately; collect files and process after run loop so we can compute stats
            else:
                print('gpa_rps_output.csv not found after run')
    else:
        print('Skipping run stage')

    # 4) Convert to dataset (create dir if needed)
    dataset_out = os.path.abspath(args.dataset_out)
    os.makedirs(os.path.dirname(dataset_out), exist_ok=True)
    # default CSV produced by patcher/modified example
    sample_csv = os.path.join(os.path.dirname(binary) if binary else '.', 'gpa_rps_output.csv')
    if not os.path.exists(sample_csv):
        print('No gpa_rps_output.csv found at', sample_csv)
        if args.dry_run:
            # create a fake sample CSV with sample data for test/dry run
            with open(sample_csv, 'w') as f:
                f.write('iter,counter_0,counter_1\n0,10,20\n1,15,25\n')
            # also write fake metadata JSON
            meta_path = sample_csv + '.meta.json'
            with open(meta_path, 'w') as m:
                m.write('{\n')
                m.write('  "device_id": 0,\n')
                m.write('  "counters": [\n')
                m.write('    {"index": 0, "name": "counter_0", "usage": "kGpaUsageTypeItems"},\n')
                m.write('    {"index": 1, "name": "counter_1", "usage": "kGpaUsageTypeItems"}\n')
                m.write('  ]\n}')
        else:
            print('Skipping conversion (sample CSV missing)')
    # Ensure any sample CSV at sample_csv is moved into the collected directory (covers pre-existing files)
    if os.path.exists(sample_csv):
            # move dry-run sample into collected_dir so we can compute stats across collected runs
            os.makedirs(args.collected_dir, exist_ok=True)
            import time
            ts = int(time.time())
            dst = os.path.join(os.path.abspath(args.collected_dir), f'gpa_rps_output_{ts}_dry.csv')
            shutil.move(sample_csv, dst)
            sample_meta = sample_csv + '.meta.json'
            if os.path.exists(sample_meta):
                shutil.move(sample_meta, dst + '.meta.json')
            collected_csvs = [dst]

    # If there are collected CSVs, compute stats and convert them to dataset (normalized numeric)
    if os.path.exists(args.collected_dir):
        collected_files = sorted([p for p in os.listdir(args.collected_dir) if p.endswith('.csv')])
    else:
        collected_files = []
    if collected_files:
        stats_path = os.path.join(os.path.abspath(args.collected_dir), 'stats.json')
        # prepare derived_path variable
        derived_path = None
        if args.normalize:
            run([sys.executable, 'scripts/compute_counter_stats.py', args.collected_dir, '--out', stats_path])
            # optionally run derived autodiscovery so convert receives --derived
            if args.derived and os.path.exists(args.derived):
                derived_path = args.derived
            elif args.discover_derived:
                # write discovered derived.json inside collected_dir
                derived_path = os.path.join(os.path.abspath(args.collected_dir), 'derived.json')
                run([sys.executable, 'scripts/discover_derived_counters.py', args.collected_dir, '--out', derived_path])
        # Convert all collected csvs
        for f in collected_files:
            csv_path = os.path.join(os.path.abspath(args.collected_dir), f)
            if args.normalize:
                cmd = [sys.executable, 'scripts/convert_counter_csv_to_dataset.py', '--in', csv_path, '--out', dataset_out, '--append', '--as-numeric', '--stats', stats_path]
                if derived_path and os.path.exists(derived_path):
                    cmd += ['--derived', derived_path]
                run(cmd)
            else:
                cmd = [sys.executable, 'scripts/convert_counter_csv_to_dataset.py', '--in', csv_path, '--out', dataset_out, '--append']
                if derived_path and os.path.exists(derived_path):
                    cmd += ['--derived', derived_path]
                run(cmd)

    # 5) Train
    if not args.no_train and not args.dry_run:
        dataset_dir = os.path.dirname(dataset_out)
        train_cmd = [sys.executable, 'train.py', '--data-dir', dataset_dir] + args.train_args.split()
        run(train_cmd)
    else:
        print('Skipping train stage')

    # Restore backups if requested
    if args.restore:
        bak = example + '.bak'
        if os.path.exists(bak):
            print('Restoring original source from', bak)
            shutil.move(bak, example)
        else:
            print('No backup found to restore')

    print('Pipeline complete. Dataset at:', dataset_out)


if __name__ == '__main__':
    raise SystemExit(main())
