#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--include', default='data/4_2/include')
    parser.add_argument('--lib', default='data/4_2/bin')
    parser.add_argument('--out-dir', default='data/gpa_runs')
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(args.runs):
        csv_out = os.path.join(args.out_dir, f'gpa_sample_{i}.csv')
        dataset_out = os.path.join(args.out_dir, f'gpa_dataset_{i}.csv')
        cmd = [sys.executable, 'scripts/build_and_run_sample.py', '--include', args.include, '--lib', args.lib, '--out', 'build/gpa_instrument_sample']
        if args.build:
            cmd.append('--run')
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError:
            print('Build or run failed for run', i)
            continue
        # run the instrumented sample to produce sample CSV
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = os.path.abspath(args.lib) + ':' + env.get('LD_LIBRARY_PATH', '')
        try:
            subprocess.check_call(['build/gpa_instrument_sample', args.lib, csv_out, str(args.iterations)], env=env)
            # convert to dataset
            subprocess.check_call([sys.executable, 'scripts/convert_counter_csv_to_dataset.py', '--in', csv_out, '--out', dataset_out])
            print('Run', i, 'wrote dataset to', dataset_out)
        except subprocess.CalledProcessError:
            print('Failed to run instrumented sample for run', i)

if __name__ == '__main__':
    main()
