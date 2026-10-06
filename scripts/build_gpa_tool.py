#!/usr/bin/env python3
import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include', default='data/4_2/include')
    parser.add_argument('--lib', default='data/4_2/bin')
    parser.add_argument('--out', default='build/gpa_list_counters')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cmd = [
        'g++', '-std=c++17', 'tools/gpa_list_counters.cpp', 'tools/gpa_loader_helper.cpp',
        '-I', args.include,
        '-L', args.lib, '-lGPUPerfAPIGL', '-lGPUPerfAPICounters', '-lGPUPerfAPIVK', '-ldl', '-lX11',
        '-o', args.out
    ]
    print('Compiling: ' + ' '.join(cmd))
    subprocess.check_call(cmd)
    print('Built', args.out)

if __name__ == '__main__':
    main()
