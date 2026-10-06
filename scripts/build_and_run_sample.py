#!/usr/bin/env python3
import os
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include', default='data/4_2/include')
    parser.add_argument('--lib', default='data/4_2/bin')
    parser.add_argument('--out', default='build/gpa_instrument_sample')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--vulkan', action='store_true', help='Build a Vulkan sample instead of OpenGL')
    parser.add_argument('--libpath', default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Check for system dependencies (GLFW)
    # For Vulkan, check for vulkan pkg
    if args.vulkan:
        try:
            subprocess.check_call(['pkg-config', '--exists', 'vulkan'])
        except Exception:
            print('Vulkan dev package not found via pkg-config. Please install libvulkan-dev.')
            raise
    else:
        try:
            subprocess.check_call(['pkg-config', '--exists', 'glfw3'])
        except Exception:
            print('GLFW3 not found via pkg-config. Please install system package (e.g. libglfw3-dev on Debian/Ubuntu)')
            raise

    if args.vulkan:
        # Build Vulkan sample
        cmd = [
            'g++', '-std=c++17', 'tools/gpa_instrument_sample_vk.cpp', 'tools/gpa_loader_helper.cpp',
            '-I', args.include,
            '-L', args.lib, '-lGPUPerfAPIVK', '-lGPUPerfAPICounters', '-ldl', '-lvulkan',
            '-o', args.out
        ]
    else:
        cmd = [
            'g++', '-std=c++17', 'tools/gpa_instrument_sample.cpp', 'tools/gpa_loader_helper.cpp',
            '-I', args.include,
            '-L', args.lib, '-lGPUPerfAPIGL', '-lGPUPerfAPICounters', '-lGPUPerfAPIVK', '-ldl', '-lX11', '-lglfw', '-lGL',
            '-o', args.out
        ]
    print('Compiling: ' + ' '.join(cmd))
    subprocess.check_call(cmd)
    print('Built', args.out)
    if args.run:
        env = os.environ.copy()
        if args.libpath:
            env['LD_LIBRARY_PATH'] = os.path.abspath(args.libpath) + ':' + env.get('LD_LIBRARY_PATH', '')
        elif args.lib:
            env['LD_LIBRARY_PATH'] = os.path.abspath(args.lib) + ':' + env.get('LD_LIBRARY_PATH', '')
        if args.vulkan:
            subprocess.check_call([args.out, args.lib, 'data/gpa_sample_vk.csv', '10'], env=env)
            subprocess.check_call([sys.executable, 'scripts/convert_counter_csv_to_dataset.py', '--in', 'data/gpa_sample_vk.csv', '--out', 'data/gpa_dataset_vk.csv'])
        else:
            subprocess.check_call([args.out, args.lib, 'data/gpa_sample.csv', '10'], env=env)
            subprocess.check_call([sys.executable, 'scripts/convert_counter_csv_to_dataset.py', '--in', 'data/gpa_sample.csv', '--out', 'data/gpa_dataset.csv'])
    # Convert to dataset format
    subprocess.check_call([sys.executable, 'scripts/convert_counter_csv_to_dataset.py', '--in', 'data/gpa_sample.csv', '--out', 'data/gpa_dataset.csv'])

if __name__ == '__main__':
    main()
