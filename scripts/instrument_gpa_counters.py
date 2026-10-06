#!/usr/bin/env python3
import argparse
import subprocess
import os
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--libdir', default='data/4_2/bin')
    parser.add_argument('--include', default='data/4_2/include')
    parser.add_argument('--out', default='data/gpa_counters.csv')
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()

    tool = 'build/gpa_list_counters'
    if args.build or not os.path.exists(tool):
        subprocess.check_call(['python3', 'scripts/build_gpa_tool.py', '--include', args.include, '--lib', args.libdir, '--out', tool])

    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = os.path.abspath(args.libdir) + ':' + env.get('LD_LIBRARY_PATH', '')
    try:
        subprocess.check_call([tool, args.libdir, args.out], env=env)
    except subprocess.CalledProcessError:
        print('Failed to run gpa tool. Falling back to parsing HTML docs...')
        subprocess.check_call(['python3', 'scripts/parse_gpuperfapi_counters.py', '--in', 'data/4_2/help/api_functions', '--out', 'data/gpa_functions.json'])
        print('Using parse results: data/gpa_functions.json')
        return
    print('Counter CSV written to', args.out)

    # Also produce JSON
    rows = []
    with open(args.out, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            idx, name, group, desc = line.strip().split(',')
            rows.append({'index': int(idx), 'name': name.strip('"'), 'group': int(group), 'description': desc.strip('"')})
    with open(os.path.splitext(args.out)[0] + '.json', 'w') as f:
        json.dump(rows, f, indent=2)
    print('Counter JSON written to', os.path.splitext(args.out)[0] + '.json')

if __name__ == '__main__':
    main()
