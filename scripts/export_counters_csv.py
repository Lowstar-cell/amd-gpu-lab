#!/usr/bin/env python3
import json
import csv
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', default='data/gpa_functions.json')
    parser.add_argument('--out', default='data/gpa_functions.csv')
    args = parser.parse_args()
    with open(args.infile, 'r') as f:
        data = json.load(f)
    with open(args.out, 'w', newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['name', 'context', 'file'])
        for d in data:
            writer.writerow([d.get('func', ''), d.get('context', ''), d.get('file', '')])
    print('Wrote', args.out)

if __name__ == '__main__':
    main()
