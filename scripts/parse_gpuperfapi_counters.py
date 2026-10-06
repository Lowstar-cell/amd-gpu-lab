#!/usr/bin/env python3
"""
Parse GPUPerfAPI HTML docs to discover counter and API function names.
Outputs a JSON and CSV with parsed function names and descriptions.

Usage: python scripts/parse_gpuperfapi_counters.py --in data/4_2/help/api_functions --out counters.json
"""
import argparse
import glob
import json
import os
from html.parser import HTMLParser


class SimpleHTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
    def handle_data(self, d):
        self.text.append(d)
    def get_text(self):
        return ' '.join(self.text).strip()


def parse_html_file(path):
    # Collect function names and short description heuristically
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        html = f.read()
    ex = SimpleHTMLTextExtractor()
    ex.feed(html)
    txt = ex.get_text()
    # Heuristic: function names starting with 'gpa_' or 'GPA'
    results = []
    for line in txt.split('\n'):
        if 'gpa_' in line or 'GPA' in line.upper():
            # collect words with gpa_
            for token in line.split():
                if token.startswith('gpa_') or token.startswith('GPA'):
                    results.append({'func': token, 'context': line.strip()})
    return results


def parse_dir(dirpath):
    out = []
    for p in glob.glob(os.path.join(dirpath, '*.html')):
        name = os.path.basename(p)
        parsed = parse_html_file(p)
        for r in parsed:
            r['file'] = name
            out.append(r)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='indir', default='data/4_2/help/api_functions')
    parser.add_argument('--out', default='data/gpa_functions.json')
    args = parser.parse_args()

    parsed = parse_dir(args.indir)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(parsed, f, indent=2)
    print(f'Wrote {len(parsed)} entries to {args.out}')


if __name__ == '__main__':
    main()
