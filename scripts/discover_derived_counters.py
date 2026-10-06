#!/usr/bin/env python3
"""
Discover derived counter expressions inside an SDK or a directory of metadata files.

This tool heuristically scans JSON, text and C/C++ header files to find simple derived
counter definitions and outputs a `derived.json` file that can be passed to the
converter (convert_counter_csv_to_dataset.py --derived derived.json).

The heuristic looks for patterns like:
  - JSON with top-level 'derived' key
  - lines containing 'derived' or 'Derived' followed by NAME = EXPR
  - simple assignment expressions NAME = token / token

The output format is:
  { "derived": [ {"name": "c_ratio", "expr": "counter_0 / counter_1"}, ... ] }

This is intentionally conservative — it only extracts simple arithmetic expressions
using [A-Za-z0-9_()+\-*/ ] tokens.
"""
import argparse
import os
import json
import re


SIMPLE_EXPR_RX = re.compile(r"^[0-9A-Za-z_()\s+\-\*/]+$")
ASSIGN_RX = re.compile(r"(?P<name>[A-Za-z0-9_]+)\s*[:=]\s*(?P<expr>[^;#]*)")


def find_in_text(text):
    results = []
    for line in text.splitlines():
        ll = line.strip()
        # look for derived declarations like 'Derived ... name = expr' or 'name = expr'
        if 'derived' in ll.lower() or '=' in ll:
            m = ASSIGN_RX.search(ll)
            if m:
                name = m.group('name')
                expr = m.group('expr').strip()
                # only allow safe token characters
                if SIMPLE_EXPR_RX.match(expr):
                    results.append({'name': name, 'expr': expr})
    return results


def scan_dir(root):
    derived = []
    visited = set()
    for dirpath, dirs, files in os.walk(root):
        for fn in files:
            path = os.path.join(dirpath, fn)
            if path in visited:
                continue
            visited.add(path)
            lname = fn.lower()
            try:
                if lname.endswith('.json'):
                    with open(path, 'r', encoding='utf-8') as jf:
                        try:
                            j = json.load(jf)
                        except Exception:
                            j = None
                        if isinstance(j, dict) and 'derived' in j:
                            for d in j.get('derived', []):
                                if isinstance(d, dict) and 'name' in d and 'expr' in d:
                                    derived.append({'name': d['name'], 'expr': d['expr']})
                        else:
                            # search for keys like formula, expr
                            for k, v in (j.items() if isinstance(j, dict) else []):
                                if isinstance(v, str) and '=' in v:
                                    res = find_in_text(v)
                                    derived.extend(res)
                elif lname.endswith(('.txt', '.md', '.h', '.hpp', '.c', '.cpp', '.rst.txt')):
                    with open(path, 'r', encoding='utf-8', errors='ignore') as tf:
                        txt = tf.read()
                        # For RST sources that contain counter tables we can parse CSV-like lines
                        if lname.endswith('.rst.txt'):
                            # lines like: "Name", "SampleType", "Usage", "Description"
                            for line in txt.splitlines():
                                parts = [p.strip().strip('"') for p in line.split(',') if p.strip().startswith('"')]
                                if len(parts) >= 3:
                                    name = parts[0]
                                    usage = parts[2]
                                    desc = parts[3] if len(parts) > 3 else ''
                                    # store in derived list if usage indicates percentage/ratio
                                    if 'percent' in usage.lower() or 'ratio' in usage.lower():
                                        # simple heuristic: look for component counters like Hits/Requests/Misses
                                        # we will return an entry that uses component names (best-effort)
                                        # Defer expression resolution to a later step; here just capture candidate
                                        derived.append({'name': name, 'expr': '' , 'meta_usage': usage, 'desc': desc})
                                # also fall back to free-text heuristic
                            found = find_in_text(txt)
                            derived.extend(found)
                        else:
                            found = find_in_text(txt)
                            derived.extend(found)
            except Exception:
                # ignore unreadable files
                continue
    # Build map of all counters found (name -> info)
    counters_map = {}
    for d in derived:
        if isinstance(d, dict) and 'name' in d:
            counters_map[d['name']] = d

    # Heuristic: for counters with usage containing 'percent' or 'ratio' try
    # to find component counters (Hits/Requests/Misses) and construct expressions
    results = {}
    for name, info in counters_map.items():
        expr = info.get('expr', '')
        usage = (info.get('meta_usage') or '').lower()
        desc = info.get('desc', '')
        if expr:
            results[name] = expr
            continue
        if 'percent' in usage or 'ratio' in usage or 'percent' in desc.lower() or 'ratio' in desc.lower():
            # attempt to find components
            # compute base by stripping common suffixes
            base = name
            for suf in ['Percent', 'PercentValue', 'Percentage', 'HitPercent', 'Ratio', 'Utilization', 'Busy']:
                if base.endswith(suf):
                    base = base[: -len(suf)]
            base = base.strip('_ ') or name

            # find possible component counters
            hits = None
            requests = None
            misses = None
            for other in counters_map.keys():
                if other == name:
                    continue
                low = other.lower()
                if base.lower() in low and ('hit' in low and hits is None):
                    hits = other
                if base.lower() in low and ('request' in low or 'req' in low or 'transaction' in low) and requests is None:
                    requests = other
                if base.lower() in low and ('miss' in low) and misses is None:
                    misses = other

            if hits and requests:
                # percent = hits / requests
                results[name] = f"{hits} / {requests}"
            elif hits and misses:
                # percent = hits / (hits + misses)
                results[name] = f"{hits} / ({hits} + {misses})"
            else:
                # fallback: if description mentions 'hits' or 'requests', try naive splits
                # don't create an expr if we can't find plausible components
                continue
        else:
            # if we parsed an explicit expr or simple assignment earlier using find_in_text, keep it
            if expr:
                results[name] = expr

    return [{'name': n, 'expr': e} for n, e in results.items()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scan_dir', nargs='?', default='data', help='Directory to scan (default=data)')
    parser.add_argument('--out', default='derived.json', help='Output derived.json path')
    args = parser.parse_args()

    if not os.path.exists(args.scan_dir):
        print('directory does not exist:', args.scan_dir)
        raise SystemExit(1)

    derived = scan_dir(args.scan_dir)
    out = {'derived': derived}
    with open(args.out, 'w', encoding='utf-8') as of:
        json.dump(out, of, indent=2)
    print('Wrote derived file with %d discovered entries to %s' % (len(derived), args.out))


if __name__ == '__main__':
    main()
