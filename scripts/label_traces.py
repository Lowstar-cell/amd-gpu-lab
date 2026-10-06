#!/usr/bin/env python3
"""
Label traces or numeric dataset rows by sample id or by a range.

This small helper updates the 'label' column in a numeric CSV file produced by
`convert_counter_csv_to_dataset.py`.

Usage examples:
  # set sample_id 0 and 2 to label 1
  python3 scripts/label_traces.py --in out.csv.numeric.csv --set-label 1 --sample-ids 0,2 --out labeled.csv

  # set label for a range of sample ids
  python3 scripts/label_traces.py --in out.csv.numeric.csv --set-label 2 --sample-range 10-20 --inplace

The script updates or writes a new CSV. It expects a header containing a 'sample_id' column and a 'label' column.
"""
import argparse
import csv
import os
import sys


def parse_sample_ids(s: str):
    # support comma separated sample ids
    parts = [p.strip() for p in s.split(',') if p.strip()]
    ids = set()
    for p in parts:
        if '-' in p:
            a, b = p.split('-', 1)
            try:
                a_i = int(a)
                b_i = int(b)
                for i in range(a_i, b_i + 1):
                    ids.add(str(i))
            except Exception:
                continue
        else:
            ids.add(str(int(p)))
    return ids


def update_labels(in_csv, out_csv=None, label=1, sample_ids=None, inplace=False, sample_range=None):
    if inplace and out_csv is not None:
        raise ValueError('cannot specify --out when --inplace is used')

    out_path = in_csv if inplace else (out_csv or in_csv + '.labeled.csv')

    if not os.path.exists(in_csv):
        raise FileNotFoundError(in_csv)

    with open(in_csv, 'r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
        if not reader:
            raise ValueError('empty CSV: ' + in_csv)
        header = reader[0].keys()
        if 'sample_id' not in header:
            raise ValueError('CSV missing sample_id column')
        if 'label' not in header:
            # try to append a label column if missing
            add_label = True
        else:
            add_label = False

    # normalize sample_ids and sample_range to sets of strings
    target_ids = set()
    if sample_ids:
        target_ids |= set(parse_sample_ids(sample_ids))
    if sample_range:
        a, b = sample_range.split('-', 1)
        try:
            a_i = int(a); b_i = int(b)
            for i in range(a_i, b_i + 1):
                target_ids.add(str(i))
        except Exception:
            pass

    # process and write out
    with open(in_csv, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        # prepare fieldnames
        fieldnames = list(reader.fieldnames)
        if 'label' not in fieldnames:
            fieldnames.append('label')

        with open(out_path, 'w', encoding='utf-8', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                sid = str(row.get('sample_id', ''))
                if sid in target_ids:
                    row['label'] = str(label)
                elif add_label and 'label' not in row:
                    row['label'] = row.get('label', '0')
                writer.writerow(row)

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--out', default=None)
    parser.add_argument('--inplace', action='store_true', help='Edit file in place')
    parser.add_argument('--set-label', type=int, default=1, help='Label value to set')
    parser.add_argument('--sample-ids', default=None, help='Comma separated sample ids or ranges (e.g. 0,1,5-8)')
    parser.add_argument('--sample-range', default=None, help='A single range like 10-20')
    args = parser.parse_args()

    try:
        out = update_labels(args.infile, out_csv=args.out, label=args.set_label, sample_ids=args.sample_ids, inplace=args.inplace, sample_range=args.sample_range)
        print('Wrote labeled CSV to', out)
    except Exception as ex:
        print('ERROR:', ex, file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
