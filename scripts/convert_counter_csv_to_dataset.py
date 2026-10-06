#!/usr/bin/env python3
import csv
import json
import argparse
import os
import ast
import math

def quantize_value(v):
    # Simple quantization into small buckets for event_type mapping
    if v == 0:
        return '0'
    # handle float or int
    try:
        vi = int(v)
    except Exception:
        try:
            vi = int(float(v))
        except Exception:
            vi = 0
    if vi < 100:
        return 'small'
    if vi < 1000:
        return 'medium'
    return 'large'

def convert(in_csv, out_csv, label=0, as_numeric=False, stats=None, append=False, norm_method='zscore', derived_cfg=None):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(in_csv, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        rows_out = []
        numeric_rows = []
        for r in reader:
            sample_id = r.get('iter', '0')
            # Build one event per counter: event_type: CNT_<name>_<quant>
            timestamp = 0
            row_numeric = {'sample_id': sample_id}
            for k, v in r.items():
                if k == 'iter':
                    continue
                # if --as-numeric mode will use numeric values; still create token events for backward compatibility
                token = f"CNT_{k}_{quantize_value(v)}"
                msg = json.dumps({k: v})
                rows_out.append({'sample_id': sample_id, 'timestamp': timestamp, 'event_type': token, 'severity': 0, 'message': msg, 'label': label})
                timestamp += 1
                # numeric attempt (apply usage-type mapping if meta present)
                try:
                    val = float(v)
                except Exception:
                    val = None
                row_numeric[k] = val
            numeric_rows.append(row_numeric)
    # write tokenized rows (existing behaviour)
    write_header = True
    if os.path.exists(out_csv):
        write_header = False
    with open(out_csv, 'a' if os.path.exists(out_csv) else 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        if write_header:
            writer.writerow(['sample_id','timestamp','event_type','severity','message','label'])
        for r in rows_out:
            writer.writerow([r['sample_id'], r['timestamp'], r['event_type'], r['severity'], r['message'], r['label']])
    print('Wrote dataset CSV to', out_csv)

    # write numeric rows if requested
    if as_numeric and numeric_rows:
        numeric_out = out_csv + '.numeric.csv'
        write_header_num = not os.path.exists(numeric_out) or not append
        with open(numeric_out, 'a' if os.path.exists(numeric_out) and append else 'w', newline='') as nf:
            if numeric_rows:
                base_cols = ['sample_id'] + [k for k in numeric_rows[0].keys() if k != 'sample_id']
                derived_names = [d.get('name') for d in derived_cfg] if derived_cfg else []
                columns = base_cols + derived_names + ['label']
            else:
                columns = ['sample_id', 'label']
            writer = csv.writer(nf)
            if write_header_num:
                writer.writerow(columns)
            # choose device-specific stats if available
            device_stats = None
            if stats is not None:
                # try to read meta file next to in_csv
                meta_path = in_csv + '.meta.json'
                device_id = None
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as mf:
                            jm = json.load(mf)
                            device_id = str(jm.get('device_id', None))
                    except Exception:
                        device_id = None
                if 'devices' in stats and device_id is not None and device_id in stats['devices']:
                    device_stats = stats['devices'][device_id]
                elif 'global' in stats:
                    device_stats = stats['global']

            # load usage mapping from meta.json if present
            usage_map = {}
            name_map = {}
            meta_path = in_csv + '.meta.json'
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as mf:
                        meta = json.load(mf)
                        for c in meta.get('counters', []):
                            try:
                                usage_map[str(c.get('index'))] = c.get('usage')
                                raw_name = c.get('name') or ''
                                safe = ''.join([ch if (ch.isalnum() or ch == '_') else '_' for ch in raw_name])
                                name_map[str(c.get('index'))] = {'raw': raw_name, 'safe': safe}
                            except Exception:
                                continue
                except Exception:
                    usage_map = {}

            def map_usage(val, usage):
                # Defensive mapping: if val is None, return None
                if val is None:
                    return None
                if not usage:
                    return val
                if usage == 'kGpaUsageTypePercentage':
                    return val / 100.0
                if usage == 'kGpaUsageTypeBytes':
                    return val / (1024.0 * 1024.0)  # to MB
                if usage == 'kGpaUsageTypeKilobytes':
                    return val / 1024.0
                if usage == 'kGpaUsageTypeMilliseconds':
                    return val / 1000.0
                if usage == 'kGpaUsageTypeNanoseconds':
                    return val / 1e9
                if usage == 'kGpaUsageTypeCycles':
                    return val / 1e6
                # items, ratio, others -> pass through
                return val

            for nr in numeric_rows:
                row = [nr.get('sample_id', '')]
                # First pass: build mapped_values by applying usage mapping (if present)
                mapped_values = {}
                for col in base_cols[1:]:
                    raw_v = nr.get(col)
                    try:
                        idx = col.split('_')[1]
                    except Exception:
                        idx = None
                    usage = usage_map.get(str(idx)) if idx is not None else None
                    if usage is not None:
                        mv = map_usage(raw_v, usage)
                    else:
                        mv = raw_v
                    mapped_values[col] = mv
                    # add human-readable mappings from meta names (raw and safe)
                    if idx is not None and idx in name_map:
                        nm = name_map[idx]
                        raw = nm.get('raw')
                        safe = nm.get('safe')
                        if raw:
                            mapped_values[raw] = mv
                        if safe:
                            mapped_values[safe] = mv

                # Second pass: write each counter column using mapped value and optional normalization
                for col in base_cols[1:]:
                    v = mapped_values.get(col)
                    if v is None:
                        row.append('')
                        continue
                    if device_stats is not None:
                        try:
                            idx = int(col.split('_')[1])
                            s = device_stats.get(str(idx)) or device_stats.get(idx)
                            if s:
                                if norm_method == 'zscore':
                                    mean = s.get('mean', 0.0)
                                    std = s.get('std', 1.0) or 1.0
                                    row.append((v - mean) / std)
                                else:  # robust
                                    median = s.get('median', 0.0)
                                    iqr = s.get('iqr', 1.0) or 1.0
                                    row.append((v - median) / iqr)
                            else:
                                row.append(v)
                        except Exception:
                            row.append(v)
                    else:
                        row.append(v)

                # Evaluate derived expressions (if any) using mapped values
                if derived_cfg:
                    for d in derived_cfg:
                        name = d.get('name')
                        func = d.get('_func')
                        try:
                            dv = func(mapped_values)
                        except Exception:
                            dv = None
                        if dv is None:
                            row.append('')
                            continue
                        if device_stats is not None:
                            s = device_stats.get(str(name)) or device_stats.get(name)
                            if s:
                                if norm_method == 'zscore':
                                    mean = s.get('mean', 0.0)
                                    std = s.get('std', 1.0) or 1.0
                                    row.append((dv - mean) / std)
                                else:
                                    median = s.get('median', 0.0)
                                    iqr = s.get('iqr', 1.0) or 1.0
                                    row.append((dv - median) / iqr)
                            else:
                                row.append(dv)
                        else:
                            row.append(dv)

                row.append(label)
                writer.writerow(row)
        print('Wrote numeric dataset to', numeric_out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', default='data/gpa_sample.csv')
    parser.add_argument('--out', default='data/gpa_dataset.csv')
    parser.add_argument('--as-numeric', action='store_true', help='Produce numeric dataset file with normalization (.numeric.csv)')
    parser.add_argument('--stats', default=None, help='Path to stats JSON used to normalize numeric conversion')
    parser.add_argument('--norm-method', choices=['zscore','robust'], default='zscore', help='Normalization method to use for numeric dataset')
    parser.add_argument('--append', action='store_true', help='Append to out CSV if it exists')
    parser.add_argument('--derived', default=None, help='Path to JSON file describing derived counter expressions')
    args = parser.parse_args()
    # (convert will handle numeric writing when --as-numeric is passed)
    # Load stats if provided
    stats = None
    if args.stats and os.path.exists(args.stats):
        with open(args.stats, 'r', encoding='utf-8') as sf:
            # load full stats JSON (may contain 'devices' and 'global')
            stats = json.load(sf)

    # load derived config if present and compile expressions
    derived_cfg = None
    if args.derived and os.path.exists(args.derived):
        try:
            with open(args.derived, 'r', encoding='utf-8') as df:
                j = json.load(df)
                # allow file to be a list or contain a top-level 'derived'
                items = j.get('derived', j) if isinstance(j, dict) else j
                # compile expressions into functions
                compiled = []
                for it in items:
                    name = it.get('name')
                    expr = it.get('expr')
                    if not name or not expr:
                        continue
                    # compile expression safely using AST
                    def compile_expr(e):
                        allowed_funcs = {'max': max, 'min': min, 'abs': abs, 'float': float, 'int': int, 'math': math}
                        parsed = ast.parse(e, mode='eval')
                        # validate nodes
                        for node in ast.walk(parsed):
                            if isinstance(node, ast.Call):
                                # must be a simple name (no attribute calls)
                                if not isinstance(node.func, ast.Name) or node.func.id not in allowed_funcs:
                                    raise ValueError('unsupported function in expression')
                            elif isinstance(node, ast.Name):
                                # allow any name (will be resolved from mapping)
                                continue
                            elif isinstance(node, (ast.Expression, ast.Load, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant, ast.Compare, ast.IfExp, ast.Tuple, ast.List, ast.Dict, ast.Subscript, ast.Index, ast.Slice, ast.Attribute, ast.BoolOp, ast.operator, ast.unaryop, ast.cmpop)):
                                continue
                            else:
                                # disallow everything else
                                raise ValueError('unsupported node in expression')
                        code = compile(parsed, '<derived>', 'eval')
                        return lambda mapping: eval(code, {'__builtins__': {}}, {**allowed_funcs, **(mapping or {})})

                    try:
                        func = compile_expr(expr)
                    except Exception as ex:
                        func = None
                    compiled.append({'name': name, 'expr': expr, '_func': func})
                derived_cfg = compiled
        except Exception:
            derived_cfg = None

    convert(args.infile, args.out, as_numeric=args.as_numeric, stats=stats, append=args.append, norm_method=getattr(args, 'norm_method', 'zscore'), derived_cfg=derived_cfg)

if __name__ == '__main__':
    main()
