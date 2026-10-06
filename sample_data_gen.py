import os
import argparse
import random
import csv

DEFAULT_EVENT_TYPES = [
    'SET_MODE','ALLOC_MEMORY','FREE_MEMORY','COMPUTE_START','COMPUTE_END','ERROR_OVERTEMP','ERROR_RST','COMPUTE_HANG','INTERRUPT','DRIVER_RECOVER'
]


def gen_sequence(event_types, max_len=50, crash=False):
    # crash indicates a higher probability of ERROR or HANG events
    seq_len = random.randint(3, max_len)
    seq = []
    for _ in range(seq_len):
        if crash and random.random() < 0.2:
            seq.append(random.choice(['COMPUTE_HANG','ERROR_OVERTEMP','ERROR_RST']))
        else:
            seq.append(random.choice(event_types))
    return seq


def write_csv(path, sequences):
    # sequences: list of (sample_id, seq, label)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id','timestamp','event_type','severity','message','label'])
        for sample_id, seq, label in sequences:
            ts = 1
            for ev in seq:
                severity = random.randint(0, 3)
                msg = f"msg {ev}"
                writer.writerow([sample_id, ts, ev, severity, msg, label])
                ts += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='./data')
    parser.add_argument('--num-files', type=int, default=2)
    parser.add_argument('--samples-per-file', type=int, default=200)
    parser.add_argument('--max-seq-len', type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for i in range(args.num_files):
        sequences = []
        for s in range(args.samples_per_file):
            crash = random.random() < 0.2
            seq = gen_sequence(DEFAULT_EVENT_TYPES, max_len=args.max_seq_len, crash=crash)
            label = 1 if crash else 0
            sequences.append((f"file{i}_sample{s}", seq, label))
        out_path = os.path.join(args.out_dir, f"data_{i:03d}.csv")
        write_csv(out_path, sequences)
        print(f"Wrote {out_path} with {len(sequences)} samples")
