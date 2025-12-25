#!/usr/bin/env python3
"""
build_synth_species_dataset.py (improved)

Creates a small synthetic species-labeled dataset from one or more FASTA files.

Features:
 - choose classes to generate: high_gc, low_gc, mutated (default: all)
 - set GC thresholds for high/low GC
 - set mutation fraction for mutated class
 - can append to existing CSV instead of overwriting (--append)
 - accept multiple FASTA files (comma separated) or auto-detect common names
 - progress prints and safety checks

Usage examples:
  # default (auto-detect FASTA in project root)
  python build_synth_species_dataset.py

  # specify FASTA explicitly
  python build_synth_species_dataset.py --fasta expanded_sample.fasta --per_class 300 --window 150

  # use multiple fastas
  python build_synth_species_dataset.py --fasta ecoli.fna,yeast.fna --per_class 200

  # only generate high_gc and low_gc classes, append to csv
  python build_synth_species_dataset.py --classes high_gc,low_gc --append --per_class 200

Outputs:
 - data/species_synth/<class>/<class>.fasta   (multi-FASTA per class)
 - data/species_dataset/dna.csv              (sequence,species)
 - data/species_synth/manifest.json
"""
from pathlib import Path
import argparse, random, textwrap, json, sys, os

def read_fasta_concat(path: Path):
    seqs = []
    with open(path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('>'):
                continue
            seqs.append(line.upper())
    return "".join(seqs)

def gc_fraction(s):
    return (s.count('G') + s.count('C')) / len(s) if len(s)>0 else 0.0

def mutate_sequence(s, frac=0.05, rnd=None):
    if rnd is None: rnd = random
    s_list = list(s)
    nmut = max(1, int(len(s)*frac))
    for idx in rnd.sample(range(len(s)), nmut):
        orig = s_list[idx]
        choices = [c for c in "ACGT" if c != orig]
        s_list[idx] = rnd.choice(choices)
    return ''.join(s_list)

def find_fasta_candidate(candidates):
    for c in candidates:
        if not c: continue
        p = Path(c)
        if p.exists():
            return [p]
    # fallback: try a list of common names in cwd
    found = []
    for name in ["expanded_sample.fasta","expanded_sample.fa","encoli.fasta","ecoli.fasta",
                 "sample.fasta","GCF_000005845.2_ASM584v2_genomic.fna"]:
        p = Path(name)
        if p.exists():
            found.append(p)
    return found

def sample_windows_from_seq(seq, window, n_samples, allow_ambig=False, rnd=None):
    if rnd is None: rnd = random
    out = []
    L = len(seq)
    if L < window:
        return out
    attempts = 0
    max_attempts = max(100000, n_samples * 50)
    while len(out) < n_samples and attempts < max_attempts:
        attempts += 1
        start = rnd.randint(0, L - window)
        w = seq[start:start+window]
        if not allow_ambig and (set(w) - set("ACGT")):
            continue
        out.append(w)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=str, default="", help="Comma-separated FASTA paths or empty to auto-detect")
    parser.add_argument("--window", type=int, default=200, help="Window length")
    parser.add_argument("--per_class", type=int, default=500, help="Samples per class (reduce if slow)")
    parser.add_argument("--out", type=str, default="data", help="Output root directory")
    parser.add_argument("--mut_frac", type=float, default=0.05, help="Fraction positions to mutate for mutated class")
    parser.add_argument("--classes", type=str, default="high_gc,low_gc,mutated", help="Which classes to create (comma-separated)")
    parser.add_argument("--high_gc_thresh", type=float, default=0.60, help="GC fraction threshold for high_gc")
    parser.add_argument("--low_gc_thresh", type=float, default=0.40, help="GC fraction threshold for low_gc")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV instead of overwrite")
    parser.add_argument("--allow_ambig", action="store_true", help="Allow ambiguous bases in windows")
    args = parser.parse_args()

    random.seed(args.seed)
    rnd = random

    # parse classes
    selected = [c.strip() for c in args.classes.split(",") if c.strip()]
    allowed_classes = {"high_gc","low_gc","mutated"}
    for c in selected:
        if c not in allowed_classes:
            print(f"Unknown class '{c}'. Allowed: high_gc, low_gc, mutated")
            sys.exit(1)

    # collect fasta files
    fasta_list = []
    if args.fasta:
        for p in args.fasta.split(","):
            p = p.strip()
            if not p: continue
            pp = Path(p)
            if not pp.exists():
                print("FASTA not found:", pp)
                sys.exit(1)
            fasta_list.append(pp)
    else:
        detected = find_fasta_candidate([None])  # auto-detect
        if detected:
            fasta_list = detected
    if not fasta_list:
        print("No FASTA found. Provide with --fasta or put expanded_sample.fasta in the folder.")
        sys.exit(1)

    print("Using FASTA files:", ", ".join(str(p) for p in fasta_list))

    # read sequences (concatenate all FASTAs)
    full_seq = ""
    for f in fasta_list:
        s = read_fasta_concat(f)
        full_seq += s
    if len(full_seq) < args.window:
        print("Combined FASTA length smaller than window. Aborting.")
        sys.exit(1)
    print("Total concatenated sequence length:", len(full_seq))

    W = args.window
    target = args.per_class

    # prepare containers
    classes_windows = { "high_gc": [], "low_gc": [], "mutated": [] }

    # collect high/low GC windows
    attempts = 0
    max_attempts = max(100000, target*100)
    while (("high_gc" not in selected or len(classes_windows["high_gc"]) >= target) and
           ("low_gc" not in selected or len(classes_windows["low_gc"]) >= target)) == False and attempts < max_attempts:
        # loop condition: run until both of requested high/low are collected OR attempts exhausted
        break  # use more explicit loops below

    # Explicit loops per class (safer)
    if "high_gc" in selected:
        print("Sampling high GC windows (GC >= {:.2f})...".format(args.high_gc_thresh))
        while len(classes_windows["high_gc"]) < target and attempts < max_attempts:
            attempts += 1
            start = rnd.randint(0, len(full_seq)-W-1)
            w = full_seq[start:start+W]
            if not args.allow_ambig and (set(w) - set("ACGT")):
                continue
            if gc_fraction(w) >= args.high_gc_thresh:
                classes_windows["high_gc"].append(w)
        print(f"Collected high_gc: {len(classes_windows['high_gc'])}")

    if "low_gc" in selected:
        print("Sampling low GC windows (GC <= {:.2f})...".format(args.low_gc_thresh))
        while len(classes_windows["low_gc"]) < target and attempts < max_attempts:
            attempts += 1
            start = rnd.randint(0, len(full_seq)-W-1)
            w = full_seq[start:start+W]
            if not args.allow_ambig and (set(w) - set("ACGT")):
                continue
            if gc_fraction(w) <= args.low_gc_thresh:
                classes_windows["low_gc"].append(w)
        print(f"Collected low_gc: {len(classes_windows['low_gc'])}")

    if "mutated" in selected:
        print(f"Generating mutated windows (mut_frac={args.mut_frac})...")
        muts = 0
        mut_attempts = 0
        while len(classes_windows["mutated"]) < target and mut_attempts < max_attempts:
            mut_attempts += 1
            start = rnd.randint(0, len(full_seq)-W-1)
            w = full_seq[start:start+W]
            if not args.allow_ambig and (set(w) - set("ACGT")):
                continue
            m = mutate_sequence(w, frac=args.mut_frac, rnd=rnd)
            classes_windows["mutated"].append(m)
            muts += 1
        print(f"Collected mutated: {len(classes_windows['mutated'])}")

    # prepare output dirs
    out_root = Path(args.out)
    fasta_out_root = out_root / "species_synth"
    csv_out_root = out_root / "species_dataset"
    fasta_out_root.mkdir(parents=True, exist_ok=True)
    csv_out_root.mkdir(parents=True, exist_ok=True)
    csv_file = csv_out_root / "dna.csv"

    write_header = True
    if args.append and csv_file.exists():
        write_header = False

    # write FASTA and CSV (append or write)
    mode = "a" if args.append else "w"
    written = 0
    with open(csv_file, mode) as cf:
        if write_header:
            cf.write("sequence,species\n")
        for cls in ["high_gc","low_gc","mutated"]:
            if cls not in selected:
                continue
            windows = classes_windows[cls]
            dir_cls = fasta_out_root / cls
            dir_cls.mkdir(parents=True, exist_ok=True)
            fasta_file = dir_cls / f"{cls}.fasta"
            # append FASTA if exists, otherwise write new
            fmode = "a" if fasta_file.exists() else "w"
            with open(fasta_file, fmode) as fh:
                start_i = 0
                if fmode == "a":
                    # try to estimate start_i from existing entries
                    existing = sum(1 for _ in Path(fasta_file).open())
                    start_i = existing // 2  # rough but fine
                for i, w in enumerate(windows):
                    idx = start_i + i
                    fh.write(f">{cls}_{idx}\n")
                    fh.write('\n'.join(textwrap.wrap(w,80)) + '\n')
                    cf.write(f"{w},{cls}\n")
                    written += 1

    manifest = {
        "source_fastas": [str(p) for p in fasta_list],
        "window_len": W,
        "samples_per_class": target,
        "selected_classes": selected,
        "counts": {k: len(v) for k,v in classes_windows.items()},
        "csv": str(csv_file),
        "fasta_dirs": {k: str((fasta_out_root/k)) for k in classes_windows.keys()}
    }
    (fasta_out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote/updated CSV: {csv_file}  (appended={args.append})")
    print("Created/updated FASTA dirs under:", fasta_out_root)
    print("Manifest:", json.dumps(manifest, indent=2))
    print("Total sequences written to CSV:", written)

if __name__ == "__main__":
    main()
