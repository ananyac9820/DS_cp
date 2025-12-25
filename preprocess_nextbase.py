from pathlib import Path

def read_fasta(filepath):
    seq = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line.upper())
    return "".join(seq)

def make_training_pairs(sequence, window=50, step=1, out_file="train_pairs.txt"):
    with open(out_file, "w") as out:
        for i in range(0, len(sequence) - window - 1, step):
            context = sequence[i:i+window]
            next_base = sequence[i+window]
            if set(context + next_base) <= {"A","C","G","T"}:
                out.write(f"{context}\t{next_base}\n")
    print(f"Saved training pairs to {out_file}")

if __name__ == "__main__":
    fasta_path = Path(r"C:\Users\anany\OneDrive\Desktop\SEM3\DS\DS_cp\encoli.fasta")

    seq = read_fasta(fasta_path)
    print("Sequence length:", len(seq))
    make_training_pairs(seq, window=50, step=5, out_file="train_pairs.txt")
