# train_species_cnn.py
"""
Small 1D-CNN species classifier trainer.
Saves checkpoint: {"state": model.state_dict(), "labels": labels}
"""
import pandas as pd, random, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# --- CONFIG ---
CSV_PATH = "data/species_dataset/dna.csv"
SEQ_LEN = 150   # must match dataset window used when creating data
BATCH = 64
EPOCHS = 12
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_OUT = "species_model.pt"
TEST_SPLIT = 0.15
SEED = 42
AUGMENT = True   # reverse complement + small mut
MUT_FRAC = 0.02  # 2% random substitutions when augmenting
# ---------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TOK = {"A":0,"C":1,"G":2,"T":3}

def revcomp(s):
    tr = str.maketrans("ACGT","TGCA")
    return s.translate(tr)[::-1]

def augment_seq(s):
    # maybe reverse complement or mutate small fraction
    if random.random() < 0.5:
        s = revcomp(s)
    # small mutation
    if random.random() < 0.5:
        s_list = list(s)
        nmut = max(1, int(len(s)*MUT_FRAC))
        for i in random.sample(range(len(s)), nmut):
            choices = [c for c in "ACGT" if c != s_list[i]]
            s_list[i] = random.choice(choices)
        s = "".join(s_list)
    return s

class DNADataset(Dataset):
    def __init__(self, rows, seq_len=SEQ_LEN, augment=False):
        self.rows = rows
        self.seq_len = seq_len
        self.augment = augment
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        s,label = self.rows[idx]
        s = ''.join([c for c in s.upper() if c in "ACGT"])
        if len(s) < self.seq_len:
            s = (s * ((self.seq_len//len(s))+1))[:self.seq_len]
        if len(s) > self.seq_len:
            start = random.randint(0, len(s)-self.seq_len)
            s = s[start:start+self.seq_len]
        if self.augment:
            s = augment_seq(s)
        x = torch.tensor([TOK[c] for c in s], dtype=torch.long)
        return x, torch.tensor(label, dtype=torch.long)

class SmallCNN(nn.Module):
    def __init__(self, vocab=4, emb=16, n_filters=64, kernel=7, n_classes=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.conv1 = nn.Conv1d(emb, n_filters, kernel_size=kernel, padding=kernel//2)
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=kernel, padding=kernel//2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(n_filters, n_classes)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
    def forward(self, x):
        # x: (B, L) -> emb -> (B, L, E) -> permute to (B, E, L)
        e = self.emb(x).permute(0,2,1)
        h = self.relu(self.conv1(e))
        h = self.relu(self.conv2(h))
        h = self.pool(h).squeeze(-1)
        h = self.drop(h)
        return self.fc(h)

def prepare(csv_path):
    df = pd.read_csv(csv_path)
    if 'sequence' not in df.columns or 'species' not in df.columns:
        # try auto-rename
        seqc = [c for c in df.columns if 'seq' in c.lower()]
        labc = [c for c in df.columns if 'species' in c.lower() or 'label' in c.lower() or c.lower()=='class']
        if seqc and labc:
            df = df.rename(columns={seqc[0]:'sequence', labc[0]:'species'})
        else:
            raise SystemExit("CSV missing sequence/species columns. Found: "+",".join(df.columns))
    df = df.dropna(subset=['sequence','species'])
    labels = sorted(df['species'].unique())
    lab2idx = {lab:i for i,lab in enumerate(labels)}
    rows = [(r['sequence'], lab2idx[r['species']]) for _,r in df.iterrows()]
    train, test = train_test_split(rows, test_size=TEST_SPLIT, random_state=SEED, stratify=[r[1] for r in rows])
    return train, test, labels

def train():
    train_rows, test_rows, labels = prepare(CSV_PATH)
    print("Labels:", labels)
    train_ds = DNADataset(train_rows, augment=AUGMENT)
    test_ds = DNADataset(test_rows, augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=False)
    test_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False)
    model = SmallCNN(n_classes=len(labels)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    best_val = 0.0
    for ep in range(EPOCHS):
        model.train()
        tot=0; correct=0; loss_sum=0.0
        for X,y in train_dl:
            X,y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(X)
            loss = loss_fn(out,y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()*X.size(0)
            preds = out.argmax(1)
            correct += (preds==y).sum().item()
            tot += X.size(0)
        train_acc = correct/tot
        # validate
        model.eval()
        vtot=0; vcorr=0
        with torch.no_grad():
            for Xv,yv in test_dl:
                Xv,yv = Xv.to(DEVICE), yv.to(DEVICE)
                ov = model(Xv)
                pv = ov.argmax(1)
                vcorr += (pv==yv).sum().item()
                vtot += Xv.size(0)
        val_acc = vcorr/vtot
        print(f"Epoch {ep+1}/{EPOCHS} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"state": model.state_dict(), "labels": labels}, MODEL_OUT)
            print("Saved improved model ->", MODEL_OUT)
    print("Training complete. Best val:", best_val)

if __name__=="__main__":
    train()
