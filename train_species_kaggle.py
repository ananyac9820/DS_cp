# train_species_kaggle.py
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random, os
from sklearn.model_selection import train_test_split

# CONFIG - adjust if needed
CSV_PATH = "data/species_dataset/dna.csv"   # path to your dataset
SEQ_LEN = 150                               # match your window length (150)
BATCH = 64
EPOCHS = 8
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_OUT = "species_model.pt"
TEST_SPLIT = 0.15
RANDOM_SEED = 42

tok = {"A":0,"C":1,"G":2,"T":3,"N":0}

class DNADataset(Dataset):
    def __init__(self, samples, seq_len=200):
        self.samples = samples
        self.seq_len = seq_len
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        s,label = self.samples[i]
        s = s.upper()
        s_clean = ''.join([c for c in s if c in "ACGTN"])
        if len(s_clean) < self.seq_len:
            s_clean = (s_clean * ((self.seq_len//len(s_clean))+1))[:self.seq_len]
        start = 0
        if len(s_clean) > self.seq_len:
            start = random.randint(0, len(s_clean)-self.seq_len)
        sub = s_clean[start:start+self.seq_len]
        x = torch.tensor([tok.get(c,0) for c in sub], dtype=torch.long)
        return x, torch.tensor(label, dtype=torch.long)

class SpeciesModel(nn.Module):
    def __init__(self, vocab=4, emb=32, hidden=128, n_classes=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)
    def forward(self, x):
        h = self.emb(x)
        _, (hn, _) = self.lstm(h)
        return self.fc(hn[-1])

def prepare_samples(csv_path):
    df = pd.read_csv(csv_path)
    if 'sequence' not in df.columns or 'species' not in df.columns:
        possible_seq = [c for c in df.columns if 'seq' in c.lower()]
        possible_lab = [c for c in df.columns if 'species' in c.lower() or 'label' in c.lower() or c.lower()=='class']
        if possible_seq and possible_lab:
            df = df.rename(columns={possible_seq[0]:'sequence', possible_lab[0]:'species'})
        else:
            raise SystemExit("CSV missing 'sequence' or 'species' columns. Found: " + ",".join(df.columns))
    df = df.dropna(subset=['sequence','species'])
    labels = sorted(df['species'].unique())
    lab2idx = {lab:i for i,lab in enumerate(labels)}
    print("Labels found:", lab2idx)
    samples = [(row['sequence'], lab2idx[row['species']]) for _,row in df.iterrows()]
    train, test = train_test_split(samples, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=[s[1] for s in samples])
    return train, test, labels

def train():
    train_samples, test_samples, labels = prepare_samples(CSV_PATH)
    print("Train samples:", len(train_samples), "Test:", len(test_samples))
    n_classes = len(labels)
    train_ds = DNADataset(train_samples, seq_len=SEQ_LEN)
    test_ds = DNADataset(test_samples, seq_len=SEQ_LEN)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

    model = SpeciesModel(n_classes=n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(EPOCHS):
        model.train()
        tot_loss = 0; tot = 0; corr = 0
        for X,y in train_dl:
            X,y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(X)
            loss = loss_fn(out,y)
            loss.backward(); opt.step()
            tot_loss += loss.item()*X.size(0)
            pred = out.argmax(1)
            corr += (pred==y).sum().item()
            tot += X.size(0)
        train_acc = corr/tot

        # Evaluate
        model.eval()
        vtot = 0; vcorr = 0
        with torch.no_grad():
            for Xv,yv in test_dl:
                Xv,yv = Xv.to(DEVICE), yv.to(DEVICE)
                ov = model(Xv)
                pv = ov.argmax(1)
                vcorr += (pv==yv).sum().item()
                vtot += Xv.size(0)
        val_acc = vcorr/vtot
        print(f"Epoch {ep+1}/{EPOCHS}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

    torch.save({"state":model.state_dict(), "labels": labels}, MODEL_OUT)
    print("✅ Saved model ->", MODEL_OUT)
    print("Labels:", labels)

if __name__=="__main__":
    train()
