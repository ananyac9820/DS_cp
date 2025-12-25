import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

# ---------- Config ----------
SEQ_LEN = 50
BATCH_SIZE = 128
EPOCHS = 5
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Dataset ----------
class NextBaseDataset(Dataset):
    def __init__(self, file_path):
        self.samples = []
        self.tok = {"A":0,"C":1,"G":2,"T":3}
        with open(file_path) as f:
            for line in f:
                if not line.strip(): continue
                ctx, nb = line.strip().split("\t")
                if set(ctx+nb) <= {"A","C","G","T"}:
                    x = [self.tok[c] for c in ctx]
                    y = self.tok[nb]
                    self.samples.append((x, y))
        random.shuffle(self.samples)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

# ---------- Model ----------
class NextBaseModel(nn.Module):
    def __init__(self, vocab=4, embed=16, hidden=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab)
    def forward(self, x):
        x = self.emb(x)
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out

# ---------- Train ----------
def train():
    ds = NextBaseDataset("train_pairs.txt")
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = NextBaseModel().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        total_loss = 0
        for X, y in dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(dl):.4f}")

    torch.save(model.state_dict(), "nextbase_model.pt")
    print("✅ Saved model -> nextbase_model.pt")

if __name__ == "__main__":
    train()
