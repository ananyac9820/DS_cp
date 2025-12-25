# quick_test_model.py
import torch
import numpy as np

# load checkpoint
ck = torch.load("species_model.pt", map_location="cpu")
state = ck.get("state") or ck.get("state_dict") or ck
labels = ck.get("labels") or ck.get("label2idx") or None

# recover labels list
if isinstance(labels, dict):
    idx2 = {v:k for k,v in labels.items()}
    labels_list = [idx2[i] for i in range(len(idx2))]
elif isinstance(labels, (list,tuple)):
    labels_list = list(labels)
else:
    labels_list = None

print("Labels from checkpoint:", labels_list)

# import the exact model class used by trainer
from train_species_kaggle import SpeciesModel, SEQ_LEN

n = len(labels_list) if labels_list else (ck.get("n_classes") or 3)
model = SpeciesModel(n_classes=n)
model.load_state_dict(state)
model.eval()

def infer(s, W=SEQ_LEN):
    s = "".join([c for c in s.upper() if c in "ACGT"])
    s = (s + "A"*W)[:W]
    idx = [ {'A':0,'C':1,'G':2,'T':3}[c] for c in s ]
    x = torch.tensor([idx], dtype=torch.long)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).numpy().tolist()[0]
    return probs

high = "GGCGGCGCGCCGGGCGCGCGCCGCGGCCGCGCGGCGCGCCGGCGGCGCGGCGCGGCCGCGGCGCGGCGCCGCGGCCGGCGGCGCGGCGCGCCGCGGCGCGGCCGCGGCGCGGCGCG"
low  = "ATATATATTTAAATATATTAAATATATATAATATATAATATTTATATAAATAATATTTAAATATATATATAAATATAAATTTATATAAATAATTTATATATATAAATATATA"

print("High-GC probs:", infer(high))
print("Low-GC  probs:", infer(low))
