# eval_species.py
import torch, json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from train_species_kaggle import DNADataset, SpeciesModel, prepare_samples, SEQ_LEN, CSV_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "species_model.pt"

def load_checkpoint(path):
    ck = torch.load(path, map_location=DEVICE)
    state = ck.get("state") or ck.get("state_dict") or ck
    labels = ck.get("labels") or ck.get("label2idx") or None
    return state, labels

state, labels = load_checkpoint(CKPT)
if labels is None:
    print("No labels found in checkpoint. Exiting.")
    raise SystemExit(1)

labels = list(labels)
n_classes = len(labels)
print("Labels:", labels)

# Prepare full dataset (train+test) using the same CSV split logic
train_samples, test_samples, _ = prepare_samples(CSV_PATH)
print("Test samples:", len(test_samples))
test_ds = DNADataset(test_samples, seq_len=SEQ_LEN)
from torch.utils.data import DataLoader
dl = DataLoader(test_ds, batch_size=64, shuffle=False)

# load model with correct n_classes
model = SpeciesModel(n_classes=n_classes).to(DEVICE)
model.load_state_dict(state)
model.eval()

y_true = []
y_pred = []
prob_list = []

with torch.no_grad():
    for X,y in dl:
        X = X.to(DEVICE)
        out = model(X)            # logits
        probs = torch.softmax(out, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(preds.tolist())
        prob_list.extend(probs.tolist())

# Metrics
print("\nClassification report:\n")
print(classification_report(y_true, y_pred, target_names=labels, digits=4))

cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print("\nConfusion matrix:\n", cm_df)

# Save confusion matrix plot
import seaborn as sns
plt.figure(figsize=(6,5))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.title("Confusion matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")
