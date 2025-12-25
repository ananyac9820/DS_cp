# app.py
from flask import Flask, render_template, request, jsonify
import torch
import os
import random
import numpy as np

# Import your NextBaseModel from your training script
# Make sure train_nextbase.py is importable and defines NextBaseModel
try:
    from train_nextbase import NextBaseModel
except Exception as e:
    NextBaseModel = None
    print("Warning: Could not import NextBaseModel from train_nextbase.py:", e)

# Import SimpleSpeciesModel only if you used that name (trainer saved state only)
# We'll create species model from a simple LSTM class matching train script
class SimpleSpeciesModel(torch.nn.Module):
    def __init__(self, vocab=4, emb=32, hidden=128, n_classes=3):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, emb)
        self.lstm = torch.nn.LSTM(emb, hidden, batch_first=True)
        self.fc = torch.nn.Linear(hidden, n_classes)
    def forward(self, x):
        h = self.emb(x)
        _, (hn, _) = self.lstm(h)
        return self.fc(hn[-1])

app = Flask(__name__)

# Paths
NEXT_MODEL_PATH = "nextbase_model.pt"
SPECIES_MODEL_PATH = "species_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Token maps
_tok = {"A":0,"C":1,"G":2,"T":3}
_rev = {0:"A",1:"C",2:"G",3:"T"}

# ----------------- Next-base model load -----------------
next_model = None
if NextBaseModel is not None:
    try:
        next_model = NextBaseModel()
        if os.path.exists(NEXT_MODEL_PATH):
            state = torch.load(NEXT_MODEL_PATH, map_location=DEVICE)
            # support checkpoint format or raw state_dict
            if isinstance(state, dict) and ("state_dict" in state or "model" in state or "state" in state):
                sd = state.get("state_dict") or state.get("model") or state.get("state")
                next_model.load_state_dict(sd)
            else:
                next_model.load_state_dict(state)
            next_model.to(DEVICE)
            print("✅ Next-base model loaded.")
        else:
            # keep model instance but uninitialized weights — use demo fallback
            print("ℹ️ Next-base model file not found. UI will run demo fallback until model exists.")
            next_model = None
    except Exception as e:
        print("⚠️ Failed loading next-base model:", e)
        next_model = None
else:
    print("⚠️ NextBaseModel class not available (train_nextbase.py missing or error). Next-base disabled.")
    next_model = None

# ----------------- Species model load -----------------
species_model = None
species_idx2label = None
species_seq_len = 150   # must match training SEQ_LEN (change if you trained with another value)

if os.path.exists(SPECIES_MODEL_PATH):
    try:
        ckpt = torch.load(SPECIES_MODEL_PATH, map_location=DEVICE)
        labels = ckpt.get("labels") or ckpt.get("label2idx") or ckpt.get("label_map") or None
        # if label2idx (dict) found, produce idx2label
        if isinstance(labels, dict):
            idx2label = {v:k for k,v in labels.items()}
            ordered_labels = [idx2label[i] for i in range(len(idx2label))]
        elif isinstance(labels, (list,tuple)):
            ordered_labels = list(labels)
            idx2label = {i:ordered_labels[i] for i in range(len(ordered_labels))}
        else:
            ordered_labels = None
            idx2label = None

        n_classes = len(ordered_labels) if ordered_labels else (ckpt.get("n_classes") or 3)
        m = SimpleSpeciesModel(n_classes=n_classes)
        state = ckpt.get("state") or ckpt.get("state_dict") or ckpt
        # state might already be the dict
        m.load_state_dict(state)
        m.to(DEVICE).eval()
        species_model = m
        species_idx2label = {i:(ordered_labels[i] if ordered_labels else f"class_{i}") for i in range(n_classes)}
        print("✅ Species model loaded. Labels:", species_idx2label)
    except Exception as e:
        print("⚠️ Failed loading species model:", e)
        species_model = None
else:
    print("ℹ️ Species model not found — UI will use heuristic fallback until model exists.")
    species_model = None

# ----------------- Helper functions -----------------
def predict_next_base(seq):
    seq = seq.strip().upper()
    if next_model is None:
        # demo fallback: last-base frequency or random
        # simple heuristic: choose most frequent base in input context if any
        seq = ''.join([c for c in seq if c in "ACGT"])
        if len(seq) == 0:
            return random.choice(["A","C","G","T"])
        from collections import Counter
        c = Counter(seq)
        return c.most_common(1)[0][0]
    # prepare tensor using last 50 bases or sequence padding - adapt if NextBaseModel expects different preprocessing
    ctx = [ _tok[c] for c in seq[-50:] if c in _tok ]
    if len(ctx) == 0:
        return random.choice(["A","C","G","T"])
    x = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = next_model(x)
        if isinstance(out, tuple): out = out[0]
        pred = torch.argmax(out, dim=1).item()
    return _rev.get(pred, random.choice(["A","C","G","T"]))

def heuristic_species_guess(seq):
    s = ''.join([c for c in seq.upper() if c in "ACGT"])
    if len(s) == 0:
        return "Unknown"
    gc = (s.count("G") + s.count("C")) / len(s)
    if gc < 0.45:
        return "low_gc"
    elif gc < 0.55:
        return "medium_gc"
    else:
        return "high_gc"

def classify_species_with_probs(seq):
    seq = ''.join([c for c in seq.upper() if c in "ACGT"])
    if species_model is None:
        label = heuristic_species_guess(seq)
        return label, {label: 1.0}
    # prepare input of fixed length
    W = species_seq_len
    s = (seq + "A"*W)[:W]
    x = torch.tensor([[ _tok.get(c,0) for c in s ]], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = species_model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().tolist()[0]
        pred_idx = int(np.argmax(probs))
    label = species_idx2label.get(pred_idx, f"label_{pred_idx}")
    prob_dict = { species_idx2label.get(i, f"label_{i}"): float(probs[i]) for i in range(len(probs)) }
    return label, prob_dict

# ----------------- Routes -----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    sequence = request.form.get("sequence", "").strip()
    if not sequence:
        return jsonify({"error":"No sequence provided"}), 400
    nb = predict_next_base(sequence)
    return jsonify({"next_base": nb})

@app.route("/classify", methods=["POST"])
def classify():
    sequence = request.form.get("sequence", "").strip()
    if not sequence:
        return jsonify({"error":"No sequence provided"}), 400
    label, probs = classify_species_with_probs(sequence)
    return jsonify({"species": label, "probabilities": probs})

if __name__ == "__main__":
    # Flask debug off for demo if you want; set debug=True to auto-reload during dev
    app.run(debug=True)
