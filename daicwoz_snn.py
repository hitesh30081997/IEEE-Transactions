"""
DAIC-WOZ depression recognition with snnTorch
=============================================
Expected layout:
  Depression/daicwoz/Audio/300_AUDIO.wav
  Depression/daicwoz/TRANSCRIPT/300_TRANSCRIPT.csv
  Depression/daicwoz/train_split_Depression_AVEC2017.csv
  Depression/daicwoz/dev_split_Depression_AVEC2017.csv
  Depression/daicwoz/test_split_Depression_AVEC2017.csv   (usually NO labels)
  Depression/daicwoz/full_test_split.csv                  (optional, has labels)

Run:
  python daicwoz_snn.py --root Depression/daicwoz
  python daicwoz_snn.py --root Depression/daicwoz --epochs 30 --hidden 128

Pipeline:
  1. keep only the Participant's speech (transcript timestamps), drop Ellie
  2. MFCC + deltas, normalised per participant, cached to .npy
  3. 4 s segments (200 SNN time steps)
  4. train on TRAIN, pick best epoch on DEV (speaker-level F1)
  5. evaluate on TEST only if labels are available

Requirements: pip install torch snntorch librosa scikit-learn pandas
"""

import argparse, os
from collections import defaultdict

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import snntorch as snn
from snntorch import surrogate
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

SR = 16000
N_MFCC = 20
HOP_MS, WIN_MS = 20, 40
SEG_SEC, SEG_HOP_SEC = 4.0, 2.0
N_FEATS = N_MFCC * 3
MIN_TURN_SEC = 0.3          # skip tiny turns ("mm", "yeah")


# ---------------------------------------------------------------------------
# 1. READ THE SPLIT FILES (column names differ slightly between files)
# ---------------------------------------------------------------------------
def read_split(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    id_col = next(cols[c] for c in cols if "participant" in c)
    lab_col = next((cols[c] for c in cols if "binary" in c), None)
    if lab_col is None:
        print(f"  {os.path.basename(path)}: no labels, skipped")
        return None
    return [(int(r[id_col]), int(r[lab_col])) for _, r in df.iterrows()]


# ---------------------------------------------------------------------------
# 2. PARTICIPANT-ONLY AUDIO
#    Transcript: tab-separated  start_time  stop_time  speaker  value
# ---------------------------------------------------------------------------
def read_transcript(path):
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] < 3:                                  # a few files use commas
        df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df[df["speaker"].astype(str).str.strip().str.lower() == "participant"]


def participant_audio(root, pid):
    wav = os.path.join(root, "Audio", f"{pid}_AUDIO.wav")
    trn = os.path.join(root, "TRANSCRIPT", f"{pid}_TRANSCRIPT.csv")
    y, _ = librosa.load(wav, sr=SR, mono=True)
    turns = read_transcript(trn)
    chunks = []
    for _, t in turns.iterrows():
        s, e = float(t["start_time"]), float(t["stop_time"])
        if e - s >= MIN_TURN_SEC:
            chunks.append(y[int(s * SR):int(e * SR)])
    if not chunks:
        raise ValueError("no participant speech found")
    return np.concatenate(chunks)


# ---------------------------------------------------------------------------
# 3. FEATURES (cached, because each interview is 7-30 minutes)
# ---------------------------------------------------------------------------
def extract_features(y):
    hop, win = int(SR * HOP_MS / 1000), int(SR * WIN_MS / 1000)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=1024,
                                hop_length=hop, win_length=win)
    feats = np.concatenate([mfcc, librosa.feature.delta(mfcc),
                            librosa.feature.delta(mfcc, order=2)]).T
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
    return feats.astype(np.float32)                      # [frames, 60]


def load_features(root, pid, cache_dir):
    cache = os.path.join(cache_dir, f"{pid}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    feats = extract_features(participant_audio(root, pid))
    np.save(cache, feats)
    return feats


def segment(feats):
    seg = int(SEG_SEC * 1000 / HOP_MS)
    hop = int(SEG_HOP_SEC * 1000 / HOP_MS)
    if len(feats) < seg:
        feats = np.pad(feats, ((0, seg - len(feats)), (0, 0)))
    return [feats[s:s + seg] for s in range(0, len(feats) - seg + 1, hop)]


class DaicDataset(Dataset):
    def __init__(self, root, split, cache_dir, name):
        self.x, self.y, self.spk = [], [], []
        missing = []
        for i, (pid, label) in enumerate(split, 1):
            try:
                feats = load_features(root, pid, cache_dir)
            except Exception as e:
                missing.append((pid, str(e)[:60])); continue
            for s in segment(feats):
                self.x.append(torch.from_numpy(np.ascontiguousarray(s)))
                self.y.append(label); self.spk.append(pid)
            print(f"\r  {name}: {i}/{len(split)} participants", end="")
        print(f"\r  {name}: {len(set(self.spk))} participants, "
              f"{len(self.x)} segments, depressed={len({p for p, l in zip(self.spk, self.y) if l})}")
        for pid, err in missing:
            print(f"    skipped {pid}: {err}")

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i], self.spk[i]


def collate(b):
    x, y, s = zip(*b)
    return torch.stack(x), torch.tensor(y), list(s)


# ---------------------------------------------------------------------------
# 4. MODEL
# ---------------------------------------------------------------------------
class SpeechSNN(nn.Module):
    def __init__(self, n_in=N_FEATS, n_hidden=128, n_out=2, beta=0.9, drop=0.3):
        super().__init__()
        g = surrogate.fast_sigmoid(slope=25)
        self.fc1, self.fc2, self.fc3 = (nn.Linear(n_in, n_hidden),
                                        nn.Linear(n_hidden, n_hidden),
                                        nn.Linear(n_hidden, n_out))
        self.lif1 = snn.Leaky(beta=beta, learn_beta=True, spike_grad=g)
        self.lif2 = snn.Leaky(beta=beta, learn_beta=True, spike_grad=g)
        self.lif3 = snn.Leaky(beta=beta, learn_beta=True, spike_grad=g)
        self.drop = nn.Dropout(drop)

    def forward(self, x):                                # x: [B, T, F]
        x = x.permute(1, 0, 2)
        m1, m2, m3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
        out = []
        for t in range(x.size(0)):
            s1, m1 = self.lif1(self.fc1(x[t]), m1)
            s2, m2 = self.lif2(self.fc2(self.drop(s1)), m2)
            s3, m3 = self.lif3(self.fc3(self.drop(s2)), m3)
            out.append(s3)
        return torch.stack(out)                          # [T, B, 2]


# ---------------------------------------------------------------------------
# 5. TRAIN / EVAL
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    counts, labels = defaultdict(lambda: torch.zeros(2)), {}
    for x, y, spk in loader:
        c = model(x.to(device)).sum(0).cpu()
        for ci, yi, si in zip(c, y.tolist(), spk):
            counts[si] += ci; labels[si] = yi
    pids = sorted(labels)
    t = [labels[p] for p in pids]
    # ratio of depressed spikes, so ties (both 0) default to "not depressed"
    p = [int(counts[q][1] > counts[q][0]) for q in pids]
    return {"bal_acc": balanced_accuracy_score(t, p),
            "f1": f1_score(t, p, average="macro", zero_division=0),
            "f1_dep": f1_score(t, p, pos_label=1, zero_division=0),
            "cm": confusion_matrix(t, p, labels=[0, 1])}


def main(a):
    torch.manual_seed(0); np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = os.path.join(a.root, "features_cache"); os.makedirs(cache, exist_ok=True)

    train = read_split(os.path.join(a.root, "train_split_Depression_AVEC2017.csv"))
    dev = read_split(os.path.join(a.root, "dev_split_Depression_AVEC2017.csv"))
    test = (read_split(os.path.join(a.root, "full_test_split.csv"))
            or read_split(os.path.join(a.root, "test_split_Depression_AVEC2017.csv")))

    print("Loading features (first run is slow, then cached):")
    tr = DaicDataset(a.root, train, cache, "train")
    dv = DaicDataset(a.root, dev, cache, "dev")
    te = DaicDataset(a.root, test, cache, "test") if test else None

    tr_dl = DataLoader(tr, a.batch, shuffle=True, collate_fn=collate, drop_last=True)
    dv_dl = DataLoader(dv, a.batch, collate_fn=collate)

    n = np.bincount(tr.y, minlength=2)
    w = torch.tensor(len(tr.y) / (2 * n), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    model = SpeechSNN(n_hidden=a.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    best, best_ep, ckpt = -1, 0, os.path.join(a.root, "best_snn.pt")

    for ep in range(1, a.epochs + 1):
        model.train(); tot, rate = 0.0, 0.0
        for x, y, _ in tr_dl:
            x, y = x.to(device), y.to(device)
            spk = model(x)
            loss = loss_fn(spk.sum(0), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); rate += spk.mean().item()
        m = evaluate(model, dv_dl, device)
        flag = ""
        if m["f1"] > best:
            best, best_ep, flag = m["f1"], ep, " *"
            torch.save(model.state_dict(), ckpt)
        print(f"ep {ep:2d} loss {tot/len(tr_dl):.3f} firing {rate/len(tr_dl):.3f} "
              f"| DEV BA {m['bal_acc']:.2f} F1 {m['f1']:.2f} F1(dep) {m['f1_dep']:.2f}{flag}")

    model.load_state_dict(torch.load(ckpt))
    m = evaluate(model, dv_dl, device)
    print(f"\nBest epoch {best_ep}. DEV (participant level): BA {m['bal_acc']:.3f} "
          f"macro-F1 {m['f1']:.3f} F1(dep) {m['f1_dep']:.3f}\n"
          f"confusion [rows=true 0/1, cols=pred 0/1]\n{m['cm']}")
    if te and len(te):
        m = evaluate(model, DataLoader(te, a.batch, collate_fn=collate), device)
        print(f"TEST: BA {m['bal_acc']:.3f} macro-F1 {m['f1']:.3f} "
              f"F1(dep) {m['f1_dep']:.3f}\n{m['cm']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Depression/daicwoz")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    main(ap.parse_args())
