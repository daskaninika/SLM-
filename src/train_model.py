r"""
train_model.py  –  Train a decoder-only Transformer SLM on tokenised binary data
=================================================================================
Project : SLM_wData
Location: C:\Users\kanidas\OneDrive - Cisco\Desktop\SLM_wData\src\train_model.py

Reads
-----
  artifacts/train.bin          – uint16 numpy array  (training token IDs)
  artifacts/validation.bin     – uint16 numpy array  (validation token IDs)
  artifacts/tokenizer/         – trained ByteLevelBPE tokenizer

Writes
------
  models/best_model.pt         – best checkpoint (model_state_dict, epoch, val_loss, config)
  models/model_config.json     – architecture config (for generate.py to reload)
  logs/training_log.csv        – per-epoch train_loss, val_loss, lr, elapsed
"""

# ── stdlib ───────────────────────────────────────────────────────────────
import os, sys, json, csv, time, math
from dataclasses import dataclass, asdict

# ── third-party ──────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────
ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS   = os.path.join(ROOT, "artifacts")
MODELS_DIR  = os.path.join(ROOT, "models")
LOGS_DIR    = os.path.join(ROOT, "logs")
TRAIN_BIN   = os.path.join(ARTIFACTS, "train.bin")
VAL_BIN     = os.path.join(ARTIFACTS, "validation.bin")
TOK_DIR     = os.path.join(ARTIFACTS, "tokenizer")


# ╭─────────────────────────────────────────────────────────────╮
# │  HYPERPARAMETERS                                            │
# ╰─────────────────────────────────────────────────────────────╯
@dataclass
class HParams:
    # architecture
    vocab_size:  int   = 8_000
    d_model:     int   = 384
    n_heads:     int   = 6
    n_layers:    int   = 6
    d_ff:        int   = 1_536      # 4 * d_model
    block_size:  int   = 256
    dropout:     float = 0.1
    # training
    batch_size:  int   = 16
    epochs:      int   = 30
    lr:          float = 3e-4
    warmup_pct:  float = 0.05
    patience:    int   = 7
    grad_clip:   float = 1.0
    # data
    stride:      int   = 128        # overlap between windows


# ╭─────────────────────────────────────────────────────────────╮
# │  DATASET                                                    │
# ╰─────────────────────────────────────────────────────────────╯
class TokenDataset(Dataset):
    r"""Sliding-window dataset over a flat uint16 .bin file."""

    def __init__(self, bin_path: str, block_size: int, stride: int):
        raw = np.fromfile(bin_path, dtype=np.uint16)
        self.data = torch.from_numpy(raw.astype(np.int64))   # int64 for embed
        self.block_size = block_size
        self.stride = stride
        # number of windows
        self.n_windows = max(1, (len(self.data) - block_size) // stride)

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        end   = start + self.block_size + 1          # +1 for target shift
        chunk = self.data[start:end]
        x = chunk[:-1]                                # input  tokens
        y = chunk[1:]                                 # target tokens
        return x, y


# ╭─────────────────────────────────────────────────────────────╮
# │  MODEL COMPONENTS                                           │
# ╰─────────────────────────────────────────────────────────────╯
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["d_model"] % cfg["n_heads"] == 0
        self.n_heads  = cfg["n_heads"]
        self.head_dim = cfg["d_model"] // cfg["n_heads"]

        self.qkv  = nn.Linear(cfg["d_model"], 3 * cfg["d_model"])
        self.proj = nn.Linear(cfg["d_model"], cfg["d_model"])
        self.attn_drop = nn.Dropout(cfg["dropout"])
        self.proj_drop = nn.Dropout(cfg["dropout"])

        # causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg["block_size"], cfg["block_size"]))
                 .unsqueeze(0).unsqueeze(0)                       # (1,1,T,T)
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)                                        # (B,T,3C)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg["d_model"])
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg["d_model"])
        self.mlp  = nn.Sequential(
            nn.Linear(cfg["d_model"], cfg["d_ff"]),
            nn.GELU(),
            nn.Linear(cfg["d_ff"], cfg["d_model"]),
            nn.Dropout(cfg["dropout"]),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SLMTransformer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.pos_emb = nn.Embedding(cfg["block_size"],  cfg["d_model"])
        self.drop    = nn.Dropout(cfg["dropout"])
        self.blocks  = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.ln_f    = nn.LayerNorm(cfg["d_model"])
        self.head    = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)

        # weight tying
        self.tok_emb.weight = self.head.weight

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[model] SLMTransformer  –  {n_params/1e6:.2f}M parameters")

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)   # (1, T)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))       # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)                                       # (B, T, V)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss


# ╭─────────────────────────────────────────────────────────────╮
# │  LR SCHEDULE  –  linear warm-up → cosine decay             │
# ╰─────────────────────────────────────────────────────────────╯
def get_lr(step: int, total_steps: int, hp: HParams) -> float:
    warmup_steps = int(total_steps * hp.warmup_pct)
    if step < warmup_steps:
        return hp.lr * (step + 1) / warmup_steps
    decay_ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return hp.lr * max(coeff, 0.1)       # floor at 10 % of peak LR


# ╭─────────────────────────────────────────────────────────────╮
# │  VALIDATION                                                 │
# ╰─────────────────────────────────────────────────────────────╯
@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 50):
    model.eval()
    total_loss, n = 0.0, 0
    for i, (xb, yb) in enumerate(loader):
        if i >= max_batches:
            break
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


# ╭─────────────────────────────────────────────────────────────╮
# │  MAIN TRAINING LOOP                                         │
# ╰─────────────────────────────────────────────────────────────╯
def train():
    hp = HParams()

    # ── check files ──────────────────────────────────────────────
    for fpath, label in [(TRAIN_BIN, "train.bin"), (VAL_BIN, "validation.bin")]:
        if not os.path.isfile(fpath):
            sys.exit(f"[ERROR] {label} not found at {fpath}.  Run prepare_data.py first.")

    # ── read actual vocab size from tokenizer ────────────────────
    tok_json = os.path.join(TOK_DIR, "tokenizer.json")
    if os.path.isfile(tok_json):
        with open(tok_json, "r", encoding="utf-8") as f:
            tok_data = json.load(f)
        actual_vocab = len(tok_data["model"]["vocab"])
        # also count added_tokens (special tokens)
        actual_vocab += len(tok_data.get("added_tokens", []))
        print(f"[data]  tokenizer vocab size = {actual_vocab}")
        hp.vocab_size = actual_vocab
    else:
        print(f"[WARN] tokenizer.json not found; using default vocab_size={hp.vocab_size}")

    # ── device ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")

    # ── datasets / loaders ───────────────────────────────────────
    train_ds = TokenDataset(TRAIN_BIN, hp.block_size, hp.stride)
    val_ds   = TokenDataset(VAL_BIN,   hp.block_size, hp.stride)
    print(f"[data]  train windows = {len(train_ds):,}   val windows = {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True,
                              drop_last=True, num_workers=0, pin_memory=False)       #change acc to VM core
    val_loader   = DataLoader(val_ds,   batch_size=hp.batch_size, shuffle=False,
                              drop_last=False, num_workers=0, pin_memory=False)

    # ── model ────────────────────────────────────────────────────
    cfg = {
        "vocab_size": hp.vocab_size,
        "d_model":    hp.d_model,
        "n_heads":    hp.n_heads,
        "n_layers":   hp.n_layers,
        "d_ff":       hp.d_ff,
        "block_size": hp.block_size,
        "dropout":    hp.dropout,
    }
    model = SLMTransformer(cfg).to(device)

    # ── optimiser ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr, betas=(0.9, 0.95),
                                  weight_decay=0.1)

    total_steps = hp.epochs * len(train_loader)
    global_step = 0
    best_val    = float("inf")
    stall       = 0                          # early-stopping counter

    # ── dirs ─────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)

    log_path = os.path.join(LOGS_DIR, "training_log.csv")
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    csv_log  = csv.writer(log_file)
    csv_log.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_sec"])

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  Training  –  {hp.epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"  Total steps = {total_steps:,}")
    print(f"{'='*60}\n")

    for epoch in range(1, hp.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{hp.epochs:02d}", leave=True)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)

            # set LR for this step
            lr_now = get_lr(global_step, total_steps, hp)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_now:.2e}")

        avg_train = epoch_loss / max(n_batches, 1)
        avg_val   = evaluate(model, val_loader, device)
        elapsed   = time.time() - t0

        csv_log.writerow([epoch, f"{avg_train:.6f}", f"{avg_val:.6f}",
                          f"{lr_now:.2e}", f"{elapsed:.1f}"])
        log_file.flush()

        improved = ""
        if avg_val < best_val:
            best_val = avg_val
            stall = 0
            improved = "  ★ saved"
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch":            epoch,
                "val_loss":         avg_val,
                "config":           cfg,
            }, os.path.join(MODELS_DIR, "best_model.pt"))
            # also save config separately for generate.py
            with open(os.path.join(MODELS_DIR, "model_config.json"), "w") as f:
                json.dump(cfg, f, indent=2)
        else:
            stall += 1

        print(f"  ► Epoch {epoch:02d}  train={avg_train:.4f}  val={avg_val:.4f}  "
              f"lr={lr_now:.2e}  elapsed={elapsed/60:.1f}min{improved}")

        if stall >= hp.patience:
            print(f"\n[early stop] No improvement for {hp.patience} epochs. Stopping.")
            break

    log_file.close()
    print(f"\n{'='*60}")
    print(f"  Training complete.  Best val loss = {best_val:.4f}")
    print(f"  Checkpoint  → {os.path.join(MODELS_DIR, 'best_model.pt')}")
    print(f"  Config      → {os.path.join(MODELS_DIR, 'model_config.json')}")
    print(f"  Log         → {log_path}")
    print(f"  Total time  = {(time.time()-t0)/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()