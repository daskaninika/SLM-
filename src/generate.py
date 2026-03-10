"""
generate.py  –  SDD section generator using the trained SLM
============================================================
Uses the EXACT same model architecture classes as train_model.py
so that checkpoint weights load correctly.

Usage examples:
    python src/generate.py --prompt "VXLAN BGP EVPN fabric design"
    python src/generate.py --prompt "Multi-Site VXLAN EVPN border gateway" --max_tokens 512
    python src/generate.py --prompt "External connectivity VRF design" --temperature 0.8 --top_k 50
"""

import os
import sys
import json
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─────────────────────────────────────────────────────────
# 1.  MODEL ARCHITECTURE  (identical to train_model.py)
# ─────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Single linear projection for Q, K, V  (same as train_model.py)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Causal mask — upper-triangle = True means "mask out" (same as train_model.py)
        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)                                    # (B, T, 3*C)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                    # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale             # (B, heads, T, T)
        attn = attn.masked_fill(self.mask[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                        # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.proj_drop(self.out_proj(out))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block (same as train_model.py)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SLMTransformer(nn.Module):
    """Decoder-only Transformer language model (same as train_model.py)."""

    def __init__(self, vocab_size: int, d_model: int = 384, n_heads: int = 6,
                 n_layers: int = 6, d_ff: int = 1536, block_size: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, block_size, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (same as train_model.py)
        self.tok_emb.weight = self.head.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"

        tok = self.tok_emb(idx)                               # (B, T, d_model)
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T, d_model)
        x = self.drop(tok + pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)                                 # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ─────────────────────────────────────────────────────────
# 2.  LOADING HELPERS
# ─────────────────────────────────────────────────────────

def load_tokenizer(tokenizer_dir: str):
    """Load the BPE tokenizer (tries single-file first, then two-file fallback)."""
    single_file = os.path.join(tokenizer_dir, "tokenizer.json")
    vocab_file = os.path.join(tokenizer_dir, "vocab.json")
    merges_file = os.path.join(tokenizer_dir, "merges.txt")

    if os.path.exists(single_file):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(single_file)
        print(f"[INFO] Loaded tokenizer from {single_file}")
        return tok

    if os.path.exists(vocab_file) and os.path.exists(merges_file):
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders
        tok = Tokenizer(models.BPE.from_file(vocab_file, merges_file))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        print(f"[INFO] Loaded tokenizer from vocab.json + merges.txt")
        return tok

    raise FileNotFoundError(
        f"No tokenizer files found in {tokenizer_dir}. "
        f"Expected tokenizer.json or (vocab.json + merges.txt)."
    )


def load_model(checkpoint_path: str, config_path: str, device: torch.device):
    """Load model config + trained weights."""
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"[INFO] Model config: {json.dumps(config, indent=2)}")

    # Build model with same hyperparameters
    model = SLMTransformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        block_size=config["block_size"],
        dropout=config.get("dropout", 0.1),
    )

    # Load checkpoint (train_model.py saves under "model_state_dict" key)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", "?")
    print(f"[INFO] Loaded checkpoint from epoch {epoch}, val_loss={val_loss}")
    return model, config


# ─────────────────────────────────────────────────────────
# 3.  TEXT GENERATION
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model: SLMTransformer,
    tokenizer,
    prompt: str,
    max_tokens: int = 300,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    device: torch.device = torch.device("cpu"),
):
    """
    Auto-regressive text generation with temperature, top-k, top-p,
    and repetition penalty.
    """
    model.eval()
    block_size = model.block_size

    # Encode prompt
    encoded = tokenizer.encode(prompt)
    token_ids = encoded.ids
    if len(token_ids) == 0:
        print("[WARN] Prompt encoded to zero tokens. Using <DOC_START> token.")
        doc_start_id = tokenizer.token_to_id("<DOC_START>")
        token_ids = [doc_start_id] if doc_start_id is not None else [0]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Get special token IDs for stop detection
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    doc_end_id = tokenizer.token_to_id("<DOC_END>")
    stop_ids = set()
    if eos_id is not None:
        stop_ids.add(eos_id)
    if doc_end_id is not None:
        stop_ids.add(doc_end_id)

    # Track generated tokens for repetition penalty
    generated_ids = list(token_ids)

    for step in range(max_tokens):
        # Crop to block_size (keep most recent tokens)
        idx_cond = input_ids[:, -block_size:]

        # Forward pass
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # (1, vocab_size) — logits for next token

        # --- Repetition penalty ---
        if repetition_penalty != 1.0:
            for prev_id in set(generated_ids):
                if logits[0, prev_id] > 0:
                    logits[0, prev_id] /= repetition_penalty
                else:
                    logits[0, prev_id] *= repetition_penalty

        # --- Temperature ---
        if temperature > 0:
            logits = logits / temperature
        else:
            # Greedy (temperature=0 means pick the argmax)
            next_id = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            generated_ids.append(next_id.item())
            if next_id.item() in stop_ids:
                break
            continue

        # --- Top-K filtering ---
        if top_k > 0:
            top_k_val = min(top_k, logits.size(-1))
            kth_vals = torch.topk(logits, top_k_val, dim=-1).values[:, -1:]
            logits[logits < kth_vals] = float("-inf")

        # --- Top-P (nucleus) filtering ---
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above top_p
            remove_mask = cumulative_probs > top_p
            # Keep at least one token
            remove_mask[:, 0] = False
            # Scatter back to original indices
            indices_to_remove = sorted_indices[remove_mask]
            logits[0, indices_to_remove] = float("-inf")

        # --- Sample ---
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        generated_ids.append(next_id.item())

        # --- Stop check ---
        if next_id.item() in stop_ids:
            break

    # Decode all generated tokens
    output_text = tokenizer.decode(generated_ids)
    return output_text


# ─────────────────────────────────────────────────────────
# 4.  MAIN  (CLI entry point)
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate SDD text using trained SLM")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt to seed the generation")
    parser.add_argument("--max_tokens", type=int, default=300,
                        help="Maximum number of tokens to generate (default: 300)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7, 0=greedy)")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-K filtering (default: 40, 0=disabled)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-P nucleus sampling (default: 0.9)")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Repetition penalty (default: 1.2, 1.0=disabled)")
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="artifacts/checkpoints/model_config.json",
                        help="Path to model config JSON")
    parser.add_argument("--tokenizer_dir", type=str, default="artifacts/tokenizer",
                        help="Directory containing tokenizer files")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate (default: 1)")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_dir)
    print(f"[INFO] Tokenizer vocab size: {tokenizer.get_vocab_size()}")

    # Load model
    model, config = load_model(args.checkpoint, args.config, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {n_params:,}")

    # Generate
    print(f"\n{'='*70}")
    print(f"Prompt: {args.prompt}")
    print(f"Settings: temp={args.temperature}, top_k={args.top_k}, "
          f"top_p={args.top_p}, rep_penalty={args.repetition_penalty}, "
          f"max_tokens={args.max_tokens}")
    print(f"{'='*70}\n")

    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n--- Sample {i+1}/{args.num_samples} ---")
        
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
        )
        print(output)
        print()

    print(f"{'='*70}")
    print("[DONE] Generation complete.")


if __name__ == "__main__":
    main()