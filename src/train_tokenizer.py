r"""
Train Tokenizer - SLM with Real Data (Phase 2)
================================================
Project: SLM_wData
Location: C:\Users\kanidas\OneDrive - Cisco\Desktop\SLM_wData\src\train_tokenizer.py

Trains a ByteLevelBPE tokenizer on the full corpus:
  - 35 scraped Cisco documentation files
  - Core42 SDD processed chunks (if available)

Target vocab size: 8,000
Special tokens: <PAD>, <UNK>, <DOC_START>, <DOC_END>, <SECTION>, <|endoftext|>

Usage:
  cd C:\Users\kanidas\OneDrive - Cisco\Desktop\SLM_wData
  venv\Scripts\activate
  python src\train_tokenizer.py
"""

import os
import sys
import json
import glob
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TOKENIZER_DIR = ARTIFACTS_DIR / "tokenizer"

VOCAB_SIZE = 8000
MIN_FREQUENCY = 2

SPECIAL_TOKENS = [
    "<PAD>",          # 0 - padding
    "<UNK>",          # 1 - unknown
    "<DOC_START>",    # 2 - document start
    "<DOC_END>",      # 3 - document end
    "<SECTION>",      # 4 - section marker
    "<|endoftext|>",  # 5 - end of text
]


def collect_training_files():
    """Gather all .txt files from the data directory for tokenizer training."""

    training_files = []
    total_words = 0

    print(f"\n{'='*60}")
    print(f"  COLLECTING TRAINING FILES")
    print(f"{'='*60}")
    print(f"  Data directory: {DATA_DIR}\n")

    if not DATA_DIR.exists():
        print(f"  [ERROR] Data directory not found: {DATA_DIR}")
        sys.exit(1)

    # ── 1. Scraped Cisco docs ──────────────────────────────────────────
    cisco_files = sorted(glob.glob(str(DATA_DIR / "cisco_doc_*.txt")))
    print(f"  Found {len(cisco_files)} scraped Cisco doc files")
    for f in cisco_files:
        training_files.append(f)
        with open(f, "r", encoding="utf-8") as fh:
            words = len(fh.read().split())
            total_words += words

    # ── 2. Core42 processed file (if exists) ───────────────────────────
    core42_processed = DATA_DIR / "core42_sdd_processed.txt"
    if core42_processed.exists():
        training_files.append(str(core42_processed))
        with open(core42_processed, "r", encoding="utf-8") as fh:
            words = len(fh.read().split())
            total_words += words
        print(f"  Found Core42 processed SDD ({words:,} words)")
    else:
        print(f"  [INFO] Core42 processed file not found (optional)")
        print(f"         Expected at: {core42_processed}")

    # ── 3. Core42 chunk files (if exist) ───────────────────────────────
    chunk_dir = DATA_DIR / "core42_chunks"
    if chunk_dir.exists():
        chunk_files = sorted(glob.glob(str(chunk_dir / "*.txt")))
        print(f"  Found {len(chunk_files)} Core42 chunk files")
        for f in chunk_files:
            training_files.append(f)
            with open(f, "r", encoding="utf-8") as fh:
                words = len(fh.read().split())
                total_words += words
    else:
        print(f"  [INFO] Core42 chunks directory not found (optional)")

    # ── 4. Core42 section files (if exist) ─────────────────────────────
    section_dir = DATA_DIR / "core42_sections"
    if section_dir.exists():
        section_files = sorted(glob.glob(str(section_dir / "*.txt")))
        print(f"  Found {len(section_files)} Core42 section files")
        for f in section_files:
            training_files.append(f)
            with open(f, "r", encoding="utf-8") as fh:
                words = len(fh.read().split())
                total_words += words
    else:
        print(f"  [INFO] Core42 sections directory not found (optional)")

    print(f"\n  ── Summary ──")
    print(f"  Total training files : {len(training_files)}")
    print(f"  Total words (approx) : {total_words:,}")
    print(f"{'='*60}\n")

    if len(training_files) == 0:
        print("  [ERROR] No training files found! Run scrape_cisco_docs.py first.")
        sys.exit(1)

    return training_files, total_words


def train_tokenizer(training_files, total_words):
    """Train a ByteLevelBPE tokenizer on the collected files."""

    print(f"{'='*60}")
    print(f"  TRAINING BYTE-LEVEL BPE TOKENIZER")
    print(f"{'='*60}")
    print(f"  Vocab size target  : {VOCAB_SIZE:,}")
    print(f"  Min frequency      : {MIN_FREQUENCY}")
    print(f"  Special tokens     : {SPECIAL_TOKENS}")
    print(f"  Number of files    : {len(training_files)}")
    print(f"  Output directory   : {TOKENIZER_DIR}")
    print()

    # Create output directory
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Train
    print("  Training tokenizer... (this may take 1-5 minutes on CPU)")
    tokenizer.train(
        files=training_files,
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # Save tokenizer files
    tokenizer.save_model(str(TOKENIZER_DIR))
    print(f"\n  Tokenizer saved to: {TOKENIZER_DIR}")
    print(f"  Files created:")
    print(f"    - {TOKENIZER_DIR / 'vocab.json'}")
    print(f"    - {TOKENIZER_DIR / 'merges.txt'}")

    return tokenizer


def verify_tokenizer(tokenizer):
    """Run verification tests on the trained tokenizer."""

    print(f"\n{'='*60}")
    print(f"  TOKENIZER VERIFICATION")
    print(f"{'='*60}")

    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"\n  Final vocab size: {actual_vocab_size:,}")

    # Verify special tokens
    print(f"\n  Special Token IDs:")
    vocab = tokenizer.get_vocab()
    for token in SPECIAL_TOKENS:
        token_id = vocab.get(token, "NOT FOUND")
        print(f"    {token:20s} → {token_id}")

    # Test encoding/decoding with domain-specific text
    test_strings = [
        "<DOC_START> VXLAN BGP EVPN Multi-Site Design <DOC_END>",
        "interface nve1\n  source-interface loopback1\n  host-reachability protocol bgp",
        "The spine switches use eBGP to establish underlay routing with leaf nodes.",
        "IP address: 10.250.0.1/30 is assigned to the BGW uplink interface.",
        "<SECTION> Section 6: VXLAN BGP EVPN Overlay Design",
        "router bgp 65001\n  address-family l2vpn evpn\n    retain route-target all",
    ]

    print(f"\n  Encoding/Decoding Tests:")
    print(f"  {'-'*56}")
    for text in test_strings:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        num_tokens = len(encoded.ids)
        print(f"\n  Input  : {text[:70]}{'...' if len(text) > 70 else ''}")
        print(f"  Tokens : {num_tokens}")
        print(f"  IDs    : {encoded.ids[:15]}{'...' if num_tokens > 15 else ''}")
        print(f"  Decoded: {decoded[:70]}{'...' if len(decoded) > 70 else ''}")

    print(f"\n{'='*60}")
    print(f"  TOKENIZER TRAINING COMPLETE")
    print(f"{'='*60}\n")

    return actual_vocab_size


def save_tokenizer_config(actual_vocab_size, total_words, num_files):
    """Save tokenizer configuration metadata for later use."""

    config = {
        "vocab_size": actual_vocab_size,
        "min_frequency": MIN_FREQUENCY,
        "special_tokens": SPECIAL_TOKENS,
        "special_token_ids": {token: i for i, token in enumerate(SPECIAL_TOKENS)},
        "training_files_count": num_files,
        "training_words_approx": total_words,
        "tokenizer_type": "ByteLevelBPE",
        "tokenizer_dir": str(TOKENIZER_DIR),
    }

    config_path = TOKENIZER_DIR / "tokenizer_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved to: {config_path}\n")

    return config


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SLM TOKENIZER TRAINER (Phase 2 - Real Data)")
    print("=" * 60)

    # Step 1: Collect training files
    training_files, total_words = collect_training_files()

    # Step 2: Train tokenizer
    tokenizer = train_tokenizer(training_files, total_words)

    # Step 3: Verify
    actual_vocab_size = verify_tokenizer(tokenizer)

    # Step 4: Save config
    save_tokenizer_config(actual_vocab_size, total_words, len(training_files))

    print("  All done! Next step: python src\\prepare_data.py\n")