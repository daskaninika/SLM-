r"""
prepare_data.py  –  Tokenize the full corpus and create binary train / val splits
===================================================================================

Reads every .txt file under  data/
Loads the ByteLevelBPE tokenizer from  artifacts/tokenizer/
Encodes all documents → uint16 numpy arrays
Saves:
    artifacts/train.bin          (90 % of tokens)
    artifacts/validation.bin     (10 % of tokens)
    artifacts/data_config.json   (metadata for the training script)

Usage
-----
    cd C:\Users\kanidas\OneDrive - Cisco\Desktop\SLM_wData
    venv\Scripts\activate
    python src/prepare_data.py

Notes
-----
- uint16 supports token-ids 0 – 65 535  (our vocab is 8 000 → safe).
- Documents are shuffled before the split so both sets see a mix of
  Cisco docs and Core42 SDD chunks.
- A separator sequence  <DOC_END> <DOC_START>  is inserted between
  consecutive documents so the model learns document boundaries.
"""

# ── stdlib ──────────────────────────────────────────────────────────
import json
import random
import sys
import time
from pathlib import Path

# ── third-party ─────────────────────────────────────────────────────
import numpy as np
from tokenizers import Tokenizer                    # HuggingFace tokenizers

# ── paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
TOK_DIR      = ARTIFACT_DIR / "tokenizer"

TRAIN_BIN    = ARTIFACT_DIR / "train.bin"
VAL_BIN      = ARTIFACT_DIR / "validation.bin"
CONFIG_JSON  = ARTIFACT_DIR / "data_config.json"

# ── tunables ────────────────────────────────────────────────────────
VALIDATION_FRACTION = 0.10        # 10 % held-out
SEED                = 42
MIN_FILE_WORDS      = 50          # skip tiny / empty files


# =====================================================================
# helpers
# =====================================================================
def load_tokenizer() -> Tokenizer:
    """Load the tokenizer that train_tokenizer.py already built."""
    tok_json = TOK_DIR / "tokenizer.json"          # single-file format
    vocab_json = TOK_DIR / "vocab.json"             # two-file format

    if tok_json.exists():
        print(f"  [tok] Loading single-file tokenizer: {tok_json}")
        return Tokenizer.from_file(str(tok_json))

    if vocab_json.exists():
        # ByteLevelBPETokenizer saves vocab.json + merges.txt
        # We can still wrap it with the fast Tokenizer API
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders
        merges_path = TOK_DIR / "merges.txt"
        print(f"  [tok] Loading two-file tokenizer: vocab.json + merges.txt")
        tok = Tokenizer(models.BPE.from_file(
            str(vocab_json), str(merges_path)
        ))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        return tok

    print("ERROR  Tokenizer not found.  Run  train_tokenizer.py  first.")
    sys.exit(1)


def gather_documents(data_dir: Path) -> list[dict]:
    """Return a list of  {'path': Path, 'text': str}  for every usable .txt file."""
    docs = []
    for fp in sorted(data_dir.glob("*.txt")):
        text = fp.read_text(encoding="utf-8", errors="replace").strip()
        n_words = len(text.split())
        if n_words < MIN_FILE_WORDS:
            print(f"  [skip] {fp.name}  ({n_words} words – below {MIN_FILE_WORDS})")
            continue
        docs.append({"path": fp, "text": text})
    return docs


# =====================================================================
# main pipeline
# =====================================================================
def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  PREPARE DATA  –  tokenize corpus → train.bin / validation.bin")
    print("=" * 70)

    # ── 1. load tokenizer ───────────────────────────────────────────
    print("\n[1/5]  Loading tokenizer …")
    tok = load_tokenizer()
    vocab_size = tok.get_vocab_size()
    print(f"  Vocab size : {vocab_size:,}")

    # make sure special tokens are known
    doc_start_id = tok.token_to_id("<DOC_START>")
    doc_end_id   = tok.token_to_id("<DOC_END>")
    print(f"  <DOC_START> id : {doc_start_id}")
    print(f"  <DOC_END>   id : {doc_end_id}")

    if vocab_size > 65_535:
        print("WARNING  vocab_size > 65535 – uint16 will overflow!  Aborting.")
        sys.exit(1)

    # ── 2. gather docs ──────────────────────────────────────────────
    print("\n[2/5]  Gathering .txt files …")
    docs = gather_documents(DATA_DIR)
    print(f"  Usable documents : {len(docs)}")
    if not docs:
        print("ERROR  No documents found in", DATA_DIR)
        sys.exit(1)

    # ── 3. tokenize every document ──────────────────────────────────
    print("\n[3/5]  Tokenizing …")
    all_ids: list[list[int]] = []
    total_tokens = 0
    for i, doc in enumerate(docs):
        enc = tok.encode(doc["text"])
        ids = enc.ids
        all_ids.append(ids)
        total_tokens += len(ids)
        if (i + 1) % 10 == 0 or (i + 1) == len(docs):
            print(f"    {i+1}/{len(docs)} files tokenized  "
                  f"({total_tokens:,} tokens so far)")

    print(f"  Total tokens across {len(docs)} docs : {total_tokens:,}")

    # ── 4. shuffle & split ──────────────────────────────────────────
    print(f"\n[4/5]  Shuffling docs & splitting "
          f"({100*(1-VALIDATION_FRACTION):.0f}/{100*VALIDATION_FRACTION:.0f}) …")
    random.seed(SEED)
    indices = list(range(len(all_ids)))
    random.shuffle(indices)

    n_val = max(1, int(len(indices) * VALIDATION_FRACTION))
    val_indices   = set(indices[:n_val])
    train_indices = set(indices[n_val:])

    # build contiguous token streams with <DOC_END><DOC_START> separators
    def merge_docs(idx_set: set) -> np.ndarray:
        """Concatenate documents; insert boundary tokens between them."""
        merged: list[int] = []
        for idx in indices:                       # keep shuffled order
            if idx not in idx_set:
                continue
            if merged:                            # separator between docs
                if doc_end_id is not None:
                    merged.append(doc_end_id)
                if doc_start_id is not None:
                    merged.append(doc_start_id)
            merged.extend(all_ids[idx])
        return np.array(merged, dtype=np.uint16)

    train_arr = merge_docs(train_indices)
    val_arr   = merge_docs(val_indices)

    print(f"  Train tokens      : {len(train_arr):>10,}")
    print(f"  Validation tokens : {len(val_arr):>10,}")
    print(f"  Train docs        : {len(train_indices)}")
    print(f"  Validation docs   : {n_val}")

    # ── 5. save ─────────────────────────────────────────────────────
    print("\n[5/5]  Saving binary files …")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_arr.tofile(str(TRAIN_BIN))
    val_arr.tofile(str(VAL_BIN))
    print(f"  {TRAIN_BIN}          ({TRAIN_BIN.stat().st_size / 1024:.1f} KB)")
    print(f"  {VAL_BIN}     ({VAL_BIN.stat().st_size / 1024:.1f} KB)")

    # metadata for the training script
    config = {
        "vocab_size"       : vocab_size,
        "train_tokens"     : int(len(train_arr)),
        "val_tokens"       : int(len(val_arr)),
        "total_tokens"     : int(len(train_arr) + len(val_arr)),
        "train_docs"       : len(train_indices),
        "val_docs"         : n_val,
        "total_docs"       : len(docs),
        "validation_frac"  : VALIDATION_FRACTION,
        "seed"             : SEED,
        "dtype"            : "uint16",
        "train_bin"        : str(TRAIN_BIN),
        "val_bin"          : str(VAL_BIN),
        "tokenizer_dir"    : str(TOK_DIR),
        "doc_start_id"     : doc_start_id,
        "doc_end_id"       : doc_end_id,
    }
    CONFIG_JSON.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"  {CONFIG_JSON}")

    elapsed = time.time() - t0
    print(f"\n✅  Done in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()