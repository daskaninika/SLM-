# Speaker Notes: Explaining `merges.txt`

Use these notes when presenting or explaining the tokenizer’s **merges.txt** file to your audience.

---

## 1. What is merges.txt?

- **One-line summary:**  
  `merges.txt` is the list of **merge rules** for the **Byte-Level BPE (Byte-Pair Encoding)** tokenizer. It defines, in order, which pairs of subword units get merged into a single token during tokenization.

- **Where it lives:**  
  `artifacts/tokenizer/merges.txt` — created when you run `train_tokenizer.py`.

- **How it’s used:**  
  The tokenizer is loaded from either:
  - `tokenizer.json` (single file), or  
  - `vocab.json` + `merges.txt` (two-file format).  
  In this project, `generate.py` and `prepare_data.py` can load from `vocab.json` + `merges.txt` when the single-file tokenizer isn’t present.

---

## 2. File structure

- **First line:**  
  `#version: 0.2` — format version for the merges file (used by the HuggingFace `tokenizers` library).

- **Remaining lines:**  
  One merge per line: **two tokens separated by a space**.  
  Format: `left right`  
  Meaning: “Merge the token `left` with the token `right` to form one new token.”

- **Order matters:**  
  Merge #1 is applied first during training, then #2, and so on. The same order is used at inference when encoding text.

- **Approximate size:**  
  Roughly 7,700+ merge rules. Together with the initial byte/symbol vocabulary, these merges produce the final vocabulary (target size 8,000 for this project).

---

## 3. What the symbols mean (especially Ġ)

- **Ġ (U+0120):**  
  Special character used to represent **a space** before a word. So:
  - `Ġ` + `t` → token for “ t” (space + t)
  - `Ġ` + `the` → token for “ the”
  - `Ġt` + `he` → “ the” (if that merge exists)

- **Why “Ġ”?**  
  GPT-2–style BPE tokenizers use this symbol so that space is explicit in the vocabulary and merges. Byte-level BPE still works on raw bytes; the library uses this representation when writing merges.

- **Other symbols:**  
  You may see:
  - Normal letters and digits (`a`, `t`, `1`, `0`).
  - Pairs that form words or subwords: `t he`, `e d`, `in g`, `re ss`, `ab ric`, `net work`, etc.
  - Punctuation and special Unicode (e.g. `Â`, `ł`) when they appeared in the training data (e.g. Cisco docs, Core42 SDD).

---

## 4. How to “read” a few example lines

- **`Ġ t`**  
  “Merge space with the character ‘t’.”  
  So we get a single token for “ t” (space + t).

- **`Ġt he`**  
  “Merge the token for ‘ t’ with the token for ‘he’.”  
  Produces a token for “ the”.

- **`Ġ in`**  
  “Merge space with the token ‘in’.”  
  Produces a token for “ in”.

- **`etw ork`**  
  “Merge ‘etw’ and ‘ork’.”  
  Part of building tokens like “network” from subwords.

- **`fab ric`**, **`Ġ rout`**, **`Ġs cheduling`**  
  Domain-style subwords (fabric, routing, scheduling) reflecting the Cisco/networking training data.

You can say: “Each line is a single merge rule. The tokenizer learns these from the training data by repeatedly merging the most frequent pair until we hit the target vocab size.”

---

## 5. How this fits in the pipeline

1. **Training the tokenizer** (`train_tokenizer.py`):  
   The script trains Byte-Level BPE on the corpus (Cisco docs + optional Core42 SDD), then saves:
   - `vocab.json` — map from token string → id  
   - `merges.txt` — ordered list of merge rules  

2. **Loading the tokenizer** (e.g. in `generate.py`, `prepare_data.py`):  
   Code loads either `tokenizer.json` or `vocab.json` + `merges.txt` and applies the same merge order when encoding/decoding.

3. **Training the model** (`train_model.py`):  
   The model uses the tokenizer’s vocabulary size (from the same tokenizer artifacts). The merges themselves are not passed to the model — only the resulting token IDs.

4. **Generation** (`generate.py`):  
   The same tokenizer (and thus the same `merges.txt` rules) is used to encode the prompt and decode the model’s output token IDs back to text.

---

## 6. Short answers to likely questions

- **“Can we change the merges?”**  
  Yes, but only by re-training the tokenizer (`train_tokenizer.py`) with different data or settings (e.g. vocab size). The merges are learned from the corpus.

- **“Why so many lines?”**  
  We’re targeting ~8,000 vocab size. We start with 256 bytes plus special tokens, then add one new token per merge. So we need thousands of merges to reach that size.

- **“What if we add more training data later?”**  
  Re-run `train_tokenizer.py` so merges (and vocab) reflect the new data. Then re-run data preparation and model training so everything stays aligned.

- **“Is merges.txt sensitive or secret?”**  
  It only describes how subwords are merged; it doesn’t contain raw training text. Treat it like part of your model artifact (e.g. don’t commit secrets in the same repo, but the file itself isn’t a credential).

---

## 7. One-slide / one-minute version

- **What:** Ordered list of BPE merge rules: “merge this pair of tokens next.”
- **Where:** `artifacts/tokenizer/merges.txt`; used with `vocab.json` to load the tokenizer.
- **Format:** First line `#version: 0.2`, then one line per merge: `left right`.
- **Ġ:** Symbol for space; e.g. `Ġ the` is a token for “ the”.
- **Role:** Defines how text is broken into tokens for this project’s SLM; same merges are used for training data preparation, model training, and generation.
