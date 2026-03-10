# Speaker Notes: Explaining `generate.py`

Use these notes when presenting or walking through `src/generate.py` (SDD section generator using the trained SLM).

---

## 1. Opening / Purpose

**Say:**  
"`generate.py` is the inference script for our SDD (Solution Design Document) small language model. It loads the trained checkpoint and tokenizer, then produces new text from a prompt—exactly the same decoder-only transformer we trained in `train_model.py`."

**Key point:**  
- Same model architecture as training so checkpoint weights load correctly. We don’t import from `train_model.py` to keep this script self-contained and runnable without the full training stack.

---

## 2. Why the Model Code Is Duplicated (Lines 24–136)

**Say:**  
"The file defines the same model classes as `train_model.py`: `CausalSelfAttention`, `TransformerBlock`, and `SLMTransformer`. We do this on purpose so that the state dict keys and shapes match the saved checkpoint exactly. If we changed layer names or dimensions here, loading would fail or give wrong results."

**If asked:**  
- "We could refactor to a shared `model.py` and import in both train and generate; the current duplication is for clarity and to keep generate runnable with minimal dependencies."

---

## 3. CausalSelfAttention (Lines 27–61)

**Say:**  
"This is multi-head causal self-attention. The important part for generation is the causal mask: each position can only attend to itself and previous positions, so at inference we get proper left-to-right autoregressive behavior. The mask is registered as a buffer so it moves with the model to the right device (CPU/GPU)."

**Optional:**  
- "Q, K, V come from one linear layer and are split; we scale by the square root of head dimension before softmax to avoid saturation."

---

## 4. TransformerBlock and SLMTransformer (Lines 64–136)

**Say:**  
"Each block is pre-norm: LayerNorm, then attention, then residual; same for the FFN. The full model stacks these blocks, adds token and position embeddings, and has a final linear head that predicts the next token. We use weight tying: the embedding and output projection share weights, which is standard in small LMs and reduces parameters."

**Optional:**  
- "`forward` returns logits and optionally loss; at inference we only use logits for the last position to predict the next token."

---

## 5. Loading the Tokenizer (Lines 144–166)

**Say:**  
"Before we can run the model, we need the same tokenizer used during training. `load_tokenizer` looks for either a single `tokenizer.json` (Hugging Face tokenizers format) or the older pair `vocab.json` and `merges.txt`. That way we support both export formats from our tokenizer training step."

**If asked:**  
- "ByteLevel pre-tokenizer and decoder keep us byte-based and avoid unknown tokens for arbitrary text."

---

## 6. Loading the Model and Config (Lines 169–196)

**Say:**  
"We load two things: the JSON config (vocab size, d_model, layers, etc.) and the checkpoint. We build an `SLMTransformer` using the config so the architecture matches training, then load the state dict from the checkpoint. We use `map_location=device` so we can load a GPU-trained checkpoint onto CPU if needed. After loading we call `model.eval()` so dropout and other training-only behavior are disabled."

**Key point:**  
- "The checkpoint is expected to have a `model_state_dict` key; that’s what `train_model.py` saves. We also log epoch and validation loss if present."

---

## 7. The Generation Loop (Lines 204–297)

**Say:**  
"Generation is autoregressive: we start from the encoded prompt and repeatedly predict the next token until we hit max_tokens or a stop token."

**Walk through step by step:**

1. **Encode prompt**  
   "We encode the prompt to token IDs. If we get zero tokens—e.g. empty or out-of-vocab—we fall back to a `<DOC_START>` token if the tokenizer has one, otherwise token 0."

2. **Stop tokens**  
   "We detect end-of-sequence with special tokens: `<|endoftext|>` and `<DOC_END>`. When we sample one of these, we stop."

3. **Context window**  
   "Each step we only feed the last `block_size` tokens (e.g. 256) into the model. So we always pass a fixed-size context; older tokens are dropped. That matches how we trained."

4. **Repetition penalty**  
   "We downweight logits for tokens that have already appeared in the sequence. That reduces repetitive phrases. The penalty is applied by dividing (or multiplying if logit is negative) by `repetition_penalty`."

5. **Temperature**  
   "We divide logits by temperature. High temperature (e.g. 0.9) makes the distribution softer and more random; low (e.g. 0.3) makes it peakier and more deterministic. Temperature 0 means we take the argmax (greedy) and skip sampling."

6. **Top-K**  
   "We zero out logits for all but the top K tokens, then re-normalize. That cuts long-tail noise and focuses on plausible next tokens."

7. **Top-P (nucleus)**  
   "We sort by probability, take the smallest set of tokens whose cumulative probability reaches `top_p`, and zero out the rest. So we dynamically choose how many tokens to keep based on the distribution."

8. **Sample**  
   "We turn the filtered logits into probabilities with softmax and sample one token with `torch.multinomial`. That token is appended to the sequence and we repeat."

**Summary:**  
"So in one sentence: we repeatedly predict the next token with the same model we trained, using temperature and top-k/top-p for diversity and repetition penalty to avoid loops, and we stop at special tokens or max length."

---

## 8. CLI and Defaults (Lines 304–371)

**Say:**  
"The script is driven by the command line. Required: `--prompt`. Everything else has defaults: 300 max tokens, temperature 0.7, top_k 40, top_p 0.9, repetition_penalty 1.2. We also allow overriding paths to checkpoint, config, and tokenizer directory, and we can ask for multiple samples with `--num_samples`."

**Example commands to show:**

- `python src/generate.py --prompt "VXLAN BGP EVPN fabric design"`
- `python src/generate.py --prompt "Multi-Site VXLAN EVPN border gateway" --max_tokens 512`
- `python src/generate.py --prompt "External connectivity VRF design" --temperature 0.8 --top_k 50`

**Say:**  
"Device is chosen automatically: CUDA if available, else CPU. We print the chosen device, model size, and generation settings so the user can see what’s being used."

---

## 9. End-to-End Flow (Recap)

**Say:**  
"End to end: we parse args, pick device, load tokenizer and model from artifacts, then for each sample we call `generate()` with the prompt and sampling knobs. The function returns the full string (prompt + generated part); we print it and repeat if `num_samples` > 1. So this script is the main way to try the trained model on new SDD-style prompts."

---

## 10. Possible Questions

| Question | Short answer |
|----------|--------------|
| Can we use GPU? | Yes; if CUDA is available we use it automatically. |
| Why no RAG/retrieval here? | This script is pure LM inference. RAG would be in a separate flow (e.g. `build_rag_index.py` + a wrapper that retrieves then prompts). |
| How do we reduce repetition? | Repetition penalty and/or lower temperature; for more control we could add n-gram blocking. |
| Why weight tying? | Saves parameters and often improves small LMs by sharing representation between input and output. |
| What if the prompt is too long? | We only keep the last `block_size` tokens, so long prompts are truncated from the left; the model never sees the very beginning. |

---

## Quick Reference: Sampling Parameters

| Parameter | Effect | When to change |
|-----------|--------|----------------|
| **temperature** | Higher = more random, lower = more deterministic. 0 = greedy. | Raise for diverse drafts, lower for focused/technical text. |
| **top_k** | Restrict to top K tokens by probability. | Lower (e.g. 20) for tighter text; 0 = no top-k. |
| **top_p** | Nucleus: keep smallest set of tokens that sum to this probability. | Similar to top_k; often use one or the other. |
| **repetition_penalty** | Downweight already-seen tokens. | Increase (e.g. 1.3) if output is repetitive. |
| **max_tokens** | Hard cap on generated length. | Set by desired section length and GPU memory. |

Use these notes to explain `generate.py` from high-level purpose down to sampling behavior and CLI usage.
