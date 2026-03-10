# Explanation & Speaker Notes: process_core42_sdd.py

Use this for a code walkthrough or presentation of the Core42 SDD processor.

---

## What This Script Does (Summary)

**process_core42_sdd.py** turns a single **raw text file** (the Core42 DCN Private Cloud Service Design Document, SDD) into **training-ready text** for the SLM. The raw file is typically copy-pasted from a PDF or Word export. The script:

1. **Cleans** the text (page numbers, headers/footers, extra whitespace).
2. **Detects** the 13 major sections using regex patterns (e.g. "1 Introduction", "5 VXLAN Fabric Basics").
3. **Splits** the document by those section boundaries.
4. **Tags** each section with XML-style markers (`<SECTION_5_START>`, `<SECTION_NAME>`, `<PRIMARY_SECTION>`, etc.) so the model can learn structure.
5. **Writes**:
   - One **combined** file: `data/core42_sdd_processed.txt` (full doc with tags).
   - **Per-section** files: `data/core42_sections/section_01_Introduction.txt`, etc.
   - **Chunked** files for long sections (5–8): `data/core42_chunks/chunk_s05_p00.txt`, etc., with ~1500 words per chunk and 20% overlap.

Sections **5–8** (VXLAN Fabric Basics, VXLAN BGP EVPN, Multi-Site VxLAN EVPN, Logical Design: External Connectivity) are marked as **primary** training targets and are the ones that get chunked for training.

---

## Speaker Notes: Code Walkthrough

### 1. Opening Slide / Intro

**Say:**  
"This script processes the Core42 DCN Private Cloud Service Design Document—a single long text file—into section-tagged, cleaned text for our SLM. It doesn’t fetch anything from the web; it assumes you’ve already saved the SDD content into `data/core42_sdd_raw.txt`. It then finds the 13 major sections, tags them for training, and writes a combined file, individual section files, and chunked files for the four primary sections."

**One-liner:**  
"Takes raw Core42 SDD text, cleans it, splits by section, tags sections for the model, and writes combined + per-section + chunked outputs."

---

### 2. Paths and Input/Output (Lines 23–27)

**Say:**  
"Paths are set relative to the project root. We read from `data/core42_sdd_raw.txt` and write to `data/core42_sdd_processed.txt`, `data/core42_sections/`, and `data/core42_chunks/`. If the raw file is missing, we exit with a clear message telling the user where to put it."

| Variable       | Purpose |
|----------------|---------|
| `BASE_DIR`     | Project root (parent of `src/`). |
| `RAW_FILE`     | Input: raw pasted SDD text. |
| `OUTPUT_FILE`  | Output: single combined, tagged file. |
| `SECTIONS_DIR` | Output: one file per section. |

---

### 3. Section Definitions (Lines 29–64)

**Say:**  
"The SDD has 13 numbered sections. We define two things: **regex patterns** to find where each section starts in the text, and a **name map** for tagging. The patterns match lines like '1 Introduction', '5 VXLAN Fabric Basics'. Section names in the map are normalized (e.g. 'Solution_Overview', 'VXLAN_BGP_EVPN') for use in filenames and tags."

**Point out:**  
- `SECTION_PATTERNS`: list of `(section_number, regex)`. Order in the list doesn’t control document order—we sort by position in the text later.
- `SECTION_NAMES`: section number → canonical name (used in tags and filenames). Note: the numeric order in the document (1–13) doesn’t have to match the names 1:1; the names reflect the actual SDD structure.
- **PRIMARY_SECTIONS = {5, 6, 7, 8}**: these are the sections we care most about for training; they get `<PRIMARY_SECTION>` tags and are the only ones we chunk.

---

### 4. clean_text() (Lines 67–93)

**Say:**  
"Raw pasted text from a PDF often has page headers, footers, and messy whitespace. `clean_text()` removes those and normalizes the rest."

**Walk through:**
- **PDF artifacts:** Strip "Page X of Y", "Cisco Confidential", "Core42 DCN Private Cloud SDD v1.1"-style headers.
- **Whitespace:** Tabs → spaces; collapse 3+ newlines to 2; remove trailing spaces on lines; remove lines that are only long runs of dashes/underscores/equals (separators).

**If asked:**  
"We use regex only on known boilerplate patterns; we don’t execute or interpret user content. The input is our own SDD copy, not arbitrary user input."

---

### 5. find_section_boundaries() (Lines 96–114)

**Say:**  
"We need to know where each of the 13 sections starts in the cleaned text. For each section we run a regex that looks for the section heading at the start of a line (with optional leading newline). We record section number, start character position, and the matched heading text. We then sort by position so boundaries are in document order."

**Code highlight:**  
- Pattern is `(?:^|\n)\s*` + the section pattern, with `re.MULTILINE` so `^` matches line start.
- We print `[FOUND]` or `[MISS]` for each section so a run shows which sections were detected; missing sections suggest the raw document format changed.

---

### 6. extract_sections() (Lines 117–141)

**Say:**  
"Given the sorted list of (section_num, start_pos, heading), we slice the text: each section runs from its start up to (but not including) the next section’s start. The last section runs to end of document. We return a dict mapping section number to its full text and print word/line counts per section."

---

### 7. tag_section() (Lines 144–165)

**Say:**  
"Each section is wrapped in training tags so the model sees structure. We emit `<SECTION_N_START>`, `<SECTION_NAME>name</SECTION_NAME>`, then optionally `<PRIMARY_SECTION>` for sections 5–8, then the section text, then `</PRIMARY_SECTION>` if primary, then `<SECTION_N_END>`. Downstream training or inference can use these tags to know which part of the document they’re in."

---

### 8. create_training_chunks() (Lines 168–218)

**Say:**  
"Sections 5–8 can be very long. We split them into overlapping chunks of about `chunk_size` words so we don’t exceed context limits and we keep some overlap for continuity. Default is 2000 words per chunk with 20% overlap. Each chunk gets a short header like '[Section 5: VXLAN_Fabric_Basics - Part 1]' so the model retains section context. We return a list of dicts with section, name, chunk_id, text, and word count."

**Code highlight:**  
- `start` advances by `chunk_size - overlap` so chunks overlap.
- The `if start >= len(words) - overlap: break` avoids infinite loops when the section is short or overlap is large.

---

### 9. main() — Step-by-Step (Lines 221–337)

**Say:**  
"main() runs a seven-step pipeline."

| Step | What happens | What to say |
|------|-----------------------------|-------------|
| **1** | Check `RAW_FILE` exists | "If the raw file is missing, we print where to put it and exit. No output is written." |
| **2** | Load raw text, clean it | "We read UTF-8 with error replacement, then run clean_text() and report size before/after." |
| **3** | Find section boundaries | "We run find_section_boundaries() on the cleaned text. If zero sections are found, we fall back to saving the whole document as one file with generic DOC_START/DOC_TYPE/CUSTOMER/SOLUTION tags and exit." |
| **4** | Extract sections | "extract_sections() slices the text using the boundaries and returns the dict of section_number → text." |
| **5** | Save per-section files | "We write each section to data/core42_sections/section_NN_Name.txt. We print which are primary (★)." |
| **6** | Build combined training file | "We build one big string: DOC_START, DOC_TYPE, CUSTOMER, SOLUTION, TECHNOLOGIES, DEVICES, then each section in order with tag_section(), then DOC_END. We write that to core42_sdd_processed.txt." |
| **7** | Create chunks for primary sections | "We take only sections 5–8, run create_training_chunks() with chunk_size=1500, and write each chunk to data/core42_chunks/ with DOC_START, SECTION, chunk text, DOC_END. Then we print summary and the next step (train_tokenizer)." |

---

### 10. Fallback When No Sections Found (Lines 250–264)

**Say:**  
"If the document format doesn’t match our patterns—for example a different heading style or language—we don’t crash. We save the entire cleaned document as one file with high-level tags (DOC_TYPE, CUSTOMER, SOLUTION) so we still have usable training data. The user can then adjust SECTION_PATTERNS and re-run."

---

### 11. Outputs Summary

**Say:**  
"After a successful run you get: (1) one combined file with all sections tagged; (2) 13 section files for inspection or selective use; (3) chunk files only for sections 5–8, for training on VXLAN/EVPN/Multi-Site/External Connectivity content. The script ends by suggesting the next step: run train_tokenizer."

| Output | Path | Purpose |
|--------|------|---------|
| Combined | `data/core42_sdd_processed.txt` | Full SDD with section and primary tags for training. |
| Sections | `data/core42_sections/section_XX_*.txt` | One file per section (no chunking). |
| Chunks | `data/core42_chunks/chunk_s*.txt` | Chunked text for primary sections only. |

---

### 12. Security / Best Practices (If Asked)

- **Input:** The script reads one fixed path; no unsanitized user path. File content is treated as text, not code.
- **Regex:** Used only for section detection and cleaning; no `eval` or execution of document content.
- **Output:** Writes only under project `data/`; paths are built from `BASE_DIR`, no user-controlled paths.
- **Encoding:** Read/write with UTF-8; `errors='replace'` on read avoids crashes on bad bytes.

---

### 13. Quick Reference: Key Functions

| Function | Purpose |
|----------|---------|
| `clean_text(text)` | Remove PDF artifacts and normalize whitespace. |
| `find_section_boundaries(text)` | Find start position of each of the 13 sections; returns sorted list of (num, pos, heading). |
| `extract_sections(text, boundaries)` | Slice text by boundaries; return {section_num: text}. |
| `tag_section(sec_num, text, is_primary)` | Wrap section in `<SECTION_N_*>` and optional `<PRIMARY_SECTION>`. |
| `create_training_chunks(sections, chunk_size)` | Split long sections into overlapping word chunks with context headers. |
| `main()` | Run the full pipeline and write all outputs. |

---

### 14. Closing / Pipeline Context

**Say:**  
"This script sits after scrape_cisco_docs—which fills data/ with Cisco whitepapers—and before prepare_data and train_tokenizer. The Core42 SDD is a separate, high-value document we process by section so the model can learn both from many Cisco docs and from this one structured design document. The section and primary tags give the model explicit structure it can use during generation or retrieval."

---

*Adjust wording for your audience; use the tables for quick reference during the walkthrough.*
