r"""
process_core42_sdd.py
---------------------
Processes the Core42 DCN Private Cloud SDD document into
section-tagged training format for the SLM.

Input:  data/core42_sdd_raw.txt  (raw pasted SDD text)
Output: data/core42_sdd_processed.txt (section-tagged, cleaned)
        data/core42_sections/section_XX_<name>.txt (individual sections)

Run:
  cd C:\Users\kanidas\OneDrive - Cisco\Desktop\slm_project_v2
  venv\Scripts\activate
  python src/process_core42_sdd.py
"""

import os
import re
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent          # slm_project_v2/
RAW_FILE = BASE_DIR / "data" / "core42_sdd_raw.txt"
OUTPUT_FILE = BASE_DIR / "data" / "core42_sdd_processed.txt"
SECTIONS_DIR = BASE_DIR / "data" / "core42_sections"

# ── Section definitions ────────────────────────────────────────────
# These are the 13 major sections from the Core42 SDD v1.1
# We map section numbers to names so we can tag them properly
SECTION_PATTERNS = [
    (1,  r"1\s+Introduction"),
    (2,  r"2\s+Design\s+Overview"),
    (3,  r"3\s+Physical\s+Connectivity"),
    (4,  r"4\s+VLAN\s*/\s*IP\s*/\s*Naming\s+Conventions"),
    (5,  r"5\s+VXLAN\s+Fabric\s+Basics"),
    (6,  r"6\s+VXLAN\s+BGP\s+EVPN"),
    (7,  r"7\s+Multi-Site\s+VxLAN\s+EVPN"),
    (8,  r"8\s+Logical\s+Design:\s+External\s+Connectivity"),
    (9,  r"9\s+Fabric\s+Traffic\s+Flows"),
    (10, r"10\s+Logical\s+Design:\s+Services"),
    (11, r"11\s+Nexus\s+Dashboard\s+Fabric\s+Controller"),
    (12, r"12\s+Identity\s+Service\s+Engine"),
    (13, r"13\s+Network\s+Management"),
]

SECTION_NAMES = {
    1:  "Introduction",
    2:  "Solution_Overview",
    3:  "Physical_Design",
    4:  "Logical_Design_Underlay",
    5:  "VXLAN_Fabric_Basics",
    6:  "VXLAN_BGP_EVPN",
    7:  "Multi_Site_VXLAN_EVPN",
    8:  "External_Connectivity",
    9:  "Network_Management",
    10: "Network_Security",
    11: "Multicast",
    12: "Quality_of_Service",
    13: "Migration_Plan",
}

# Sections 5-8 are our PRIMARY training targets
PRIMARY_SECTIONS = {5, 6, 7, 8}


def clean_text(text: str) -> str:
    """
    Clean and normalize raw SDD text.
    - Remove excessive blank lines
    - Normalize whitespace
    - Remove page headers/footers
    - Remove figure/table references that are just labels
    """
    # Remove common PDF artifacts
    # "Page X of Y" patterns
    text = re.sub(r'[Pp]age\s+\d+\s+of\s+\d+', '', text)
    # "Cisco Confidential" headers/footers
    text = re.sub(r'Cisco\s+(Confidential|Systems)', '', text, flags=re.IGNORECASE)
    # "Core42" repeated headers
    text = re.sub(r'Core42\s+DCN\s+Private\s+Cloud\s+SDD\s+v[\d.]+', '', text)
    
    # Normalize whitespace
    # Replace tabs with spaces
    text = text.replace('\t', '    ')
    # Collapse 3+ newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove trailing spaces on each line
    text = re.sub(r' +\n', '\n', text)
    # Remove lines that are just dashes or underscores (separators)
    text = re.sub(r'\n[-_=]{5,}\n', '\n', text)
    
    return text.strip()


def find_section_boundaries(text: str):
    """
    Find where each section starts in the text.
    Returns list of (section_number, start_position, match_text).
    """
    boundaries = []
    
    for sec_num, pattern in SECTION_PATTERNS:
        # Search for section heading - try with and without leading newline
        match = re.search(r'(?:^|\n)\s*' + pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            boundaries.append((sec_num, match.start(), match.group().strip()))
            print(f"  [FOUND] Section {sec_num}: '{match.group().strip()[:60]}' at position {match.start()}")
        else:
            print(f"  [MISS]  Section {sec_num}: pattern '{pattern}' not found")
    
    # Sort by position in document
    boundaries.sort(key=lambda x: x[1])
    return boundaries


def extract_sections(text: str, boundaries: list) -> dict:
    """
    Extract text for each section based on boundaries.
    Each section runs from its heading to the start of the next section.
    Returns dict: {section_number: section_text}
    """
    sections = {}
    
    for i, (sec_num, start_pos, heading) in enumerate(boundaries):
        # End position is start of next section, or end of document
        if i + 1 < len(boundaries):
            end_pos = boundaries[i + 1][1]
        else:
            end_pos = len(text)
        
        section_text = text[start_pos:end_pos].strip()
        sections[sec_num] = section_text
        
        # Stats
        word_count = len(section_text.split())
        line_count = len(section_text.split('\n'))
        print(f"  Section {sec_num} ({SECTION_NAMES.get(sec_num, 'Unknown')}): "
              f"{word_count} words, {line_count} lines")
    
    return sections


def tag_section(sec_num: int, text: str, is_primary: bool) -> str:
    """
    Wrap a section in training tags.
    Primary sections (5-8) get extra tags for detailed generation.
    """
    sec_name = SECTION_NAMES.get(sec_num, f"Section_{sec_num}")
    
    tagged = f"<SECTION_{sec_num}_START>\n"
    tagged += f"<SECTION_NAME>{sec_name}</SECTION_NAME>\n"
    
    if is_primary:
        tagged += f"<PRIMARY_SECTION>\n"
    
    tagged += text + "\n"
    
    if is_primary:
        tagged += f"</PRIMARY_SECTION>\n"
    
    tagged += f"<SECTION_{sec_num}_END>\n"
    
    return tagged


def create_training_chunks(sections: dict, chunk_size: int = 2000) -> list:
    """
    For very long sections, split into overlapping chunks
    suitable for training. Each chunk maintains context.
    
    chunk_size: approximate number of words per chunk
    overlap: words of overlap between chunks
    """
    chunks = []
    overlap = chunk_size // 5   # 20% overlap
    
    for sec_num, text in sections.items():
        words = text.split()
        sec_name = SECTION_NAMES.get(sec_num, f"Section_{sec_num}")
        
        if len(words) <= chunk_size:
            # Section fits in one chunk
            chunks.append({
                'section': sec_num,
                'name': sec_name,
                'chunk_id': 0,
                'text': text,
                'words': len(words)
            })
        else:
            # Split into overlapping chunks
            chunk_id = 0
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_text = ' '.join(words[start:end])
                
                # Add section context header to each chunk
                context_header = f"[Section {sec_num}: {sec_name} - Part {chunk_id + 1}]\n"
                
                chunks.append({
                    'section': sec_num,
                    'name': sec_name,
                    'chunk_id': chunk_id,
                    'text': context_header + chunk_text,
                    'words': end - start
                })
                
                chunk_id += 1
                start = end - overlap
                
                # Prevent infinite loop for very small overlaps
                if start >= len(words) - overlap:
                    break
    
    return chunks


def main():
    print("=" * 60)
    print("Core42 SDD Processor")
    print("=" * 60)
    
    # ── Step 1: Check input file exists ────────────────────────
    if not RAW_FILE.exists():
        print(f"\n[ERROR] Raw SDD file not found: {RAW_FILE}")
        print(f"\nPlease save the Core42 SDD text to:")
        print(f"  {RAW_FILE}")
        print(f"\nThen run this script again.")
        sys.exit(1)
    
    print(f"\n[1] Loading raw SDD from: {RAW_FILE}")
    raw_text = RAW_FILE.read_text(encoding='utf-8', errors='replace')
    print(f"    Raw file size: {len(raw_text):,} characters, {len(raw_text.split()):,} words")
    
    # ── Step 2: Clean text ─────────────────────────────────────
    print(f"\n[2] Cleaning text...")
    cleaned = clean_text(raw_text)
    reduction = (1 - len(cleaned) / len(raw_text)) * 100
    print(f"    Cleaned: {len(cleaned):,} characters ({reduction:.1f}% reduction)")
    
    # ── Step 3: Find section boundaries ────────────────────────
    print(f"\n[3] Finding section boundaries...")
    boundaries = find_section_boundaries(cleaned)
    print(f"    Found {len(boundaries)} of {len(SECTION_PATTERNS)} sections")
    
    if len(boundaries) == 0:
        print("\n[WARNING] No sections found! The document format may differ.")
        print("  Saving entire document as one training file...")
        
        # Fallback: save entire cleaned document
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        output_text = "<DOC_START>\n"
        output_text += "<DOC_TYPE>Service_Design_Document</DOC_TYPE>\n"
        output_text += "<CUSTOMER>Core42</CUSTOMER>\n"
        output_text += "<SOLUTION>DCN_Private_Cloud</SOLUTION>\n"
        output_text += cleaned + "\n"
        output_text += "<DOC_END>\n"
        
        OUTPUT_FILE.write_text(output_text, encoding='utf-8')
        print(f"    Saved to: {OUTPUT_FILE}")
        print("    Done (fallback mode).")
        return
    
    # ── Step 4: Extract sections ───────────────────────────────
    print(f"\n[4] Extracting sections...")
    sections = extract_sections(cleaned, boundaries)
    
    # ── Step 5: Save individual section files ──────────────────
    print(f"\n[5] Saving individual section files...")
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    for sec_num, sec_text in sections.items():
        sec_name = SECTION_NAMES.get(sec_num, f"Section_{sec_num}")
        filename = f"section_{sec_num:02d}_{sec_name}.txt"
        filepath = SECTIONS_DIR / filename
        filepath.write_text(sec_text, encoding='utf-8')
        
        is_primary = sec_num in PRIMARY_SECTIONS
        marker = " ★ PRIMARY" if is_primary else ""
        print(f"    Saved: {filename} ({len(sec_text.split())} words){marker}")
    
    # ── Step 6: Create combined training file ──────────────────
    print(f"\n[6] Creating combined training file...")
    
    combined = "<DOC_START>\n"
    combined += "<DOC_TYPE>Service_Design_Document</DOC_TYPE>\n"
    combined += "<CUSTOMER>Core42</CUSTOMER>\n"
    combined += "<SOLUTION>DCN_Private_Cloud</SOLUTION>\n"
    combined += "<TECHNOLOGIES>VXLAN,BGP_EVPN,Multi_Site,NDFC,VRF_Lite,vPC</TECHNOLOGIES>\n"
    combined += "<DEVICES>Nexus_9364D_GX2A,Nexus_9332D_GX2B,Nexus_93180YC_FX3B,Cat_9500,Cat_9300</DEVICES>\n\n"
    
    for sec_num in sorted(sections.keys()):
        is_primary = sec_num in PRIMARY_SECTIONS
        tagged = tag_section(sec_num, sections[sec_num], is_primary)
        combined += tagged + "\n"
    
    combined += "<DOC_END>\n"
    
    OUTPUT_FILE.write_text(combined, encoding='utf-8')
    print(f"    Saved: {OUTPUT_FILE}")
    print(f"    Total: {len(combined):,} characters, {len(combined.split()):,} words")
    
    # ── Step 7: Create training chunks for primary sections ────
    print(f"\n[7] Creating training chunks for primary sections (5-8)...")
    
    primary_sections = {k: v for k, v in sections.items() if k in PRIMARY_SECTIONS}
    chunks = create_training_chunks(primary_sections, chunk_size=1500)
    
    chunks_dir = BASE_DIR / "data" / "core42_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    for chunk in chunks:
        chunk_filename = f"chunk_s{chunk['section']:02d}_p{chunk['chunk_id']:02d}.txt"
        chunk_path = chunks_dir / chunk_filename
        
        chunk_text = f"<DOC_START>\n"
        chunk_text += f"<SECTION>{chunk['name']}</SECTION>\n"
        chunk_text += chunk['text'] + "\n"
        chunk_text += f"<DOC_END>\n"
        
        chunk_path.write_text(chunk_text, encoding='utf-8')
    
    print(f"    Created {len(chunks)} training chunks in {chunks_dir}")
    for chunk in chunks:
        print(f"      Section {chunk['section']} Part {chunk['chunk_id']}: {chunk['words']} words")
    
    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Sections found:    {len(sections)}/{len(SECTION_PATTERNS)}")
    print(f"  Primary sections:  {len(primary_sections)} (Sections 5-8)")
    print(f"  Training chunks:   {len(chunks)}")
    print(f"  Output files:")
    print(f"    Combined:  {OUTPUT_FILE}")
    print(f"    Sections:  {SECTIONS_DIR}/")
    print(f"    Chunks:    {chunks_dir}/")
    print(f"\nNext step: Run  python src/train_tokenizer.py")


if __name__ == "__main__":
    main()