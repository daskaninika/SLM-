# Speaker Notes: scrape_cisco_docs.py (Tabular)

Use these tables to explain the Cisco documentation scraper in a walkthrough or presentation.

---

## Overview

| Aspect | Detail |
|--------|--------|
| **Role** | First step in the SLM training pipeline. |
| **Input** | Hardcoded list of 38 Cisco URLs (whitepapers, design guides; some duplicates). |
| **Output** | `data/cisco_doc_01.txt` … `cisco_doc_XX.txt` and `artifacts/scrape_report.json`. |
| **Purpose** | Produce clean, structured text for training a small language model on Cisco datacenter/Nexus content. |
| **One-liner to say** | "Fetches each URL, strips HTML to clean text, writes one .txt per page and a JSON report." |

---

## Configuration (Lines 28–96)

| Setting | Value | What to say |
|---------|--------|-------------|
| **RAW_URLS** | 38 URLs | We de-duplicate later; duplicates in the list are intentional. |
| **OUTPUT_DIR** | `data/` (under project root) | Path derived from script location so it works from any cwd. |
| **ARTIFACTS_DIR** | `artifacts/` | For scrape_report.json. |
| **REQUEST_TIMEOUT** | 30 s | Avoid hanging on slow responses. |
| **DELAY_BETWEEN_REQUESTS** | 2 s | Polite to cisco.com; no hammering. |
| **MAX_RETRIES** | 3 | Retry on transient failures. |
| **HEADERS** | User-Agent, Accept, Accept-Language | Look like a normal browser; reduce blocking. |
| **Credentials** | None | Only public GETs; no API keys or auth. |

---

## Pipeline Steps (High Level)

| Step | Function / Location | Say | Key detail |
|------|---------------------|-----|------------|
| 1 | `deduplicate_urls()` (99–113) | "We normalize URLs and keep first occurrence; order preserved for stable file numbering." | Set for "seen", list for order; print raw vs unique count. |
| 2 | `fetch_html()` (116–163) | "We GET each URL with retries and backoff; record status or error in the report." | See "Fetch HTML behavior" table. |
| 3 | `extract_text_from_html()` (166–291) | "BeautifulSoup parses HTML; we remove junk, find main content, emit structured text." | See "Extract text" tables. |
| 4 | `quality_check()` (294–315) | "We only save pages that pass a simple quality gate." | See "Quality check" table. |
| 5 | `url_to_short_name()` (318–334) | "We derive a safe, short filename from the last path segment of the URL." | Truncate to 60 chars; safe chars only; index in final name. |
| 6 | `main()` (337–410) | "main() runs the pipeline and writes the JSON report." | See "Main pipeline" table. |

---

## Fetch HTML Behavior (Step 2)

| Outcome | Action | Note |
|---------|--------|------|
| **200** | Return HTML string. | Success. |
| **403** | Log, back off 5× attempt seconds, retry; else record 403. | Longer backoff for forbidden. |
| **404** | No retry; record 404. | Page missing. |
| **Other HTTP** | Retry with 3× attempt backoff; else record status. | Transient errors. |
| **Timeout** | Retry with backoff; else record "Timeout". | |
| **ConnectionError** | Retry with backoff; else record error. | |
| **TLS/certs** | Default `requests` behavior. | "We don't disable TLS or cert verification." |

---

## Extract Text — Remove Junk (Step 3a)

| What we remove | Why |
|----------------|-----|
| `<script>`, `<style>`, `<noscript>`, `<iframe>` | No code or boilerplate in training text. |
| HTML comments | Avoid noise. |
| nav, header, footer, sidebars | Navigation and chrome. |
| .navbar, .sidebar, .breadcrumb, .cookie-banner, etc. | Per selector list in script. |
| #header, #footer, [role='navigation'], etc. | Keep only main content. |

---

## Extract Text — Find Main Content (Step 3b)

| Strategy | What we do |
|----------|------------|
| **1** | Try content selectors: `article`, `main`, `.content-body`, `.document-content`, `.cisco-content`, `#fw-content`, `.chapter-body`, etc. Use only if text length > 200 chars. |
| **2** | If none match, use `body` or root. |
| **Page types** | Script targets `/td/docs/` (technical) and `/products/collateral/` (marketing); selectors chosen for both. |

---

## Extract Text — Emit Structured Text (Step 3c)

| HTML element | Output format |
|--------------|----------------|
| `<title>` | `# Title` (after stripping " - Cisco"–style suffixes). |
| **h1–h6** | `#`, `##`, … `######` + heading text. |
| **p** | Plain line + blank line. |
| **li** | `  • item`. |
| **pre** / code in pre | ` ``` ` … code … ` ``` `. |
| **table** | Rows as pipe-separated lines. |
| **figure** with figcaption | `[Figure: caption]`. |
| **Cleanup** | Collapse long newlines; strip trailing space; remove boilerplate regex (e.g. "Was this helpful?", copyright, "Bias-Free Language"). |
| **Security** | "We only parse and extract text; no eval or remote scripts—no script injection from page content." |

---

## Quality Check (Step 4)

| Criterion | Requirement | If failed |
|-----------|-------------|-----------|
| Non-empty | Text must not be empty. | Skip; record in report. |
| Word count | ≥ 50 words. | Skip "Too short"; avoids error/placeholder pages. |
| Content lines | ≥ 5 non-empty lines. | Skip "Too few content lines"; avoids nav-only. |
| **Say** | "Failed pages don't get a file; we increment skipped_quality and keep the training set clean." | |

---

## Main Pipeline — main() Flow

| # | Action | Note |
|---|--------|------|
| 1 | Print banner; create `data/` and `artifacts/`. | |
| 2 | De-duplicate URLs; print count. | |
| 3 | Init report: total_urls, successful, failed, skipped_quality, total_words, files[], errors[]. | |
| 4 | For each URL: fetch → extract → quality check. On fail: update report, continue. | |
| 5 | On success: add header (SOURCE, TYPE, SCRAPED), footer `<DOC_END>`; write file; update report. | |
| 6 | Sleep DELAY_BETWEEN_REQUESTS between requests. | |
| 7 | Print summary (counts, file list, errors). | |
| 8 | Write `artifacts/scrape_report.json`. | JSON for other tools; no HTML or credentials in report. |
| 9 | If any success, print next-step hint (e.g. prepare_data). | |

---

## Security and Best Practices

| Topic | Point to make |
|-------|----------------|
| **Credentials** | No hardcoded secrets; only public GETs. |
| **HTTPS** | Default requests behavior; TLS and cert verification enabled. |
| **Input** | URLs from our list; if you add config/CLI URLs later, allowlist domains (e.g. cisco.com). |
| **Output** | Write only to `data/` and `artifacts/`; no user-driven paths (no path traversal). |
| **Dependencies** | Keep `requests` and `beautifulsoup4` updated. |

---

## Closing / Next Steps

| What to say |
|-------------|
| "After this script we have one .txt per successful doc and a JSON report. Next: other data prep if needed, then `prepare_data.py` to build the dataset, then `train_tokenizer.py` and `train_model.py`. This scraper is the single source of Cisco doc text for the pipeline." |

---

## Key Functions (Quick Reference)

| Function | Purpose |
|----------|---------|
| `deduplicate_urls()` | Normalize and de-duplicate URL list; preserve order. |
| `fetch_html()` | GET with retries, backoff, and status/error handling. |
| `extract_text_from_html()` | Parse HTML, remove chrome, emit structured text. |
| `quality_check()` | Enforce minimum length and line count. |
| `url_to_short_name()` | Derive a safe, short filename from URL. |
| `main()` | Run full pipeline and write report. |

---

*Use these tables alongside the script; adjust wording for your audience.*
