"""
scrape_cisco_docs.py
====================
Scrapes Cisco HTML documentation pages and saves clean text files
for SLM training data.

Usage:
    python src/scrape_cisco_docs.py

Output:
    data/cisco_doc_01.txt  through  data/cisco_doc_XX.txt
    artifacts/scrape_report.json  (summary of what was scraped)

Author: kanidas
Date:   2026-03-06
"""

import os
import re
import json
import time
import hashlib
import requests
from bs4 import BeautifulSoup, Comment

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

# All 38 URLs from user (includes duplicates — we will de-duplicate)
RAW_URLS = [
    # 1-5
    "https://www.cisco.com/c/en/us/td/docs/dcn/whitepapers/cisco-ipfm-design-guide.html",
    "https://www.cisco.com/c/en/us/td/docs/dcn/whitepapers/extending-kubernetes-clusters-with-nx-os-vxlan-and-calico.html",
    "https://www.cisco.com/c/en/us/td/docs/dcn/whitepapers/tenant-routed-multicast-in-nexus9000-vxlan-bgp-evpn-fabrics.html",
    "https://www.cisco.com/c/en/us/td/docs/dcn/whitepapers/cisco-vxlan-bgp-evpn-design-and-implementation-guide.html",
    "https://www.cisco.com/c/en/us/td/docs/dcn/whitepapers/securing-datacenters-with-microsegmentation-and-vxlan-gpo.html",
    # 6-10
    "https://www.cisco.com/c/en/us/td/docs/dcn/whitepapers/q-in-vni-over-vxlan-fabric-deployment-guide.html",
    "https://www.cisco.com/c/en/us/td/docs/dcn/whitepapers/roce-storage-implementation-over-nxos-vxlan-fabrics.html",
    "https://www.cisco.com/c/en/us/products/collateral/switches/nexus-9000-series-switches/telco-data-center-wp.html",
    "https://www.cisco.com/c/en/us/td/docs/dcn/whitepapers/nextgen-oob-datacenter-mgmt-nw-with-evpn-vxlan.html",
    "https://www.cisco.com/c/en/us/products/collateral/switches/nexus-9000-series-switches/white-paper-c11-739942.html",
    # 11-15
    "https://www.cisco.com/c/en/us/products/collateral/switches/nexus-9000-series-switches/layer4-layer7-service-redir-ply-based-redir-wp.html",
    "https://www.cisco.com/c/en/us/products/collateral/switches/nexus-9000-series-switches/guide-c07-742142.html",
    "https://www.cisco.com/c/en/us/products/collateral/switches/nexus-9000-series-switches/whitepaper-c11-742114.html",
    "https://www.cisco.com/c/en/us/products/collateral/switches/nexus-9000-series-switches/whitepaper-c11-742114.html",  # duplicate of 13
    "https://www.cisco.com/c/en/us/products/collateral/switches/nexus-9000-series-switches/white-paper-c11-744191.html",
    # 16-20
    "https://www.cisco.com/c/en/us/products/collateral/switches/nexus-9000-series-switches/white-paper-c11-743731.html",
    "https://www.cisco.com/c/en/us/products/collateral/switches/nexus-9000-series-switches/white-paper-c11-743132.html",
    "https://www.cisco.com/c/en/us/products/collateral/switches/nexus-9000-series-switches/guide-c07-734107.html",
    "https://www.cisco.com/c/en/us/td/docs/dcn/whitepapers/cisco-vxlan-multi-site-and-service-node-integration.html",
    "https://www.cisco.com/c/en/us/products/collateral/data-center-networking/nexus-hyperfabric/nexus-9000-ai-era-ds.html",
    # 21-25
    "https://www.cisco.com/c/en/us/products/collateral/networking/cloud-networking-switches/nexus-9000-switches/nexus-9000-ai-networking-wp.html",
    "https://www.cisco.com/c/en/us/products/collateral/switches/nexus-9000-series-switches/nexus-9000-series-switches-ai-clusters-wp.html",
    "https://www.cisco.com/c/en/us/td/docs/dcn/whitepapers/cisco-addressing-ai-ml-network-challenges.html",
    "https://www.cisco.com/c/en/us/td/docs/dcn/whitepapers/cisco-data-center-networking-blueprint-for-ai-ml-applications.html",
    "https://www.cisco.com/c/en/us/td/docs/dcn/whitepapers/cisco-trustworthy-technologies-nexus-9000.html",
    # 26-30
    "https://www.cisco.com/c/en/us/td/docs/dcn/whitepapers/cvd-for-data-center-networking-blueprint-for-ai.html",
    "https://www.cisco.com/c/en/us/products/collateral/switches/nexus-9000-series-switches/nexus-9800-series-switches-wp.html",
    "https://www.cisco.com/c/en/us/products/collateral/switches/nexus-9000-series-switches/nexus-9800-series-switches-wp.html",  # duplicate of 27
    "https://www.cisco.com/c/en/us/td/docs/dcn/whitepapers/acl-tcam-in-cisco-cloud-scale-asics-for-nexus-9000-series-switches-white-paper.html",
    "https://www.cisco.com/c/en/us/products/collateral/switches/nexus-9000-series-switches/flexible-fwd-nexus-9000-wp.html",
    # 31-35
    "https://www.cisco.com/c/en/us/products/collateral/switches/nexus-9000-series-switches/nexus-9000-span-drop-wp.html",
    "https://www.cisco.com/c/en/us/td/docs/dcn/whitepapers/cisco-nexus-9300-h-series-switches.html",
    "https://www.cisco.com/c/en/us/td/docs/dcn/whitepapers/cisco-nexus-9300-h-series-switches.html",  # duplicate of 32
    "https://www.cisco.com/c/en/us/products/collateral/switches/nexus-9000-series-switches/white-paper-c11-737199.html",
    "https://www.cisco.com/c/en/us/products/collateral/switches/nexus-9000-series-switches/white-paper-c11-741518.html",
    # 36-38
    "https://www.cisco.com/c/en/us/products/collateral/switches/nexus-9000-series-switches/white-paper-c11-738488.html",
    "https://www.cisco.com/c/en/us/td/docs/switches/datacenter/nexus9000/sw/7-x/programmability/guide/b_Cisco_Nexus_9000_Series_NX-OS_Programmability_Guide_7x/b_Cisco_Nexus_9000_Series_NX-OS_Programmability_Guide_7x_chapter_01110.html",
    "https://www.cisco.com/c/en/us/products/collateral/switches/nexus-9000-series-switches/white-paper-c11-732453.html",
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")

# Polite scraping settings
REQUEST_TIMEOUT = 30          # seconds
DELAY_BETWEEN_REQUESTS = 2   # seconds (be polite to cisco.com)
MAX_RETRIES = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ──────────────────────────────────────────────
# STEP 1: De-duplicate URLs
# ──────────────────────────────────────────────

def deduplicate_urls(url_list):
    """Remove duplicate URLs while preserving order."""
    seen = set()
    unique = []
    for url in url_list:
        url_clean = url.strip().rstrip("/")
        if url_clean not in seen:
            seen.add(url_clean)
            unique.append(url_clean)
    print(f"[DEDUP] {len(url_list)} raw URLs → {len(unique)} unique URLs")
    return unique


# ──────────────────────────────────────────────
# STEP 2: Fetch HTML
# ──────────────────────────────────────────────

def fetch_html(url, retries=MAX_RETRIES):
    """
    Fetch raw HTML from a URL with retries and polite delay.
    Returns (html_string, status_code) or (None, error_message).
    """
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True
            )
            if response.status_code == 200:
                return response.text, 200
            elif response.status_code == 403:
                print(f"  [WARN] 403 Forbidden (attempt {attempt}/{retries}): {url}")
                if attempt < retries:
                    time.sleep(5 * attempt)  # back off longer for 403
                    continue
                return None, "403 Forbidden"
            elif response.status_code == 404:
                print(f"  [WARN] 404 Not Found: {url}")
                return None, "404 Not Found"
            else:
                print(f"  [WARN] HTTP {response.status_code} (attempt {attempt}/{retries})")
                if attempt < retries:
                    time.sleep(3 * attempt)
                    continue
                return None, f"HTTP {response.status_code}"
        except requests.exceptions.Timeout:
            print(f"  [WARN] Timeout (attempt {attempt}/{retries})")
            if attempt < retries:
                time.sleep(3 * attempt)
                continue
            return None, "Timeout"
        except requests.exceptions.ConnectionError as e:
            print(f"  [WARN] Connection error (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(3 * attempt)
                continue
            return None, f"ConnectionError: {e}"
        except Exception as e:
            return None, f"Exception: {e}"
    return None, "Max retries exceeded"


# ──────────────────────────────────────────────
# STEP 3: Extract clean text from HTML
# ──────────────────────────────────────────────

def extract_text_from_html(html_string, url):
    """
    Parse HTML and extract main content text.
    Handles two Cisco page layouts:
      - /td/docs/  pages (technical documentation)
      - /products/collateral/  pages (marketing whitepapers)
    """
    soup = BeautifulSoup(html_string, "html.parser")

    # ---- Remove unwanted elements ----
    # Scripts, styles, comments
    for element in soup.find_all(["script", "style", "noscript", "iframe"]):
        element.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove navigation, headers, footers, sidebars
    selectors_to_remove = [
        "nav", "header", "footer",
        ".navbar", ".nav-bar", ".navigation",
        ".sidebar", ".side-bar", ".side-nav",
        ".footer", ".header", ".breadcrumb", ".breadcrumbs",
        ".cookie-banner", ".cookie-notice",
        ".social-share", ".share-buttons",
        "#header", "#footer", "#nav", "#sidebar",
        ".topnav", ".top-nav",
        ".feedback-section", ".rating-section",
        ".related-content", ".recommended",
        "[role='navigation']", "[role='banner']",
        "[role='contentinfo']",
    ]
    for selector in selectors_to_remove:
        for element in soup.select(selector):
            element.decompose()

    # ---- Find main content ----
    main_content = None

    # Strategy 1: Look for standard main content containers
    content_selectors = [
        "article",
        "main",
        "[role='main']",
        ".content-body",
        ".document-content",
        ".article-content",
        "#content",
        ".content",
        "#main-content",
        ".main-content",
        ".page-content",
        ".body-content",
        # Cisco-specific selectors
        ".cisco-content",
        ".cdc-content",
        "#fw-content",
        ".chapter-body",
        ".conceptbody",
        ".sectionbody",
    ]

    for selector in content_selectors:
        found = soup.select_one(selector)
        if found and len(found.get_text(strip=True)) > 200:
            main_content = found
            break

    # Strategy 2: If no main content found, use the body
    if main_content is None:
        main_content = soup.body if soup.body else soup

    # ---- Extract text with structure ----
    lines = []
    title = None

    # Get page title
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
        # Clean up common Cisco title suffixes
        for suffix in [" - Cisco", " — Cisco", "| Cisco"]:
            if title.endswith(suffix):
                title = title[: -len(suffix)].strip()

    if title:
        lines.append(f"# {title}")
        lines.append("")

    # Process content elements in order
    for element in main_content.descendants:
        if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            text = element.get_text(strip=True)
            if text and len(text) > 1:
                level = int(element.name[1])
                prefix = "#" * level
                lines.append("")
                lines.append(f"{prefix} {text}")
                lines.append("")

        elif element.name == "p":
            text = element.get_text(strip=True)
            if text and len(text) > 1:
                lines.append(text)
                lines.append("")

        elif element.name == "li":
            text = element.get_text(strip=True)
            if text and len(text) > 1:
                lines.append(f"  • {text}")

        elif element.name == "pre" or (
            element.name == "code" and element.parent.name == "pre"
        ):
            if element.name == "pre":
                code_text = element.get_text()
                if code_text.strip():
                    lines.append("")
                    lines.append("```")
                    lines.append(code_text.rstrip())
                    lines.append("```")
                    lines.append("")

        elif element.name in ["table"]:
            # Extract table as simple text
            rows = element.find_all("tr")
            if rows:
                lines.append("")
                for row in rows:
                    cells = row.find_all(["th", "td"])
                    cell_texts = [c.get_text(strip=True) for c in cells]
                    if any(cell_texts):
                        lines.append(" | ".join(cell_texts))
                lines.append("")

        elif element.name == "figure":
            # Get figure caption if any
            caption = element.find("figcaption")
            if caption:
                lines.append(f"[Figure: {caption.get_text(strip=True)}]")
                lines.append("")

    # ---- Join and clean ----
    text = "\n".join(lines)

    # Clean up excessive whitespace
    text = re.sub(r"\n{4,}", "\n\n\n", text)  # max 3 consecutive newlines
    text = re.sub(r"[ \t]+\n", "\n", text)     # trailing spaces
    text = re.sub(r"\n[ \t]+\n", "\n\n", text) # lines with only whitespace

    # Remove common boilerplate phrases
    boilerplate_patterns = [
        r"Was this document helpful\?.*",
        r"Log in.*?to rate.*",
        r"© \d{4} Cisco.*?reserved\.?",
        r"All rights reserved\.?",
        r"Contact Cisco.*",
        r"Updated:.*?\d{4}",
        r"Document ID:.*",
        r"Bias-Free Language.*?learn more\.",
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    return text.strip()


# ──────────────────────────────────────────────
# STEP 4: Quality check
# ──────────────────────────────────────────────

def quality_check(text, url):
    """
    Basic quality checks on extracted text.
    Returns (is_ok, message).
    """
    if not text:
        return False, "Empty text"

    word_count = len(text.split())

    if word_count < 50:
        return False, f"Too short ({word_count} words)"

    # Check if it's mostly navigation/boilerplate
    lines = text.strip().split("\n")
    non_empty_lines = [l for l in lines if l.strip()]
    if len(non_empty_lines) < 5:
        return False, f"Too few content lines ({len(non_empty_lines)})"

    return True, f"OK ({word_count} words, {len(non_empty_lines)} lines)"


# ──────────────────────────────────────────────
# STEP 5: Generate filename from URL
# ──────────────────────────────────────────────

def url_to_short_name(url):
    """Create a human-readable short name from URL for the filename."""
    # Extract the last meaningful part of the URL path
    path = url.split("//")[1] if "//" in url else url
    # Get the filename part (without .html)
    parts = path.rstrip("/").split("/")
    name = parts[-1] if parts else "unknown"
    name = name.replace(".html", "").replace(".htm", "")

    # Shorten if too long
    if len(name) > 60:
        name = name[:60]

    # Clean characters
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return name


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  CISCO DOCUMENTATION SCRAPER FOR SLM TRAINING DATA")
    print("=" * 70)
    print()

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Step 1: De-duplicate
    urls = deduplicate_urls(RAW_URLS)
    print()

    # Tracking
    report = {
        "total_urls": len(urls),
        "successful": 0,
        "failed": 0,
        "skipped_quality": 0,
        "total_words": 0,
        "files": [],
        "errors": [],
    }

    # Step 2-4: Fetch, extract, save each URL
    for idx, url in enumerate(urls, start=1):
        file_num = f"{idx:02d}"
        short_name = url_to_short_name(url)
        filename = f"cisco_doc_{file_num}_{short_name}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)

        print(f"[{file_num}/{len(urls)}] Scraping: {url[:80]}...")

        # Fetch
        html, status = fetch_html(url)

        if html is None:
            print(f"  ✗ FAILED: {status}")
            report["failed"] += 1
            report["errors"].append({"url": url, "error": str(status)})
            continue

        # Extract
        text = extract_text_from_html(html, url)

        # Quality check
        is_ok, message = quality_check(text, url)

        if not is_ok:
            print(f"  ✗ QUALITY FAIL: {message}")
            report["skipped_quality"] += 1
            report["errors"].append({"url": url, "error": f"Quality: {message}"})
            continue

        # Add source header
        header = (
            f"<DOC_START>\n"
            f"<SOURCE>{url}</SOURCE>\n"
            f"<TYPE>cisco_documentation</TYPE>\n"
            f"<SCRAPED>2026-03-06</SCRAPED>\n"
            f"---\n"
        )
        footer = "\n<DOC_END>\n"
        full_text = header + text + footer

        # Save
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_text)

        word_count = len(text.split())
        report["successful"] += 1
        report["total_words"] += word_count
        report["files"].append({
            "filename": filename,
            "url": url,
            "words": word_count,
            "status": message,
        })

        print(f"  ✓ Saved: {filename} ({word_count:,} words)")

        # Polite delay
        if idx < len(urls):
            time.sleep(DELAY_BETWEEN_REQUESTS)

    # ── Summary ──
    print()
    print("=" * 70)
    print("  SCRAPING COMPLETE — SUMMARY")
    print("=" * 70)
    print(f"  Total unique URLs:     {report['total_urls']}")
    print(f"  Successfully scraped:  {report['successful']}")
    print(f"  Failed (HTTP errors):  {report['failed']}")
    print(f"  Skipped (quality):     {report['skipped_quality']}")
    print(f"  Total words extracted: {report['total_words']:,}")
    print()

    if report["files"]:
        print("  Files created:")
        for f in report["files"]:
            print(f"    • {f['filename']}  ({f['words']:,} words)")
        print()

    if report["errors"]:
        print("  Errors:")
        for e in report["errors"]:
            print(f"    • {e['url'][:60]}... → {e['error']}")
        print()

    # Save report
    report_path = os.path.join(ARTIFACTS_DIR, "scrape_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {report_path}")
    print()

    # Guidance for next step
    if report["successful"] > 0:
        print("  ╔══════════════════════════════════════════════════╗")
        print("  ║  NEXT STEP:                                     ║")
        print("  ║  Run: python src/process_core42_sdd.py           ║")
        print("  ║  (to process the Core42 SDD document)           ║")
        print("  ╚══════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()