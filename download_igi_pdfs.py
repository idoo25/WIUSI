"""
download_igi_pdfs.py — Run this on your LOCAL machine (not Colab)

Cloudflare blocks Google Cloud IPs (Colab), but your home IP works fine.

Usage:
    1. Install Python 3.8+ if you don't have it
    2. pip install requests pdfplumber pandas
    3. Place luvansh_updated.csv in the same folder as this script
    4. python download_igi_pdfs.py
    5. Upload the output (diamonds_full.csv) back to Colab

Takes ~5-10 minutes for 2,064 diamonds with 15 threads.
"""

import re
import io
import os
import sys
import time
import requests
import pdfplumber
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ── Config ─────────────────────────────────────────────────────
THREADS = 15
INPUT_CSV = "luvansh_updated.csv"
OUTPUT_CSV = "diamonds_full.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

print_lock = Lock()
progress = {"done": 0, "ok": 0, "fail": 0}


# ── PDF parsing ────────────────────────────────────────────────
def parse_pdf_text(full_text):
    """Extract proportions from IGI PDF text."""
    props = {}

    def find_f(pattern):
        m = re.search(pattern, full_text, re.I)
        return float(m.group(1)) if m else None

    def find_s(pattern):
        m = re.search(pattern, full_text, re.I)
        return m.group(1).strip() if m else None

    # Measurements & L/W
    meas_m = re.search(
        r'(\d+\.\d+\s*[-\u2013]\s*\d+\.\d+\s*[Xx\u00d7]\s*\d+\.\d+)', full_text)
    if meas_m:
        props["pdf_measurements"] = meas_m.group(1).strip()
        dims = re.findall(r'(\d+\.\d+)', meas_m.group(1))
        if len(dims) >= 2:
            l, w = float(dims[0]), float(dims[1])
            if min(l, w) > 0:
                props["pdf_lw_ratio"] = round(max(l, w) / min(l, w), 3)

    # Named fields
    props["pdf_table_pct"]      = find_f(r'Table\s*:?\s*(\d+(?:\.\d+)?)\s*%')
    props["pdf_depth_pct"]      = find_f(r'Depth\s*:?\s*(\d+(?:\.\d+)?)\s*%')
    props["pdf_crown_angle"]    = find_f(r'Crown\s*Angle\s*:?\s*(\d+\.\d+)')
    props["pdf_pavilion_angle"] = find_f(r'Pavilion\s*Angle\s*:?\s*(\d+\.\d+)')
    props["pdf_crown_height"]   = find_f(r'Crown\s*Height\s*:?\s*(\d+\.\d+)')
    props["pdf_pavilion_depth"] = find_f(r'Pavilion\s*Depth\s*:?\s*(\d+\.\d+)')

    # Proportions diagram: "13.5% 58% 33.1° 40.9° 43% Pointed 61%"
    prop_m = re.search(
        r'(\d+\.\d+)%\s+(\d+)%\s+(\d+\.\d+)[\u00b0]\s+(\d+\.\d+)[\u00b0]\s+'
        r'(\d+(?:\.\d+)?)%\s+\w+\s+(\d+(?:\.\d+)?)%',
        full_text)
    if prop_m:
        props.setdefault("pdf_crown_height",   float(prop_m.group(1)))
        props.setdefault("pdf_table_pct",      float(prop_m.group(2)))
        props.setdefault("pdf_crown_angle",    float(prop_m.group(3)))
        props.setdefault("pdf_pavilion_angle", float(prop_m.group(4)))
        props.setdefault("pdf_pavilion_depth", float(prop_m.group(5)))
        props.setdefault("pdf_depth_pct",      float(prop_m.group(6)))

    # Two angles side by side
    if not props.get("pdf_crown_angle") or not props.get("pdf_pavilion_angle"):
        ang_m = re.search(r'(\d{2}\.\d+)[\u00b0]\s+(\d{2}\.\d+)[\u00b0]', full_text)
        if ang_m:
            props.setdefault("pdf_crown_angle",    float(ang_m.group(1)))
            props.setdefault("pdf_pavilion_angle", float(ang_m.group(2)))

    # Grading fields
    props["pdf_polish"]       = find_s(r'Polish\s*:?\s*(EXCELLENT|VERY\s*GOOD|GOOD|FAIR|POOR)')
    props["pdf_symmetry"]     = find_s(r'Symmetry\s*:?\s*(EXCELLENT|VERY\s*GOOD|GOOD|FAIR|POOR)')
    props["pdf_fluorescence"] = find_s(r'Fluorescence\s*:?\s*(NONE|FAINT|MEDIUM|STRONG|VERY\s*STRONG)')
    props["pdf_girdle"]       = find_s(r'Girdle\s*:?\s*([A-Za-z][A-Za-z\s]*(?:\(Faceted\))?)')
    props["pdf_culet"]        = find_s(r'Culet\s*:?\s*(None|Pointed|Very\s*Small|Small|Medium|Large)')
    props["pdf_cut_grade"]    = find_s(r'Cut\s*(?:Grade)?\s*:?\s*(IDEAL|EXCELLENT|VERY\s*GOOD|GOOD)')
    props["pdf_carat"]        = find_f(r'Carat\s*Weight\s*:?\s*(\d+\.\d+)')
    props["pdf_color"]        = find_s(r'Color\s*Grade\s*:?\s*([A-Z])\b')
    props["pdf_clarity"]      = find_s(r'Clarity\s*Grade\s*:?\s*(FL|IF|VVS[12]|VS[12]|SI[12]|I[123])')

    return props


def process_row(idx, url, sku, total):
    """Download one PDF and extract proportions."""
    props = {}
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        props = parse_pdf_text(full_text)
    except Exception as e:
        props["pdf_error"] = str(e)[:120]

    with print_lock:
        progress["done"] += 1
        d = progress["done"]
        if props.get("pdf_error"):
            progress["fail"] += 1
        else:
            progress["ok"] += 1

        if d <= 3 or d % 100 == 0 or d == total:
            ca = props.get('pdf_crown_angle', '-')
            pa = props.get('pdf_pavilion_angle', '-')
            err = props.get('pdf_error', '')
            ok, fail = progress["ok"], progress["fail"]
            if err:
                print(f"  [{d}/{total}] {sku:<35} ERROR: {err[:50]}  (ok={ok} fail={fail})")
            else:
                print(f"  [{d}/{total}] {sku:<35} CrAngle={ca}  PavAngle={pa}  (ok={ok} fail={fail})")

    return idx, props


def main():
    # ── Load CSV ───────────────────────────────────────────
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: '{INPUT_CSV}' not found in {os.getcwd()}")
        print(f"Place the CSV file next to this script and try again.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} diamonds from {INPUT_CSV}")

    # ── Quick test ─────────────────────────────────────────
    test_url = df["web_certificate_url"].dropna().iloc[0]
    print(f"\nTesting: {test_url}")
    try:
        resp = requests.get(test_url, headers=HEADERS, timeout=15)
        print(f"Status: {resp.status_code}  Size: {len(resp.content)} bytes")
        if resp.status_code != 200:
            print(f"\nERROR: Got {resp.status_code}. Your IP might also be blocked.")
            print("Try opening this URL in your browser to verify it works:")
            print(f"  {test_url}")
            sys.exit(1)
        # Parse test PDF
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            text = pdf.pages[0].extract_text() or ""
            print(f"PDF OK! First 150 chars: {text[:150]}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # ── Download all PDFs ──────────────────────────────────
    has_url = df["web_certificate_url"].notna()
    indices = df.index[has_url].tolist()
    total = len(indices)
    print(f"\nDownloading {total} IGI PDFs with {THREADS} threads ...\n")

    pdf_columns = [
        "pdf_measurements", "pdf_lw_ratio",
        "pdf_table_pct", "pdf_depth_pct",
        "pdf_crown_angle", "pdf_pavilion_angle",
        "pdf_crown_height", "pdf_pavilion_depth",
        "pdf_polish", "pdf_symmetry", "pdf_fluorescence",
        "pdf_girdle", "pdf_culet",
        "pdf_cut_grade", "pdf_carat", "pdf_color", "pdf_clarity",
        "pdf_error",
    ]
    for col in pdf_columns:
        df[col] = None

    start = time.time()

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = {}
        for idx in indices:
            url = df.at[idx, "web_certificate_url"]
            sku = df.at[idx, "web_sku"] if "web_sku" in df.columns else ""
            futures[executor.submit(process_row, idx, url, sku, total)] = idx

        for future in as_completed(futures):
            idx, props = future.result()
            for k, v in props.items():
                if k in pdf_columns:
                    df.at[idx, k] = v

    elapsed = time.time() - start

    # ── Save ───────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)

    got_angle = df["pdf_crown_angle"].notna().sum()
    got_error = df["pdf_error"].notna().sum()
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.0f}s")
    print(f"Crown angle extracted: {got_angle}/{total}")
    print(f"Errors: {got_error}/{total}")
    print(f"Saved: {OUTPUT_CSV} ({len(df)} rows, {len(df.columns)} columns)")
    print(f"{'='*60}")
    print(f"\nUpload {OUTPUT_CSV} back to Colab to continue.")


if __name__ == "__main__":
    main()
