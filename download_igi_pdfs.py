"""
download_igi_pdfs.py — Run on your LOCAL machine (not Colab)

Uses Selenium to navigate to each IGI PDF URL (passes Cloudflare),
Chrome auto-downloads the PDF, then PyMuPDF extracts proportions.

Usage:
    1. pip install selenium pymupdf pandas
    2. Download chromedriver for your Chrome version:
       https://googlechromelabs.github.io/chrome-for-testing/
    3. Edit the paths below (CHROMEDRIVER, INPUT_CSV, etc.)
    4. python download_igi_pdfs.py
    5. Upload diamonds_full.csv back to Colab

Features:
    - Multiple Chrome drivers for parallel downloads (default 5)
    - Resume support: re-run to pick up where you left off
    - Auto-saves progress every 50 diamonds
    - Combines heuristic + regex PDF parsing for best accuracy
"""

import os
import re
import sys
import csv
import time
import shutil
import fitz  # PyMuPDF
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# ═══════════════════════════════════════════════════════════════════
# EDIT THESE PATHS FOR YOUR MACHINE
# ═══════════════════════════════════════════════════════════════════
INPUT_CSV = r"luvansh_updated.csv"
OUTPUT_CSV = r"diamonds_full.csv"
CHROMEDRIVER = r"chromedriver.exe"         # path to chromedriver
DL_BASE_DIR = r"pdf_dl"                    # base folder for PDF downloads
DRIVERS = 5                                # number of Chrome windows (5 is safe, 10 max)
# ═══════════════════════════════════════════════════════════════════

# Columns extracted from PDF
PDF_COLS = [
    "pdf_crown_angle",
    "pdf_pavilion_angle",
    "pdf_crown_height",
    "pdf_pavilion_depth",
    "pdf_table_pct",
    "pdf_depth_pct",
    "pdf_lw_ratio",
    "pdf_measurements",
    "pdf_girdle",
    "pdf_culet",
    "pdf_polish",
    "pdf_symmetry",
    "pdf_fluorescence",
    "pdf_cut_grade",
    "pdf_carat",
    "pdf_color",
    "pdf_clarity",
    "pdf_error",
]

print_lock = Lock()
save_lock = Lock()
progress = {"done": 0, "ok": 0, "fail": 0, "skip": 0, "total": 0}


# ── Chrome driver setup ───────────────────────────────────────────
def create_driver(dl_dir):
    """Create a Chrome driver that auto-downloads PDFs to dl_dir."""
    os.makedirs(dl_dir, exist_ok=True)

    opts = Options()
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_experimental_option("prefs", {
        "download.default_directory": os.path.abspath(dl_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True,
    })

    service = Service(CHROMEDRIVER)
    driver = webdriver.Chrome(service=service, options=opts)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


def clear_dl_dir(dl_dir):
    """Remove all files from the download directory."""
    if os.path.exists(dl_dir):
        for f in os.listdir(dl_dir):
            try:
                os.remove(os.path.join(dl_dir, f))
            except Exception:
                pass


def wait_for_pdf(dl_dir, timeout=30):
    """Wait for a PDF to finish downloading in dl_dir."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            files = os.listdir(dl_dir)
        except Exception:
            time.sleep(0.5)
            continue
        pdfs = [f for f in files if f.lower().endswith(".pdf")]
        downloading = [f for f in files if f.endswith(".crdownload") or f.endswith(".tmp")]
        if pdfs and not downloading:
            time.sleep(0.3)  # let file handle release
            return os.path.join(dl_dir, pdfs[0])
        time.sleep(0.5)
    return None


# ── PDF parsing ───────────────────────────────────────────────────
def parse_pdf(pdf_path):
    """
    Extract proportions from an IGI PDF using PyMuPDF.

    Two-pass approach:
      1. Heuristic: extract all % and ° values, classify by value range
         (works for the proportions diagram where values aren't labeled)
      2. Regex: look for labeled fields like "Crown Angle: 34.5°"
         (works for text-based sections of the PDF)
    """
    data = {c: None for c in PDF_COLS}

    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        lines = []

        for page in doc:
            # Block-based extraction (preserves layout better)
            for block in page.get_text("blocks"):
                block_text = block[4] if len(block) > 4 else ""
                full_text += block_text + "\n"
                for line in block_text.split("\n"):
                    t = line.strip()
                    if t:
                        lines.append(t)
        doc.close()

    except Exception as e:
        data["pdf_error"] = f"open: {str(e)[:100]}"
        return data

    # ── Pass 1: Heuristic classification (proportions diagram) ────
    pcts = []
    angles = []
    girdle_parts = []

    for t in lines:
        # Percentages: "57%", "62.1%"
        m = re.match(r'^(\d+(?:\.\d+)?)%$', t)
        if m:
            pcts.append(float(m.group(1)))

        # Angles: "34.5°", "40.8°"
        m = re.match(r'^(\d+(?:\.\d+)?)[°]$', t)
        if m:
            angles.append(float(m.group(1)))

        # Girdle: "Medium", "Thick", "(Faceted)"
        if any(g in t for g in ["Medium", "Thick", "Thin", "Faceted",
                                 "Slightly", "Very"]) and len(t) < 40:
            girdle_parts.append(t)

        # Culet: "Pointed", "None"
        if t in ("Pointed", "None", "Very Small", "Small", "Medium", "Large"):
            if data["pdf_culet"] is None:
                data["pdf_culet"] = t

    # Classify percentages by value range
    for val in pcts:
        if 10.0 <= val <= 20.0 and data["pdf_crown_height"] is None:
            data["pdf_crown_height"] = val
        elif 52.0 <= val <= 65.0 and data["pdf_table_pct"] is None:
            data["pdf_table_pct"] = val
        elif 40.0 <= val <= 46.0 and data["pdf_pavilion_depth"] is None:
            data["pdf_pavilion_depth"] = val
        elif 58.0 <= val <= 66.0 and data["pdf_depth_pct"] is None:
            data["pdf_depth_pct"] = val

    # Classify angles by value range
    for val in angles:
        if 30.0 <= val <= 38.0 and data["pdf_crown_angle"] is None:
            data["pdf_crown_angle"] = val
        elif 38.0 <= val <= 43.0 and data["pdf_pavilion_angle"] is None:
            data["pdf_pavilion_angle"] = val

    # Build girdle string
    if girdle_parts:
        data["pdf_girdle"] = " ".join(girdle_parts).replace("(", "").replace(")", "").strip()

    # ── Pass 2: Regex for labeled fields ──────────────────────────
    def find_f(pattern):
        m = re.search(pattern, full_text, re.I)
        return float(m.group(1)) if m else None

    def find_s(pattern):
        m = re.search(pattern, full_text, re.I)
        return m.group(1).strip() if m else None

    # Fill in anything the heuristic missed
    if data["pdf_crown_angle"] is None:
        data["pdf_crown_angle"] = find_f(r'Crown\s*Angle\s*:?\s*(\d+\.\d+)')
    if data["pdf_pavilion_angle"] is None:
        data["pdf_pavilion_angle"] = find_f(r'Pavilion\s*Angle\s*:?\s*(\d+\.\d+)')
    if data["pdf_crown_height"] is None:
        data["pdf_crown_height"] = find_f(r'Crown\s*Height\s*:?\s*(\d+\.\d+)')
    if data["pdf_pavilion_depth"] is None:
        data["pdf_pavilion_depth"] = find_f(r'Pavilion\s*Depth\s*:?\s*(\d+\.\d+)')
    if data["pdf_table_pct"] is None:
        data["pdf_table_pct"] = find_f(r'Table\s*:?\s*(\d+(?:\.\d+)?)\s*%')
    if data["pdf_depth_pct"] is None:
        data["pdf_depth_pct"] = find_f(r'Depth\s*:?\s*(\d+(?:\.\d+)?)\s*%')

    # Measurements & L/W ratio
    meas_m = re.search(
        r'(\d+\.\d+\s*[-\u2013]\s*\d+\.\d+\s*[Xx\u00d7]\s*\d+\.\d+)', full_text)
    if meas_m:
        data["pdf_measurements"] = meas_m.group(1).strip()
        dims = re.findall(r'(\d+\.\d+)', meas_m.group(1))
        if len(dims) >= 2:
            l, w = float(dims[0]), float(dims[1])
            if min(l, w) > 0:
                data["pdf_lw_ratio"] = round(max(l, w) / min(l, w), 3)

    # Grading fields
    data["pdf_polish"] = data["pdf_polish"] or find_s(
        r'Polish\s*:?\s*(EXCELLENT|VERY\s*GOOD|GOOD|FAIR|POOR)')
    data["pdf_symmetry"] = data["pdf_symmetry"] or find_s(
        r'Symmetry\s*:?\s*(EXCELLENT|VERY\s*GOOD|GOOD|FAIR|POOR)')
    data["pdf_fluorescence"] = data["pdf_fluorescence"] or find_s(
        r'Fluorescence\s*:?\s*(NONE|FAINT|MEDIUM|STRONG|VERY\s*STRONG)')
    data["pdf_cut_grade"] = find_s(
        r'Cut\s*(?:Grade)?\s*:?\s*(IDEAL|EXCELLENT|VERY\s*GOOD|GOOD)')
    data["pdf_carat"] = find_f(r'Carat\s*Weight\s*:?\s*(\d+\.\d+)')
    data["pdf_color"] = find_s(r'Color\s*Grade\s*:?\s*([A-Z])\b')
    data["pdf_clarity"] = find_s(
        r'Clarity\s*Grade\s*:?\s*(FL|IF|VVS[12]|VS[12]|SI[12]|I[123])')

    return data


# ── Worker function ───────────────────────────────────────────────
def worker(worker_id, rows_for_worker, df, dl_dir):
    """
    Each worker gets its own Chrome driver and download directory.
    Processes its assigned rows sequentially.
    """
    driver = None
    results = []

    try:
        driver = create_driver(dl_dir)
        total = progress["total"]

        for idx, row_idx in enumerate(rows_for_worker):
            url = df.at[row_idx, "web_certificate_url"]
            sku = df.at[row_idx, "web_sku"] if "web_sku" in df.columns else ""

            # Download PDF
            clear_dl_dir(dl_dir)
            try:
                driver.get(url)
            except Exception as e:
                data = {c: None for c in PDF_COLS}
                data["pdf_error"] = f"navigate: {str(e)[:80]}"
                results.append((row_idx, data))
                with print_lock:
                    progress["done"] += 1
                    progress["fail"] += 1
                continue

            # First request: give extra time for Cloudflare
            timeout = 90 if idx == 0 else 30
            pdf_path = wait_for_pdf(dl_dir, timeout=timeout)

            if pdf_path:
                data = parse_pdf(pdf_path)
                results.append((row_idx, data))
                with print_lock:
                    progress["done"] += 1
                    if data.get("pdf_error"):
                        progress["fail"] += 1
                    else:
                        progress["ok"] += 1
                    d = progress["done"]
                    ok, fail = progress["ok"], progress["fail"]
                    if d <= 5 or d % 50 == 0 or d == total:
                        ca = data.get("pdf_crown_angle", "-")
                        pa = data.get("pdf_pavilion_angle", "-")
                        err = data.get("pdf_error", "")
                        if err:
                            print(f"  [{d}/{total}] W{worker_id} {sku:<30} ERROR: {err[:40]}  (ok={ok} fail={fail})")
                        else:
                            print(f"  [{d}/{total}] W{worker_id} {sku:<30} Cr={ca} Pav={pa}  (ok={ok} fail={fail})")
            else:
                data = {c: None for c in PDF_COLS}
                data["pdf_error"] = "download timeout"
                results.append((row_idx, data))
                with print_lock:
                    progress["done"] += 1
                    progress["fail"] += 1
                    d = progress["done"]
                    if d <= 5 or d % 50 == 0:
                        print(f"  [{d}/{total}] W{worker_id} {sku:<30} TIMEOUT")

            # Auto-save every 50 results
            if len(results) % 50 == 0 and results:
                with save_lock:
                    for ri, rd in results:
                        for k, v in rd.items():
                            if k in PDF_COLS:
                                df.at[ri, k] = v
                    df.to_csv(OUTPUT_CSV, index=False)
                    results.clear()

    except Exception as e:
        print(f"  [Worker {worker_id}] Fatal error: {e}")
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    return results


# ── Main ──────────────────────────────────────────────────────────
def main():
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: '{INPUT_CSV}' not found in {os.getcwd()}")
        sys.exit(1)

    if not os.path.exists(CHROMEDRIVER):
        print(f"ERROR: Chromedriver not found at '{CHROMEDRIVER}'")
        print("Download from: https://googlechromelabs.github.io/chrome-for-testing/")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} diamonds from {INPUT_CSV}")

    # ── Resume support ────────────────────────────────────────
    # Add PDF columns if they don't exist
    for col in PDF_COLS:
        if col not in df.columns:
            df[col] = None

    # If output CSV exists, load previous results
    if os.path.exists(OUTPUT_CSV):
        try:
            df_prev = pd.read_csv(OUTPUT_CSV)
            # Merge previous PDF results
            for col in PDF_COLS:
                if col in df_prev.columns:
                    df[col] = df_prev[col]
            already_done = df["pdf_crown_angle"].notna().sum()
            print(f"Resumed: {already_done} diamonds already processed")
        except Exception:
            pass

    # Find rows that still need processing
    has_url = df["web_certificate_url"].notna()
    needs_work = df["pdf_crown_angle"].isna() & df["pdf_error"].isna()
    indices = df.index[has_url & needs_work].tolist()

    if not indices:
        print("All diamonds already processed!")
        df.to_csv(OUTPUT_CSV, index=False)
        return

    print(f"Diamonds to process: {len(indices)}")
    print(f"Using {DRIVERS} Chrome drivers\n")

    progress["total"] = len(indices)
    progress["done"] = 0
    progress["ok"] = 0
    progress["fail"] = 0

    # Split work across workers
    chunks = [[] for _ in range(DRIVERS)]
    for i, idx in enumerate(indices):
        chunks[i % DRIVERS].append(idx)

    # Create download directories
    dl_dirs = []
    for i in range(DRIVERS):
        d = os.path.join(DL_BASE_DIR, f"w{i}")
        os.makedirs(d, exist_ok=True)
        dl_dirs.append(d)

    print(f"Starting {DRIVERS} Chrome windows...\n")
    start = time.time()

    # Run workers in parallel
    with ThreadPoolExecutor(max_workers=DRIVERS) as executor:
        futures = {}
        for i in range(DRIVERS):
            if chunks[i]:  # only start if there's work
                futures[executor.submit(worker, i, chunks[i], df, dl_dirs[i])] = i

        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                remaining = future.result()
                # Save any remaining results
                with save_lock:
                    for ri, rd in remaining:
                        for k, v in rd.items():
                            if k in PDF_COLS:
                                df.at[ri, k] = v
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")

    elapsed = time.time() - start

    # ── Final save ────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)

    got_angle = df["pdf_crown_angle"].notna().sum()
    got_error = df["pdf_error"].notna().sum()
    total_rows = len(df)

    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.0f}s ({elapsed/max(len(indices),1):.1f}s per diamond)")
    print(f"  Crown angle extracted: {got_angle}/{total_rows}")
    print(f"  Errors: {got_error}/{total_rows}")
    print(f"  Saved: {OUTPUT_CSV}")
    print(f"{'='*60}")

    # Show sample results
    has_data = df["pdf_crown_angle"].notna()
    if has_data.sum() > 0:
        print("\nSample results:")
        show_cols = ["web_sku", "carat", "price",
                     "pdf_crown_angle", "pdf_pavilion_angle",
                     "pdf_crown_height", "pdf_pavilion_depth",
                     "pdf_table_pct", "pdf_depth_pct", "pdf_girdle", "pdf_culet"]
        show_cols = [c for c in show_cols if c in df.columns]
        print(df[has_data][show_cols].head(10).to_string())

    # Cleanup download dirs
    try:
        shutil.rmtree(DL_BASE_DIR, ignore_errors=True)
    except Exception:
        pass

    print(f"\nUpload {OUTPUT_CSV} back to Colab to continue.")


if __name__ == "__main__":
    main()
