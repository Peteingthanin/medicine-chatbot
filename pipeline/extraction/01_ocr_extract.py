import os
import sys
import time

# Add Poppler to PATH for Windows
poppler_path = r"C:\poppler\poppler-25.12.0\Library\bin"
if poppler_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = os.environ.get("PATH", "") + ";" + poppler_path

import json
import glob
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from typhoon_ocr import ocr_document

load_dotenv()

INPUT_DIR = r".\fda_pdfs"
OUTPUT_DIR = r".\extracted_data\raw"
LOG_FILE = r".\extracted_data\ocr_extraction_log.txt"


def setup_directories():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def get_pdf_files():
    pdf_files = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    return sorted(pdf_files)


def get_processed_files():
    """Return set of PDF filenames (without extension) that already have .md output."""
    existing_md = glob.glob(os.path.join(OUTPUT_DIR, "*.md"))
    return {Path(m).stem for m in existing_md}


def extract_text_ocr(pdf_path):
    try:
        markdown = ocr_document(pdf_path)
        return markdown, None
    except Exception as e:
        return None, str(e)


MAX_RETRIES = 5
RETRY_DELAY = 10

def extract_text_with_retry(pdf_path, max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    for attempt in range(max_retries):
        try:
            markdown = ocr_document(pdf_path)
            return markdown, None
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "429" in error_str or "too many" in error_str:
                if attempt < max_retries - 1:
                    print(f"  Rate limited, retrying in {delay}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(delay)
                else:
                    return None, f"Rate limit exceeded after {max_retries} retries"
            else:
                return None, str(e)
    return None, "Max retries exceeded"


def save_json(filename, data):
    json_path = os.path.join(OUTPUT_DIR, f"{filename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_md(filename, text):
    md_path = os.path.join(OUTPUT_DIR, f"{filename}.md")
    content = f"# {filename}\n\n{text}"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)


def log_result(filename, status, error=None):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        if status == "success":
            f.write(f"OK: {filename}\n")
        else:
            f.write(f"FAIL: {filename} - {error}\n")


def process_batch(batch_size=10):
    setup_directories()

    pdf_files = get_pdf_files()
    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}")
        return

    # Skip already-processed files
    processed = get_processed_files()
    pending = [f for f in pdf_files if Path(f).stem not in processed]

    if not pending:
        print(f"All {len(pdf_files)} PDFs already processed. Nothing to do.")
        return

    batch = pending[:batch_size]
    print(f"Processing {len(batch)} PDFs (skipping {len(processed)} already done)...")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Remaining after this batch: {len(pending) - len(batch)}")

    success_count = 0
    fail_count = 0

    for pdf_path in tqdm(batch, desc="OCR extraction"):
        filename = Path(pdf_path).stem
        text, error = extract_text_with_retry(pdf_path)

        if error:
            log_result(filename, "failed", error)
            fail_count += 1
            print(f"\nFAILED: {filename} - {error}")
        else:
            #save_json(filename, {
            #    "source_file": filename + ".pdf",
            #    "raw_text": text,
            #    "extraction_status": "success",
            #    "error_message": None
            #})
            save_md(filename, text)
            log_result(filename, "success")
            success_count += 1

    print(f"\n--- Batch Complete ---")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 9999
    process_batch(batch_size)