import os
import json
import glob
from pathlib import Path
from tqdm import tqdm
from llama_index.core import SimpleDirectoryReader

INPUT_DIR = r".\fda_pdfs"
OUTPUT_DIR = r".\extracted_data\raw"
LOG_FILE = r".\extracted_data\extraction_log.txt"


def setup_directories():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def get_pdf_files():
    pdf_files = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    return sorted(pdf_files)


def extract_text_from_pdf(pdf_path):
    try:
        reader = SimpleDirectoryReader(input_files=[pdf_path])
        documents = reader.load_data()

        full_text = ""
        for doc in documents:
            full_text += doc.text + "\n\n"

        return full_text, len(documents), None
    except Exception as e:
        return None, 0, str(e)


def save_json(filename, data):
    json_path = os.path.join(OUTPUT_DIR, f"{filename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_md(filename, text, page_count):
    md_path = os.path.join(OUTPUT_DIR, f"{filename}.md")
    content = f"# {filename}\n\nTotal pages: {page_count}\n\n---\n\n{text}"
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

    batch = pdf_files[:batch_size]
    print(f"Processing {len(batch)} PDFs...")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    success_count = 0
    fail_count = 0

    for pdf_path in tqdm(batch, desc="Extracting text"):
        filename = Path(pdf_path).stem
        text, page_count, error = extract_text_from_pdf(pdf_path)

        if error:
            log_result(filename, "failed", error)
            fail_count += 1
            print(f"\nFAILED: {filename} - {error}")
        else:
            save_json(filename, {
                "source_file": filename + ".pdf",
                "total_pages": page_count,
                "raw_text": text,
                "extraction_status": "success",
                "error_message": None
            })
            save_md(filename, text, page_count)
            log_result(filename, "success")
            success_count += 1

    print(f"\n--- Batch Complete ---")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    import sys
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    process_batch(batch_size)