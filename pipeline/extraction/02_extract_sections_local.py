import os
import json
import glob
import re
from pathlib import Path
from tqdm import tqdm
from llama_cpp import Llama

MODEL_PATH = r"C:\LM_Models\Qwen3.5-9B-UD-Q6_K_XL.gguf"

INPUT_DIR = r".\extracted_data\raw"
OUTPUT_DIR = r".\extracted_data\structured"
LOG_FILE = r".\extracted_data\section_extraction_log.txt"

SECTIONS = [
    "ยานี้คืออะไร",
    "ข้อควรรู้ก่อนใช้ยา",
    "วิธีใช้ยา",
    "ข้อควรปฏิบัติระหว่างใช้ยา",
    "อันตรายที่อาจเกิดจากยา",
    "ควรเก็บยานี้อย่างไร",
    "ลักษณะและส่วนประกอบของยา"
]

SECTION_PROMPT = f"""คุณคือผู้เชี่ยวชาญในการแยกส่วนข้อมูลยา จากเอกสาร markdown ต่อไปนี้ ให้แยกข้อมูลออกเป็นหมวดหมู่ 7 หมวดดังนี้:

1. ยานี้คืออะไร - ชื่อยา คำอธิบายทั่วไป
2. ข้อควรรู้ก่อนใช้ยา - ข้อห้าม ข้อควรระวัง
3. วิธีใช้ยา - การรับประทาน การใช้
4. ข้อควรปฏิบัติระหว่างใช้ยา - ผลข้างเคียง การปฏิบัติ
5. อันตรายที่อาจเกิดจากยา - ผลข้างเคียงที่รุนแรง
6. ควรเก็บยานี้อย่างไร - การเก็บรักษา
7. ลักษณะและส่วนประกอบของยา - ส่วนประกอบ ลักษณะภายนอก

คืนค่าเป็น JSON เท่านั้น ไม่ต้องมี markdown code block ใช้รูปแบบดังนี้:
{{
  "pill_name": "ชื่อยา",
  "sections": {{
    "ยานี้คืออะไร": "",
    "ข้อควรรู้ก่อนใช้ยา": "",
    "วิธีใช้ยา": "",
    "ข้อควรปฏิบัติระหว่างใช้ยา": "",
    "อันตรายที่อาจเกิดจากยา": "",
    "ควรเก็บยานี้อย่างไร": "",
    "ลักษณะและส่วนประกอบของยา": ""
  }},
  "source_file": "ชื่อไฟล์ต้นฉบับ",
  "extraction_status": "success"
}}

ถ้าไม่พบข้อมูลในหมวดใด ให้ใส่ "ไม่พบข้อมูล" """

print("Loading model into RX 6800 XT VRAM...")
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=8192,
    chat_format="chatml",
    verbose=True
)


def setup_directories():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def get_raw_md_files():
    md_files = glob.glob(os.path.join(INPUT_DIR, "*.md"))
    return sorted(md_files)


def call_llama_cpp(raw_text):
    try:
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": f"{SECTION_PROMPT}\n\nเอกสาร:\n\n{raw_text[:4000]}"
                }
            ]
        )
        return response['choices'][0]['message']['content'], None
    except Exception as e:
        return None, str(e)


def parse_json_response(response_text):
    try:
        json_str = response_text.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()

        data = json.loads(json_str)
        return data, None
    except json.JSONDecodeError as e:
        try:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return data, None
        except:
            pass
        return None, f"JSON parse error: {str(e)}"


def save_structured_json(filename, data):
    output_path = os.path.join(OUTPUT_DIR, f"{filename}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def log_result(filename, status, error=None):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        if status == "success":
            f.write(f"OK: {filename}\n")
        else:
            f.write(f"FAIL: {filename} - {error}\n")


def process_batch(batch_size=10):
    setup_directories()

    md_files = get_raw_md_files()
    if not md_files:
        print(f"No MD files found in {INPUT_DIR}")
        return

    batch = md_files[:batch_size]
    print(f"Processing {len(batch)} MD files...")
    print(f"Model: {MODEL_PATH}")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    success_count = 0
    fail_count = 0

    for md_path in tqdm(batch, desc="Extracting sections"):
        filename = Path(md_path).stem

        with open(md_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        source_file = filename + ".pdf"

        response_text, error = call_llama_cpp(raw_text)

        if error:
            log_result(filename, "failed", error)
            fail_count += 1
            print(f"\nFAILED: {filename} - {error}")
            continue

        structured_data, parse_error = parse_json_response(response_text)

        if parse_error:
            save_structured_json(filename, {
                "pill_name": "unknown",
                "sections": {section: "การแยกส่วนล้มเหลว" for section in SECTIONS},
                "source_file": source_file,
                "extraction_status": "failed",
                "error_message": parse_error,
                "raw_response": response_text[:500]
            })
            log_result(filename, "failed", parse_error)
            fail_count += 1
            print(f"\nFAILED: {filename} - {parse_error}")
            continue

        structured_data["source_file"] = source_file
        structured_data["extraction_status"] = "success"

        save_structured_json(filename, structured_data)
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