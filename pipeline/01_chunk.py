"""
01_chunk.py  -  Medication .md -> hierarchical JSON chunks

Output format per chunk:
{
  "text": "[Cefoperazone] ยานี้คือยาอะไร > ยานี้มีชื่อว่าอะไร\n\nยานี้มีชื่อว่า...\nเป็นยาฆ่าเชื้อ...",
  "data": {
    "topic":    "ยานี้คือยาอะไร",
    "subtopic": "ยานี้มีชื่อว่าอะไร",
    "detail":   ["ยานี้มีชื่อว่า เซ-โฟ-เพอ-รา-โซน (cefoperazone)", "เป็นยาฆ่าเชื้อแบคทีเรีย..."]
  },
  "metadata": {
    "pill_name":   "Cefoperazone",
    "section":     "ยานี้คือยาอะไร",
    "subsection":  "ยานี้มีชื่อว่าอะไร",
    "source_file": "1. Cefoperazone_injection_PIL_final.md"
  }
}

text   -> embedded into vector store (pill + topic + subtopic + joined detail)
data   -> structured for UI display (detail as bullet array)
metadata -> stored in ChromaDB for filtering (section, subsection, pill_name)

When a section has no subsections: subtopic and subsection are null.

Parsing strategy:
  1. Regex: split on numbered headers "1." and "1.1" (handles ## and plain text)
  2. If regex finds < MIN_SECTIONS -> LLM fallback
  3. If LLM also fails -> store whole doc as one chunk
"""

import os
import re
import json
import glob
from pathlib import Path
from tqdm import tqdm
import ollama

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_DIR  = r".\extracted_data\raw"
OUTPUT_DIR = r".\extracted_data\chunks"
LOG_FILE   = r".\extracted_data\chunk_log.txt"

OLLAMA_MODEL = "qwen2.5:7b"   # non-thinking variant - no <think> tags
MIN_SECTIONS = 3               # fall back to LLM if regex finds fewer top-level sections

STANDARD_SECTIONS = [
    "ยานี้คือยาอะไร",
    "ข้อควรรู้ก่อนใช้ยา",
    "วิธีใช้ยา",
    "ข้อควรปฏิบัติระหว่างใช้ยา",
    "อันตรายที่อาจเกิดจากยา",
    "ควรเก็บยานี้อย่างไร",
    "ลักษณะและส่วนประกอบของยา",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_pill_name(filename: str) -> str:
    """Pull the English drug name from the filename.
    e.g. '1. Cefoperazone_injection_PIL_final.md' -> 'Cefoperazone'
    """
    stem = Path(filename).stem
    stem = re.sub(r'^\d+\.?\s*', '', stem)   # strip "1. "
    parts = re.split(r'[_\s]', stem)
    return parts[0].strip() if parts else stem.strip()


def parse_detail_lines(raw: str) -> list:
    """
    Convert raw markdown content into a clean bullet list.
    Strips '* ' and '- ' markers and removes empty lines.
    """
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[*\-]\s+', '', line)   # strip markdown bullet markers
        if line:
            lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Regex-based hierarchical parser
# ---------------------------------------------------------------------------

# Matches MAIN sections: "## 1. ยา..." or "1. ยา..." (single integer, Thai/alpha title)
MAIN_SECTION_RE = re.compile(
    r'^#{0,2}\s*(\d+)\.\s+([\u0E00-\u0E7Fa-zA-Z][^\n]*)',
    re.MULTILINE
)

# Matches SUBSECTIONS: "### 1.1. ยา..." or "1.1 ยา..."
SUBSECTION_RE = re.compile(
    r'^#{0,3}\s*(\d+\.\d+)\.?\s+([\u0E00-\u0E7Fa-zA-Z][^\n]*)',
    re.MULTILINE
)


def is_main_section(match: re.Match, text: str) -> bool:
    """Guard: skip if the line is actually a subsection pattern (e.g. '1.1')."""
    line_start = text[match.start(): match.start() + 15]
    if re.match(r'#{0,2}\s*\d+\.\d+', line_start):
        return False
    num = int(match.group(1))
    return 1 <= num <= 7


def strip_title_line(md_text: str) -> str:
    """Remove the top-level '# ...' title line so it doesn't become a junk chunk."""
    return re.sub(r'^#[^#][^\n]*\n', '', md_text, count=1).strip()


def regex_parse(md_text: str) -> list:
    """
    Parse md_text into:
    [
      {
        "section_title": "ข้อควรรู้ก่อนใช้ยา",
        "intro": "",                           <- text before first subsection
        "subsections": [
          {"subsection_title": "ห้ามใช้ยานี้เมื่อไร", "content": "..."},
          {"subsection_title": "ข้อควรระวัง",          "content": "..."},
        ]
      },
      ...
    ]
    """
    md_text = strip_title_line(md_text)
    main_matches = [m for m in MAIN_SECTION_RE.finditer(md_text) if is_main_section(m, md_text)]

    if not main_matches:
        return []

    sections = []
    for i, main_m in enumerate(main_matches):
        title        = main_m.group(2).strip()
        body_start   = main_m.end()
        body_end     = main_matches[i + 1].start() if i + 1 < len(main_matches) else len(md_text)
        body         = md_text[body_start:body_end]

        sub_matches  = list(SUBSECTION_RE.finditer(body))
        intro        = body[:sub_matches[0].start()].strip() if sub_matches else body.strip()

        subsections  = []
        for j, sub_m in enumerate(sub_matches):
            sub_title   = sub_m.group(2).strip()
            sub_start   = sub_m.end()
            sub_end     = sub_matches[j + 1].start() if j + 1 < len(sub_matches) else len(body)
            sub_content = body[sub_start:sub_end].strip()
            subsections.append({"subsection_title": sub_title, "content": sub_content})

        sections.append({"section_title": title, "intro": intro, "subsections": subsections})

    return sections


# ---------------------------------------------------------------------------
# LLM fallback parser
# ---------------------------------------------------------------------------

LLM_PROMPT = """\
คุณคือผู้เชี่ยวชาญด้านยา แยกเนื้อหาต่อไปนี้เป็น 7 หมวดหลัก
และในแต่ละหมวดให้แยกเป็นหมวดย่อยพร้อมรายการข้อมูลย่อย
ส่งคืนเป็น JSON เท่านั้น ห้ามมี markdown code block

รูปแบบ JSON:
{
  "sections": [
    {
      "section_title": "ยานี้คือยาอะไร",
      "subsections": [
        {
          "subsection_title": "ยานี้มีชื่อว่าอะไร",
          "detail": ["ยานี้มีชื่อว่า ...", "เป็นยาในกลุ่ม ..."]
        }
      ]
    }
  ]
}

ถ้าหมวดใดไม่มีหมวดย่อย ให้ใส่ subsections เป็น []
และเพิ่ม "detail": ["..."] ไว้ใน section โดยตรง
ถ้าไม่พบข้อมูลในหมวดใด ให้ข้ามหมวดนั้นไป
"""


def strip_think_tags(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def llm_parse(md_text: str) -> list | None:
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": f"{LLM_PROMPT}\n\nเอกสาร:\n\n{md_text[:4000]}"}]
        )
        raw = strip_think_tags(response['message']['content'])
        raw = re.sub(r'^```(?:json)?', '', raw.strip(), flags=re.MULTILINE)
        raw = re.sub(r'```$', '', raw.strip(), flags=re.MULTILINE)
        data = json.loads(raw.strip())
        return data.get("sections", [])
    except Exception as e:
        print(f"    LLM error: {e}")
        return None


# ---------------------------------------------------------------------------
# Chunk builder
# ---------------------------------------------------------------------------

def build_chunks(sections: list, pill_name: str, source_file: str) -> list:
    """
    Convert parsed section hierarchy into flat list of RAG chunks.

    Each chunk:
      text     = "[pill] topic > subtopic\n\nbullet1\nbullet2"   <- for embedding + LLM
      data     = { topic, subtopic, detail: [...] }              <- for UI display
      metadata = { pill_name, section, subsection, source_file } <- for ChromaDB filtering
    """
    chunks = []

    for sec in sections:
        section_title = sec.get("section_title", "").strip()
        subsections   = sec.get("subsections", [])
        intro         = sec.get("intro", "").strip()

        if not section_title:
            continue

        if subsections:
            # --- One chunk per subsection ---
            for sub in subsections:
                sub_title = sub.get("subsection_title", "").strip()

                # LLM path gives detail directly; regex path gives raw content string
                if "detail" in sub:
                    detail = [d.strip() for d in sub["detail"] if d.strip()]
                else:
                    raw    = sub.get("content", "").strip()
                    combined = f"{intro}\n\n{raw}" if intro else raw
                    detail = parse_detail_lines(combined)

                if not detail:
                    continue

                text = f"[{pill_name}] {section_title} > {sub_title}\n\n" + "\n".join(detail)

                chunks.append({
                    "text": text,
                    "data": {
                        "topic":    section_title,
                        "subtopic": sub_title,
                        "detail":   detail,
                    },
                    "metadata": {
                        "pill_name":   pill_name,
                        "section":     section_title,
                        "subsection":  sub_title,
                        "source_file": source_file,
                    }
                })
        else:
            # --- No subsections: one chunk for the whole section ---
            if "detail" in sec:
                detail = [d.strip() for d in sec["detail"] if d.strip()]
            else:
                raw    = intro or sec.get("content", "")
                detail = parse_detail_lines(raw)

            if not detail:
                continue

            text = f"[{pill_name}] {section_title}\n\n" + "\n".join(detail)

            chunks.append({
                "text": text,
                "data": {
                    "topic":    section_title,
                    "subtopic": None,
                    "detail":   detail,
                },
                "metadata": {
                    "pill_name":   pill_name,
                    "section":     section_title,
                    "subsection":  None,
                    "source_file": source_file,
                }
            })

    return chunks


# ---------------------------------------------------------------------------
# File processor
# ---------------------------------------------------------------------------

def process_file(md_path: str) -> tuple:
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    filename  = Path(md_path).name
    pill_name = extract_pill_name(filename)

    # Step 1: regex
    sections = regex_parse(md_text)
    if len(sections) >= MIN_SECTIONS:
        return build_chunks(sections, pill_name, filename), "regex"

    # Step 2: LLM fallback
    print(f"\n    [{filename}] regex found {len(sections)} section(s) -> LLM fallback")
    sections = llm_parse(md_text)
    if sections:
        return build_chunks(sections, pill_name, filename), "llm"

    # Step 3: hard fallback
    print(f"    [{filename}] LLM failed -> single-chunk fallback")
    return [{
        "text": md_text,
        "data": {"topic": "full_document", "subtopic": None, "detail": [md_text]},
        "metadata": {"pill_name": pill_name, "section": "full_document",
                     "subsection": None, "source_file": filename}
    }], "fallback"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def process_all():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("method   | total | sub_chunks | filename\n")
        f.write("-" * 65 + "\n")

    md_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.md")))
    if not md_files:
        print(f"No .md files found in {INPUT_DIR}")
        return

    print(f"Found {len(md_files)} .md file(s)\nOutput -> {OUTPUT_DIR}\n")

    stats = {"regex": 0, "llm": 0, "fallback": 0, "total_chunks": 0, "sub_chunks": 0}

    for md_path in tqdm(md_files, desc="Chunking"):
        filename       = Path(md_path).stem
        chunks, method = process_file(md_path)

        with open(os.path.join(OUTPUT_DIR, f"{filename}.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        sub_count             = sum(1 for c in chunks if c["metadata"].get("subsection"))
        stats[method]        += 1
        stats["total_chunks"] += len(chunks)
        stats["sub_chunks"]  += sub_count

        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(f"{method.upper():<8} | {len(chunks):>5} | {sub_count:>10} | {filename}\n")

    print(f"\n--- Done ---")
    print(f"Regex    : {stats['regex']} files")
    print(f"LLM      : {stats['llm']} files")
    print(f"Fallback : {stats['fallback']} files")
    print(f"Total chunks    : {stats['total_chunks']}")
    print(f"Subtopic chunks : {stats['sub_chunks']}")
    print(f"Output -> {OUTPUT_DIR}")


if __name__ == "__main__":
    process_all()
