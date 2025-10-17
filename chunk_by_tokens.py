from pathlib import Path
import os
import subprocess
import hashlib
import json
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

IN_DIR = Path("./knowledge_base")
OUT_JSONL = Path("./chunks/chunks.jsonl")
OUT_CSV = Path("./chunks/chunks.csv")

def get_repo_root() -> Path:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True
        ).strip()
        return Path(out)
    except Exception:
        return Path.cwd()

REPO_ROOT = get_repo_root()

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=200,
    encoding_name="cl100k_base",
    separators=["\n\n", "\n", " ", ""],
    add_start_index=True,
)

def stable_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

rows = []
with OUT_JSONL.open("w", encoding="utf-8") as fjsonl:
    for path in tqdm(sorted(IN_DIR.glob("*.txt"))):
        text = path.read_text(encoding="utf-8", errors="ignore")
        doc_id = stable_id(str(path.resolve()))
        try:
            source_rel = os.path.relpath(str(path.resolve()), start=str(REPO_ROOT))
        except Exception:
            source_rel = path.as_posix()

        docs = splitter.create_documents(
            texts=[text],
            metadatas=[{"source": source_rel}],
        )

        for idx, d in enumerate(docs):
            meta = d.metadata or {}
            start = int(meta.get("start_index", 0))
            end = start + len(d.page_content)
            rec = {
                "id": f"{doc_id}:{idx}",
                "doc_id": doc_id,
                "source": meta.get("source", source_rel),
                "chunk_index": idx,
                "start_char": start,
                "end_char": end,
                "text": d.page_content
            }
            fjsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append(rec)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"Done: {OUT_JSONL} ({len(rows)} чанков), {OUT_CSV}")
