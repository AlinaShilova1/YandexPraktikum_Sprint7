# chunk_by_tokens.py
from pathlib import Path
import json
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
import pandas as pd

IN_DIR = Path("./knowledge_base")     # <- замени при необходимости
OUT_JSONL = Path("./chunks/chunks.jsonl")
OUT_CSV   = Path("./chunks/chunks.csv")

# 1) Сплиттер "по токенам" (пример: 800 токенов, overlap 200)
#   Это хороший баланс для ответов и качества поиска.
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=200,
    encoding_name="cl100k_base",   # токенизатор OpenAI (соответствует text-embedding-3-*)
    separators=["\n\n", "\n", " ", ""],  # логическая и мягкая рекурсивность
)

def stable_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

rows = []
with OUT_JSONL.open("w", encoding="utf-8") as fjsonl:
    for path in tqdm(sorted(IN_DIR.glob("*.txt"))):
        text = path.read_text(encoding="utf-8", errors="ignore")
        doc_id = stable_id(str(path.resolve()))

        # add_start_index=True — положит 'start_index' в metadata (позиция чанка в исходном тексте)
        docs = splitter.create_documents(
            texts=[text],
            metadatas=[{"source": str(path.relative_to(Path.cwd()))}],
            add_start_index=True
        )

        # Превращаем в «ровные» записи
        for idx, d in enumerate(docs):
            meta = d.metadata or {}
            start = int(meta.get("start_index", 0))
            end   = start + len(d.page_content)
            rec = {
                "id": f"{doc_id}:{idx}",
                "doc_id": doc_id,
                "source": meta.get("source", str(path)),
                "chunk_index": idx,
                "start_char": start,
                "end_char": end,
                "text": d.page_content
            }
            fjsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append(rec)

# Для удобного просмотра — CSV (без текста или с укороченным)
df = pd.DataFrame(rows)
# Если файл огромный, можно не писать столбец 'text' в CSV:
# df = df.drop(columns=["text"])
df.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"Done: {OUT_JSONL} ({len(rows)} чанков), {OUT_CSV}")
