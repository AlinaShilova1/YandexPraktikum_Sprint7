from future import annotations
import argparse, os, json, time, math
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

def batched(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

def load_chunks(jsonl_path: Path):
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                yield rec

def main():
    ap = argparse.ArgumentParser(description="Compute embeddings for chunks and store with metadata")
    ap.add_argument("--chunks", default="./chunks/chunks.jsonl", help="Path to chunks.jsonl")
    ap.add_argument("--out-parquet", default="./vectors/embeddings.parquet", help="Output Parquet file")
    ap.add_argument("--out-jsonl", default="./vectors/embeddings.jsonl", help="Optional JSONL with vectors")
    ap.add_argument("--model", default="text-embedding-3-small", help="Embedding model id")
    ap.add_argument("--batch-size", type=int, default=128, help="Batch size per API call")
    ap.add_argument("--include-text", action="store_true", help="Store chunk text in outputs")
    args = ap.parse_args()

    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не задан. В Codespaces выполните: export OPENAI_API_KEY=sk-...")

    client = OpenAI(api_key=api_key)

    chunks_path = Path(args.chunks)
    out_parquet = Path(args.out_parquet)
    out_jsonl   = Path(args.out_jsonl)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    rows_for_jsonl: List[Dict[str, Any]] = []

    # Однопроходно читаем, чтобы не держать всё в памяти
    all_recs = list(load_chunks(chunks_path))
    total = len(all_recs)
    print(f"[INFO] chunks: {total}; model: {args.model}; batch={args.batch_size}")

    # Подготовка входа: inputs и соответствующие метаданные
    inputs: List[str] = [r["text"] for r in all_recs]
    metas:  List[Dict[str, Any]] = all_recs

    # Вызовы по батчам с простым бэкоффом на случай 429/5xx
    done = 0
    for batch_idx, batch in enumerate(batched(list(zip(inputs, metas)), args.batch_size), start=1):
        batch_inputs = [x[0] for x in batch]
        batch_metas  = [x[1] for x in batch]
        for attempt in range(6):
            try:
                resp = client.embeddings.create(model=args.model, input=batch_inputs)
                vectors = [d.embedding for d in resp.data]
                break
            except Exception as e:
                wait = 1.5 ** attempt
                print(f"[WARN] embeddings call failed (attempt {attempt+1}): {e}; sleep {wait:.1f}s")
                time.sleep(wait)
        else:
            raise RuntimeError("Не удалось получить эмбеддинги после повторов.")

        for vec, meta in zip(vectors, batch_metas):
            row = {
                "id": meta.get("id"),
                "doc_id": meta.get("doc_id"),
                "source": meta.get("source"),
                "chunk_index": meta.get("chunk_index"),
                "start_char": meta.get("start_char"),
                "end_char": meta.get("end_char"),
                "embedding": vec,
            }
            if args.include_text:
                row["text"] = meta.get("text")
            records.append(row)

            # JSONL — удобен для стриминга в БД
            rows_for_jsonl.append(row)

        done += len(batch)
        print(f"[OK] batch {batch_idx}: +{len(batch)} (total {done}/{total})")

    df = pd.DataFrame(records)
    # Для корректного сохранения списка float в Parquet нужен pyarrow
    try:
        import pyarrow as pa  # noqa: F401
        df.to_parquet(out_parquet, index=False)
    except Exception:
        print("[WARN] pyarrow не установлен; сохраняю как CSV (embedding будет обрезан до строки). Установите: pip install pyarrow")
        df.to_csv(out_parquet.with_suffix(".csv"), index=False, encoding="utf-8")

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows_for_jsonl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[DONE] saved: {out_parquet} (+ JSONL: {out_jsonl}); vectors={len(records)}")

if __name__ == "__main__":
    main()
