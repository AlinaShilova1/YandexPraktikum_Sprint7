import argparse, json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

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
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser(description="Local embeddings with SentenceTransformers")
    ap.add_argument("--chunks", default="./chunks/chunks.jsonl")
    ap.add_argument("--out-parquet", default="./vectors/embeddings.parquet")
    ap.add_argument("--out-jsonl", default="./vectors/embeddings.jsonl")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cpu")  # "cpu" | "cuda"
    ap.add_argument("--include-text", action="store_true")
    args = ap.parse_args()

    model = SentenceTransformer(args.model, device=args.device)
    dim = model.get_sentence_embedding_dimension()

    chunks_path = Path(args.chunks)
    out_parquet = Path(args.out_parquet); out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl = Path(args.out_jsonl);     out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    recs = list(load_chunks(chunks_path))
    texts: List[str] = [r["text"] for r in recs]
    metas: List[Dict[str, Any]] = recs
    print(f"[INFO] chunks: {len(texts)}; model: {args.model}; dim={dim}; batch={args.batch_size}; device={args.device}")

    rows, rows_jsonl = [], []
    for batch in tqdm(batched(list(zip(texts, metas)), args.batch_size), total=(len(texts)+args.batch_size-1)//args.batch_size):
        b_texts = [x[0] for x in batch]
        b_metas = [x[1] for x in batch]
        vecs = model.encode(b_texts, batch_size=len(b_texts), convert_to_numpy=True, normalize_embeddings=False)
        for v, m in zip(vecs, b_metas):
            row = {
                "id": m.get("id"),
                "doc_id": m.get("doc_id"),
                "source": m.get("source"),
                "chunk_index": m.get("chunk_index"),
                "start_char": m.get("start_char"),
                "end_char": m.get("end_char"),
                "embedding": v.tolist(),
                "model": args.model,
                "dim": int(dim),
            }
            if args.include_text:
                row["text"] = m.get("text")
            rows.append(row); rows_jsonl.append(row)

    df = pd.DataFrame(rows)
    try:
        import pyarrow as pa  # noqa
        df.to_parquet(out_parquet, index=False)
    except Exception:
        print("[WARN] pyarrow не установлен или не поддерживается; сохраняю CSV")
        df.to_csv(out_parquet.with_suffix(".csv"), index=False, encoding="utf-8")

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows_jsonl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[DONE] saved: {out_parquet} (+ JSONL: {out_jsonl}); vectors={len(rows)}; dim={dim}")

if __name__ == "__main__":
    main()
