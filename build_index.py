import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm

def load_embeddings(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    elif path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Неизвестный формат: {path.suffix}")

def ensure_float32(mat: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(mat.astype("float32"))

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def main():
    ap = argparse.ArgumentParser(description="Build FAISS index from precomputed embeddings")
    ap.add_argument("--embeddings", default="./vectors/embeddings.jsonl", help="JSONL/Parquet/CSV с эмбеддингами")
    ap.add_argument("--out-dir", default="./index", help="Куда сохранить индекс и метаданные")
    ap.add_argument("--metric", choices=["cosine","ip"], default="cosine", help="Метрика для поиска")
    args = ap.parse_args()

    in_path = Path(args.embeddings)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_index = out_dir / "faiss.index"
    out_meta  = out_dir / "meta.jsonl"
    out_info  = out_dir / "info.json"

    df = load_embeddings(in_path)
    if "embedding" not in df.columns:
        raise RuntimeError("В файле нет колонки 'embedding' (ожидается список чисел на строку)")

    # извлекаем матрицу векторов
    vecs = np.array(df["embedding"].tolist(), dtype="float32")
    dim = vecs.shape[1]
    print(f"[INFO] vectors: {vecs.shape[0]} x {dim}")

    # cosine = dot(normalized(a), normalized(b))
    if args.metric == "cosine":
        vecs = l2_normalize(vecs)
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatIP(dim)  # ip — без нормализации

    index.add(ensure_float32(vecs))
    faiss.write_index(index, str(out_index))

    # сохраним метаданные построчно (id, source, позиции, текст при наличии)
    with out_meta.open("w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="write meta"):
            meta = {
                "id": row.get("id"),
                "doc_id": row.get("doc_id"),
                "source": row.get("source"),
                "chunk_index": int(row.get("chunk_index")) if not pd.isna(row.get("chunk_index")) else None,
                "start_char": int(row.get("start_char"))   if not pd.isna(row.get("start_char")) else None,
                "end_char": int(row.get("end_char"))       if not pd.isna(row.get("end_char")) else None,
            }
            if "text" in df.columns:
                meta["text"] = row.get("text")
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    info = {
        "dim": int(dim),
        "metric": args.metric,
        "source_embeddings_file": str(in_path),
        "index_file": str(out_index),
        "meta_file": str(out_meta),
        "count": int(vecs.shape[0]),
    }
    out_info.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] index: {out_index}")
    print(f"[DONE] meta:  {out_meta}")
    print(f"[DONE] info:  {out_info}")

if __name__ == "__main__":
    main()
