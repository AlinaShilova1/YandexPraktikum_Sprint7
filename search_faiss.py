import argparse, json, os
from pathlib import Path
import faiss
import numpy as np

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def load_meta(meta_path: Path):
    metas = []
    with meta_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metas.append(json.loads(line))
    # мапа id -> meta
    by_id = {m["id"]: m for m in metas if m.get("id") is not None}
    return metas, by_id

def embed_query_local(query: str, model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
    return vec  # [1, dim]

def embed_query_openai(query: str, model_name: str) -> np.ndarray:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не задан")
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=model_name, input=[query])
    vec = np.array([resp.data[0].embedding], dtype="float32")
    return vec

def main():
    ap = argparse.ArgumentParser(description="Search FAISS index (kNN) with a text query")
    ap.add_argument("--index", default="./index/faiss.index")
    ap.add_argument("--meta",  default="./index/meta.jsonl")
    ap.add_argument("--info",  default="./index/info.json")
    ap.add_argument("--query", required=True, help="Текст запроса")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Модель для векторизации запроса (локально)")
    ap.add_argument("--openai", action="store_true", help="Использовать OpenAI embeddings для запроса")
    ap.add_argument("--openai-model", default="text-embedding-3-small", help="Если --openai, какая модель")
    args = ap.parse_args()

    info = json.loads(Path(args.info).read_text(encoding="utf-8"))
    metric = info.get("metric", "cosine")
    dim    = int(info.get("dim"))

    index = faiss.read_index(args.index)
    metas, by_id = load_meta(Path(args.meta))

    if args.openai:
        q = embed_query_openai(args.query, args.openai_model)
        if metric == "cosine":
            q = l2_normalize(q)
    else:
        q = embed_query_local(args.query, args.model)
        if metric == "cosine":
            q = l2_normalize(q)

    D, I = index.search(q.astype("float32"), args.topk)  # distances & indices
    print(f"\nQuery: {args.query}\n")
    for rank, (dist, idx) in enumerate(zip(D[0], I[0]), start=1):
        if idx < 0:
            continue
        meta = metas[idx]
        score = float(dist)
        print(f"[{rank}] score={score:.4f}  source={meta.get('source')}  chunk={meta.get('chunk_index')}  "
              f"pos=({meta.get('start_char')},{meta.get('end_char')})  id={meta.get('id')}")
        if "text" in meta and meta["text"]:
            snippet = meta["text"].strip().replace("\n", " ")
            print(f"     {snippet[:300]}{'…' if len(snippet)>300 else ''}")

if __name__ == "__main__":
    main()
