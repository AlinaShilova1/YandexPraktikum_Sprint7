# -*- coding: utf-8 -*-
import os, json, argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------- Retrieval ----------
def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1 if v.ndim==2 else 0, keepdims=True) + 1e-12
    return v / n

def load_index(index_path: Path, info_path: Path, meta_path: Path):
    index = faiss.read_index(str(index_path))
    info = json.loads(info_path.read_text(encoding="utf-8"))
    metas = []
    with meta_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metas.append(json.loads(line))
    return index, info, metas

def embed_query(text: str, model: SentenceTransformer, metric: str) -> np.ndarray:
    vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
    if metric == "cosine":
        vec = l2_normalize(vec)
    return vec

def retrieve(index, metas, qvec: np.ndarray, topk: int) -> List[Tuple[int, float, Dict]]:
    D, I = index.search(qvec, topk)
    out = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0: continue
        out.append((int(idx), float(dist), metas[int(idx)]))
    return out

# ---------- Prompt building ----------
FEW_SHOTS = [
    {
        "q": "Как называется столица планеты Ти’лора?",
        "a": "Столица планеты Ти’лора называется Сайрон."
    },
    {
        "q": "Какая энергия питает узел HyperRelay?",
        "a": "Согласно документу, HyperRelay питается от ядра VoidCore."
    }
]

SYSTEM_PROMPT = (
    "Ты русскоязычный помощник по базе знаний. Сначала думай шаг за шагом, "
    "но НЕ раскрывай ход рассуждений. В ответ выводи только: "
    "1) краткий ответ; 2) краткое объяснение (2–3 пункта); 3) список источников с позициями. "
    "Опирайся исключительно на предоставленные фрагменты контекста."
)

def format_context(hits: List[Tuple[int,float,Dict]], max_chars_per_chunk: int = 1200) -> str:
    blocks = []
    for rank, (_, score, m) in enumerate(hits, start=1):
        txt = (m.get("text") or "").strip()
        if len(txt) > max_chars_per_chunk:
            txt = txt[:max_chars_per_chunk] + "…"
        src = m.get("source")
        pos = f"{m.get('start_char')},{m.get('end_char')}"
        blocks.append(f"[CTX {rank}] score={score:.4f} source={src} pos=({pos})\n{txt}")
    return "\n\n".join(blocks)

def build_prompt(query: str, context_blocks: str) -> str:
    shots = "\n\n".join([f"Q: {s['q']}\nA: {s['a']}" for s in FEW_SHOTS])
    return (
        f"System: {SYSTEM_PROMPT}\n\n"
        f"{shots}\n\n"
        f"Контекст:\n{context_blocks}\n\n"
        f"Вопрос пользователя: {query}\n\n"
        f"Формат ответа строго:\n"
        f"- Ответ: <краткий ответ одной-двумя фразами>\n"
        f"- Объяснение:\n"
        f"  • пункт 1\n"
        f"  • пункт 2\n"
        f"- Источники:\n"
        f"  • <source #chunk/позиции>\n"
    )

# ---------- LLM backends ----------
def generate_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не задан.")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":SYSTEM_PROMPT},
                  {"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=600
    )
    return resp.choices[0].message.content.strip()

def generate_llama_cpp(prompt: str) -> str:
    from llama_cpp import Llama
    gguf = os.getenv("GGUF_MODEL")
    if not gguf or not Path(gguf).exists():
        raise RuntimeError("GGUF_MODEL не задан или файл не найден.")
    llm = Llama(model_path=gguf, n_ctx=4096, n_threads=os.cpu_count() or 4, verbose=False)
    # Универсальный текстовый формат (без спец-токенов чата — работает с большинством instruct-моделей)
    full_prompt = f"{prompt}\nОтвет:"
    out = llm(
        full_prompt,
        temperature=0.2,
        max_tokens=600,
        stop=["\nSystem:", "Q:", "\n\nQ:"]
    )
    return out["choices"][0]["text"].strip()

# ---------- REPL ----------
def main():
    ap = argparse.ArgumentParser(description="RAG REPL over FAISS index")
    ap.add_argument("--index", default="./index/faiss.index")
    ap.add_argument("--meta",  default="./index/meta.jsonl")
    ap.add_argument("--info",  default="./index/info.json")
    ap.add_argument("--emb-model", default="intfloat/multilingual-e5-base", help="SentenceTransformer модель для запроса")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--backend", choices=["llama","openai"], default="llama", help="Генеративный бэкенд")
    args = ap.parse_args()

    index, info, metas = load_index(Path(args.index), Path(args.info), Path(args.meta))
    metric = info.get("metric","cosine")
    print(f"[READY] index loaded; metric={metric}; chunks={len(metas)}")

    emb_model = SentenceTransformer(args.emb_model, device="cpu")

    while True:
        try:
            q = input("\n❓ Вопрос (или /q для выхода): ").strip()
        except EOFError:
            break
        if not q or q == "/q":
            break

        qvec = embed_query(q, emb_model, metric)
        hits = retrieve(index, metas, qvec.astype("float32"), topk=args.topk)
        if not hits:
            print("⛔ Ничего не найдено в индексе."); continue

        ctx = format_context(hits)
        prompt = build_prompt(q, ctx)

        try:
            if args.backend == "openai":
                answer = generate_openai(prompt)
            else:
                answer = generate_llama_cpp(prompt)
        except Exception as e:
            print(f"⚠️ Ошибка генерации: {e}")
            continue

        print("\n" + "="*80)
        print(answer)
        print("="*80)

if __name__ == "__main__":
    main()
