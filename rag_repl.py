#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1 if v.ndim == 2 else 0, keepdims=True) + 1e-12
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
        if idx < 0:
            continue
        out.append((int(idx), float(dist), metas[int(idx)]))
    return out

FEW_SHOTS = [
    {"role": "user", "content": "Как называется оружие Мина?"},
    {"role": "assistant", "content": "Оружие Мина называется Нохалбур."},
    {"role": "user", "content": "В какой стране родилась Эйменория?"},
    {"role": "assistant", "content": "Эйменория родом из Элвастана."},
]

SYSTEM_PROMPT = (
    "Ты русскоязычный помощник по базе знаний. Сначала размышляй шаг за шагом, "
    "но не раскрывай внутренние рассуждения пользователю. В ответе верни только: "
    "1) краткий ответ; 2) два пункта пояснений; 3) источники с позициями. "
    "Опирайся исключительно на предоставленные фрагменты контекста."
)

def format_context(hits: List[Tuple[int, float, Dict]], max_chars_per_chunk: int = 900) -> str:
    blocks = []
    for rank, (_, score, m) in enumerate(hits, start=1):
        txt = (m.get("text") or "").strip()
        if len(txt) > max_chars_per_chunk:
            txt = txt[:max_chars_per_chunk] + "…"
        src = m.get("source")
        pos = f"{m.get('start_char')},{m.get('end_char')}"
        blocks.append(f"[CTX {rank}] score={score:.4f} source={src} pos=({pos})\n{txt}")
    return "\n\n".join(blocks)

def build_prompt(query: str, context_blocks: str) -> List[Dict[str, str]]:
    task = (
        f"Контекст:\n{context_blocks}\n\n"
        f"Вопрос пользователя: {query}\n\n"
        f"Формат ответа строго:\n"
        f"- Ответ: <1–2 фразы>\n"
        f"- Объяснение:\n"
        f"  • пункт 1\n"
        f"  • пункт 2\n"
        f"- Источники:\n"
        f"  • <source #chunk/позиции>\n"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}] + FEW_SHOTS + [{"role": "user", "content": task}]

def generate_openai(messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не задан.")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=600
    )
    return resp.choices[0].message.content.strip()

def generate_llama_cpp(messages: List[Dict[str, str]]) -> str:
    from llama_cpp import Llama
    gguf = os.getenv("GGUF_MODEL")
    if not gguf or not Path(gguf).exists():
        raise RuntimeError("GGUF_MODEL не задан или файл не найден.")
    llm = Llama(
        model_path=gguf,
        chat_format="qwen",
        n_ctx=2048,
        n_threads=max(2, (os.cpu_count() or 4) // 2),
        verbose=False
    )
    out = llm.create_chat_completion(
        messages=messages,
        temperature=0.2,
        top_p=0.9,
        max_tokens=512,
        repeat_penalty=1.15
    )
    return out["choices"][0]["message"]["content"].strip()

def main():
    ap = argparse.ArgumentParser(description="RAG REPL over FAISS index")
    ap.add_argument("--index", default="./index/faiss.index")
    ap.add_argument("--meta", default="./index/meta.jsonl")
    ap.add_argument("--info", default="./index/info.json")
    ap.add_argument("--emb-model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--backend", choices=["llama", "openai"], default="llama")
    args = ap.parse_args()

    index, info, metas = load_index(Path(args.index), Path(args.info), Path(args.meta))
    metric = info.get("metric", "cosine")
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
            print("⛔ Ничего не найдено в индексе.")
            continue

        ctx = format_context(hits, max_chars_per_chunk=900)
        messages = build_prompt(q, ctx)

        try:
            if args.backend == "openai":
                answer = generate_openai(messages)
            else:
                answer = generate_llama_cpp(messages)
        except Exception as e:
            print(f"⚠️ Ошибка генерации: {e}")
            continue

        print("\n" + "=" * 80)
        print(answer)
        print("=" * 80)

if __name__ == "__main__":
    main()
