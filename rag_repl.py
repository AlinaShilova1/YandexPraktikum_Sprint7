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

# ---------------- Retrieval ----------------
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

# ---------------- Prompt building ----------------
BASE_FEW_SHOTS = [
    {"role": "user", "content": "Как называется столица планеты Ти’лора?"},
    {"role": "assistant", "content": "Столица планеты Ти’лора называется Сайрон."},
    {"role": "user", "content": "Какая энергия питает узел HyperRelay?"},
    {"role": "assistant", "content": "Согласно документу, HyperRelay питается от ядра VoidCore."},
]

SYSTEM_PROMPT = (
    "Ты русскоязычный помощник по базе знаний. Сначала размышляй шаг за шагом, "
    "но не раскрывай внутренние рассуждения пользователю. В ответе верни только: "
    "1) краткий ответ; 2) два пункта пояснений; 3) источники с позициями. "
    "Опирайся исключительно на предоставленные фрагменты контекста."
)

def format_context(hits: List[Tuple[int, float, Dict]], max_chars_per_chunk: int) -> str:
    blocks = []
    for rank, (_, score, m) in enumerate(hits, start=1):
        txt = (m.get("text") or "").strip()
        if len(txt) > max_chars_per_chunk:
            txt = txt[:max_chars_per_chunk] + "…"
        src = m.get("source")
        pos = f"{m.get('start_char')},{m.get('end_char')}"
        blocks.append(f"[CTX {rank}] score={score:.4f} source={src} pos=({pos})\n{txt}")
    return "\n\n".join(blocks)

def build_messages(query: str, context_blocks: str, shots_pairs: int) -> List[Dict[str, str]]:
    shots = BASE_FEW_SHOTS[: max(0, shots_pairs * 2)]
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
    return [{"role": "system", "content": SYSTEM_PROMPT}] + shots + [{"role": "user", "content": task}]

# ---------------- LLM ----------------
class LlamaBackend:
    def init(self, model_path: str, chat_format: str = "qwen", n_ctx: int = 4096, n_threads: int = 4):
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=model_path,
            chat_format=chat_format,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False
        )
        self.n_ctx = n_ctx

    def tokenize_len(self, text: str) -> int:
        toks = self.llm.tokenize(text.encode("utf-8"))
        return len(toks)

    def messages_token_len(self, messages: List[Dict[str, str]]) -> int:
        joined = ""
        for m in messages:
            joined += f"{m.get('role','user').strip()}: {m.get('content','').strip()}\n"
        return self.tokenize_len(joined)

    def generate(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        out = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.2,
            top_p=0.9,
            max_tokens=max_tokens,
            repeat_penalty=1.15
        )
        return out["choices"][0]["message"]["content"].strip()

# ---------------- REPL ----------------
def main():
    ap = argparse.ArgumentParser(description="RAG REPL (FAISS + llama.cpp)")
    ap.add_argument("--model-path", required=True, help="Путь к GGUF-модели")
    ap.add_argument("--chat-format", default="qwen")
    ap.add_argument("--index", default="./index/faiss.index")
    ap.add_argument("--meta", default="./index/meta.jsonl")
    ap.add_argument("--info", default="./index/info.json")
    ap.add_argument("--emb-model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--shots", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--max-out", type=int, default=512)
    ap.add_argument("--chunk-chars", type=int, default=900)
    args = ap.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"GGUF-модель не найдена: {model_path}")

    index, info, metas = load_index(Path(args.index), Path(args.info), Path(args.meta))
    metric = info.get("metric", "cosine")
    print(f"[READY] index loaded; metric={metric}; chunks={len(metas)}")

    emb_model = SentenceTransformer(args.emb_model, device="cpu")

    n_threads = max(2, (os.cpu_count() or 4) // 2)
    backend = LlamaBackend(model_path=str(model_path), chat_format=args.chat_format, n_ctx=args.ctx, n_threads=n_threads)
    print(f"[LLM] llama.cpp backend; ctx={args.ctx}; threads={n_threads}; model={model_path.name}")

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
            print("⛔ Ничего не найдено.")
            continue

        ctx_text = format_context(hits, max_chars_per_chunk=args.chunk_chars)
        messages = build_messages(q, ctx_text, shots_pairs=args.shots)

        try:
            answer = backend.generate(messages, max_tokens=args.max_out)
        except Exception as e:
            print(f"⚠️ Ошибка генерации: {e}")
            continue

        print("\n" + "=" * 80)
        print(answer)
        print("=" * 80)

if __name__ == "__main__":
    main()
