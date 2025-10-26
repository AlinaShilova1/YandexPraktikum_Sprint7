# -*- coding: utf-8 -*-
import argparse, json, re, sys, warnings
from pathlib import Path
from functools import lru_cache

# --- Morph init ---
morph = None
try:
    from pymorphy3 import MorphAnalyzer as MorphAnalyzer3
    morph = MorphAnalyzer3()
except Exception:
    import inspect
    from collections import namedtuple
    from inspect import signature, Parameter
    if not hasattr(inspect, "getargspec"):
        ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")
        def _getargspec(func):
            sig = signature(func)
            params = list(sig.parameters.values())
            args_no_default = [p.name for p in params
                               if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
                               and p.default is Parameter.empty]
            args_with_default = [p.name for p in params
                                 if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
                                 and p.default is not Parameter.empty]
            varargs = next((p.name for p in params if p.kind == Parameter.VAR_POSITIONAL), None)
            varkw  = next((p.name for p in params if p.kind == Parameter.VAR_KEYWORD), None)
            defaults = tuple(p.default for p in params
                             if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
                             and p.default is not Parameter.empty) or None
            return ArgSpec(args_no_default + args_with_default, varargs, varkw, defaults)
        inspect.getargspec = _getargspec  # type: ignore
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated", module="pymorphy2")
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()

SEQUEL_SUFFIX_RE = r'(?:\s*[:\-]?\s*(?:\d+|[IVX]{1,4}))$'
TOKEN_RE = re.compile(r'([A-Za-zА-Яа-яЁё0-9]+|[\-–—]|[^\w\s]|[\s]+)', re.U)
LATIN_RE = re.compile(r'^[A-Za-z]+(?:[-\s][A-Za-z]+)*$')

def tokenize(text: str): return TOKEN_RE.findall(text)
def is_word(tok: str):   return bool(re.match(r'^[A-Za-zА-Яа-яЁё0-9]+$', tok))
def is_space(tok: str):  return tok.isspace()
def is_hyphen(tok: str): return tok in ('-','–','—')
def is_english_string(s: str) -> bool: return bool(LATIN_RE.fullmatch(s.strip()))

def _yo2e(s: str) -> str:
    return s.replace("Ё","Е").replace("ё","е")

@lru_cache(maxsize=200000)
def lemma(w: str) -> str:
    w = _yo2e(w.lower())
    try:
        return _yo2e(morph.parse(w)[0].normal_form)
    except Exception:
        return w

@lru_cache(maxsize=200000)
def possible_lemmas(w: str) -> set[str]:
    """Все нормальные формы для данного слова (с учётом неоднозначности).
       Нормализуем ё→е и нижний регистр, чтобы ловить фамилии типа 'Старка' → 'старк'."""
    w = _yo2e(w.lower())
    try:
        return {_yo2e(p.normal_form) for p in morph.parse(w)}
    except Exception:
        return {w}

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")

def _strip_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"(?m)^\s*//.*?$", "", s)
    return s

def load_mapping(path: Path) -> dict:
    raw = _read_text(path)
    cleaned = _strip_comments(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        import json5
        data = json5.loads(raw)
    data = {k.strip(): v.strip() for k, v in data.items() if k and v and k.strip() and v.strip()}
    return data

def expand_person_aliases(mapping: dict, enable: bool = True) -> dict:
    if not enable: return mapping
    extra = {}
    for src, dst in list(mapping.items()):
        s_parts = re.split(r'[ \-–—]+', src.strip())
        d_parts = re.split(r'[ \-–—]+', dst.strip())
        if len(s_parts) == 2 and len(d_parts) == 2:
            s_first, s_last = s_parts
            d_first, d_last = d_parts
            extra.setdefault(s_first, d_first)
            extra.setdefault(s_last, d_last)
    for k, v in extra.items():
        mapping.setdefault(k, v)
    return mapping

def prepare_entries(mapping: dict):
    entries = []
    for src, dst in mapping.items():
        sequel = re.search(SEQUEL_SUFFIX_RE, src)
        src_clean = re.sub(SEQUEL_SUFFIX_RE, '', src).strip()
        dst_base  = re.sub(SEQUEL_SUFFIX_RE, '', dst).strip()

        src_tokens = re.split(r'[ \-–—]+', src_clean)
        dst_tokens = re.split(r'[ \-–—]+', dst_base)

        # ключи-«ожидаемые леммы» для каждого токена
        src_keys   = [t.lower() if is_english_string(t) else lemma(t) for t in src_tokens]

        entries.append({
            "src": src, "dst": dst,
            "src_tokens": src_tokens, "dst_tokens": dst_tokens,
            "src_keys": src_keys, "sequel_src": sequel.group(0) if sequel else ""
        })
    entries.sort(key=lambda e: len(e["src_tokens"]), reverse=True)
    return entries

def transfer_caps(src_word: str, dst_word: str) -> str:
    if src_word.isupper(): return dst_word.upper()
    if src_word.istitle(): return dst_word[:1].upper() + dst_word[1:]
    return dst_word

def heuristic_inflect(dst_base: str, src_word: str) -> str:
    if is_english_string(dst_base):
        return transfer_caps(src_word, dst_base)
    low = _yo2e(dst_base.lower())
    p = morph.parse(_yo2e(src_word))[0]
    case = getattr(p.tag, "case", None)
    if case in {"gent"}:
        if low.endswith("а"):  return transfer_caps(src_word, dst_base[:-1]+"ы")
        if low.endswith("я"):  return transfer_caps(src_word, dst_base[:-1]+"и")
        if low.endswith("ия"): return transfer_caps(src_word, dst_base[:-2]+"ии")
    if case in {"datv"}:
        if low.endswith("а"):  return transfer_caps(src_word, dst_base[:-1]+"е")
        if low.endswith("я"):  return transfer_caps(src_word, dst_base[:-1]+"е")
        if low.endswith("ия"): return transfer_caps(src_word, dst_base[:-2]+"ии")
    if case in {"accs"}:
        if low.endswith("а"):  return transfer_caps(src_word, dst_base[:-1]+"у")
        if low.endswith("я"):  return transfer_caps(src_word, dst_base[:-1]+"ю")
    if case in {"ablt"}:
        if low.endswith("а"):  return transfer_caps(src_word, dst_base[:-1]+"ой")
        if low.endswith("я"):  return transfer_caps(src_word, dst_base[:-1]+"ей")
        if low.endswith("ия"): return transfer_caps(src_word, dst_base[:-2]+"ией")
    if case in {"loct"}:
        if low.endswith("а"):  return transfer_caps(src_word, dst_base[:-1]+"е")
        if low.endswith("я"):  return transfer_caps(src_word, dst_base[:-1]+"е")
        if low.endswith("ия"): return transfer_caps(src_word, dst_base[:-2]+"ии")
    return transfer_caps(src_word, dst_base)

def inflect_like(src_word: str, dst_base: str) -> str:
    if is_english_string(dst_base):
        return transfer_caps(src_word, dst_base)
    p_src = morph.parse(_yo2e(src_word))[0]
    p_tgt = morph.parse(_yo2e(dst_base))[0]
    need = set()
    if getattr(p_src.tag, "case", None):   need.add(p_src.tag.case)
    if getattr(p_src.tag, "number", None): need.add(p_src.tag.number)
    if getattr(p_tgt.tag, "gender", None) and getattr(p_src.tag, "gender", None):
        need.add(p_src.tag.gender)
    try:
        inflected = p_tgt.inflect(need) if need else None
    except Exception:
        inflected = None
    form = inflected.word if inflected else heuristic_inflect(dst_base, src_word)
    return transfer_caps(src_word, form)

def match_phrase(tokens, i, entry):
    keys = entry["src_keys"]; j = i; matched = []
    for need in keys:
        while j < len(tokens) and (is_space(tokens[j]) or is_hyphen(tokens[j])): j += 1
        if j >= len(tokens) or not is_word(tokens[j]): return None
        cur = tokens[j]

        if is_english_string(cur):
            cur_key = _yo2e(cur.lower())
            if cur_key != need: return None
        else:
            # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: сравниваем с ВСЕМИ леммами токена ---
            # это ловит «Старка/Старком» → лемма «старк», «Вижна/Вижном» → «вижн»
            lemmas = possible_lemmas(cur)
            if need not in lemmas:
                return None

        matched.append(j); j += 1
        
    k = j; tail = ""
    save_k = k
    while k < len(tokens) and is_space(tokens[k]): k += 1
    if k < len(tokens) and is_word(tokens[k]) and re.fullmatch(r'(?:\d+|[IVX]{1,4})', tokens[k]):
        tail = "".join(tokens[save_k:k+1]); k += 1
    return matched, j, k, tail

def apply_replacement(tokens, pos_list, entry, tail):
    dst_tokens = entry["dst_tokens"]
    out_words = []
    for idx, dst_base in enumerate(dst_tokens):
        src_idx = pos_list[min(idx, len(pos_list)-1)]
        out_words.append(inflect_like(tokens[src_idx], dst_base))
    seps = []
    for a,b in zip(pos_list, pos_list[1:]): seps.append("".join(tokens[a+1:b]))
    rebuilt = []
    for i, w in enumerate(out_words):
        rebuilt.append(w)
        if i < len(seps): rebuilt.append(seps[i])
    if entry["sequel_src"]: rebuilt.append(entry["sequel_src"])
    if tail: rebuilt.append(tail)
    first, last = pos_list[0], pos_list[-1]
    tokens[first] = "".join(rebuilt)
    for t in range(first+1, last+1): tokens[t] = ""

def replace_text(text: str, mapping: dict) -> tuple[str, int]:
    entries = prepare_entries(mapping)
    tokens = tokenize(text); i = 0
    replacements = 0
    while i < len(tokens):
        if not is_word(tokens[i]): i += 1; continue
        matched = None
        for e in entries:
            m = match_phrase(tokens, i, e)
            if m: matched = (e, *m); break
        if not matched:
            i += 1; continue
        entry, positions, j, k, tail = matched
        apply_replacement(tokens, positions, entry, tail)
        replacements += 1
        i = k
    return "".join(tokens), replacements

def main():
    ap = argparse.ArgumentParser(description="Пакетная замена по готовому словарю (RU-склонения, EN — без склонения).")
    ap.add_argument("--in-dir", required=True, help="Папка с исходными .txt")
    ap.add_argument("--mapping", required=True, help="Готовый terms_map.json (ключ→значение)")
    ap.add_argument("--out-dir", required=True, help="Куда сохранить заменённые файлы")
    ap.add_argument("--no-aliases", action="store_true", help="Не добавлять авто-алиасы Имя/Фамилия")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping(Path(args.mapping))
    if not mapping:
        print("[ERROR] В словаре нет ни одной пары с непустым значением.", file=sys.stderr); sys.exit(1)

    if not args.no_aliases:
        mapping = expand_person_aliases(mapping, enable=True)

    files = sorted(in_dir.glob("*.txt"))
    if not files:
        print(f"[WARN] В {in_dir} нет .txt файлов", file=sys.stderr)

    total_repl = 0
    print(f"[INFO] пар в словаре: {len(mapping)}; файлов: {len(files)}")
    for idx, p in enumerate(files, 1):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        res, cnt = replace_text(txt, mapping)
        (out_dir / p.name).write_text(res, encoding="utf-8")
        total_repl += cnt
        print(f"[OK] [{idx}/{len(files)}] → {out_dir/p.name} (замен: {cnt})")

    print(f"[DONE] Всего замен: {total_repl}")

if __name__ == "__main__":
    main()

