# -*- coding: utf-8 -*-
import argparse, json, os, re, sys, random, warnings
from pathlib import Path
from collections import Counter, defaultdict
from difflib import SequenceMatcher

# ---------------- Morph init: prefer pymorphy3 (fast, Py3.12-friendly) ----------------
morph = None
USING_MORPH3 = False
try:
    from pymorphy3 import MorphAnalyzer as MorphAnalyzer3
    morph = MorphAnalyzer3()
    USING_MORPH3 = True
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
# -------------------------------------------------------------------------------------

random.seed(42)

SEQUEL_SUFFIX_RE = r'(?:\s*[:\-]?\s*(?:\d+|[IVX]{1,4}))$'
TOKEN_RE = re.compile(r'([A-Za-zА-Яа-яЁё0-9]+|[\-–—]|[^\w\s]|[\s]+)', re.U)
def tokenize(text: str): return TOKEN_RE.findall(text)
def is_word(tok: str) -> bool: return bool(re.match(r'^[A-Za-zА-Яа-яЁё0-9]+$', tok))
def is_space(tok: str) -> bool: return tok.isspace()
def is_hyphen(tok: str) -> bool: return tok in ('-','–','—')

def lemmatize(word: str) -> str:
    return morph.parse(word)[0].normal_form

def similarity(a, b): return SequenceMatcher(None, a, b).ratio()

def load_seed_mapping(path_str: str):
    if not path_str:
        return {}
    p = Path(path_str)
    if not p.exists() or not p.is_file():
        print(f"[WARN] seed-mapping не найден: {p}. Продолжаю без него.", file=sys.stderr)
        return {}
    raw = p.read_text(encoding="utf-8-sig")
    if not raw.strip():
        print(f"[WARN] seed-mapping пустой: {p}.", file=sys.stderr)
        return {}
    cleaned = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    cleaned = re.sub(r"(?m)^\s*//.*?$", "", cleaned)
    return json.loads(cleaned)

def base_key(s: str) -> str:
    s2 = s.strip().lower()
    s2 = re.sub(r'[\s\-–—]+', ' ', s2)
    s2 = re.sub(SEQUEL_SUFFIX_RE, '', s2)
    return s2

# --- генерация выдуманных названий (нейтральный стиль) ---
SYL_FIRST = ["Ли","За","Ка","Но","Ри","Та","Се","Эй","Де","Ар","Ви","Ма","Ко","Эл","Исо","Те","Ха","На","Ла","Ша"]
SYL_MID   = ["ни","ро","ва","кси","лор","ви","тра","мен","нек","вар","мор","кал","сен","вер","дар","ли","нон","хал","сал"]
SYL_LAST  = ["н","с","р","ль","в","кс","рр","м","рд","т","нн","сс","рт","нд","льд"]
SUF_PERSON_LAST = ["ов","ев","ин","ик","нер","вик","кроу","драк","восс","кальд","лорн","вирт","мидсон","стерн"]
SUF_PLACE  = ["ия","ана","ория","нвель","бург","хейм","стан","град","поль","вилль"]

SUF_ORG    = [" Альянса"," Орден"," Синдикат"," Корпорация"," Консорциум"," Легион"]
SUF_TECH   = [" Кристалл"," Сфера"," Жезл"," Рукавица"," Барьер"," Модуль"]
SUF_EVENT  = [" Война"," Сражение"," Осада"," Инцидент"," Противостояние"]

def gen_person():
    first = random.choice(SYL_FIRST)+random.choice(SYL_MID)
    last  = random.choice(SYL_FIRST)+random.choice(SYL_MID)+random.choice(SUF_PERSON_LAST)
    return f"{first} {last}"

def gen_place():
    base = random.choice(SYL_FIRST)+random.choice(SYL_MID)
    return base + random.choice(SUF_PLACE)

def gen_org():
    base = random.choice(SYL_FIRST)+random.choice(SYL_MID)+random.choice(SYL_LAST)
    return base + random.choice(SUF_ORG)

def gen_tech():
    base = random.choice(SYL_FIRST)+random.choice(SYL_MID)+random.choice(SYL_LAST)
    return base + random.choice(SUF_TECH)

def gen_event():
    base = random.choice(SYL_FIRST)+random.choice(SYL_MID)+random.choice(SYL_LAST)
    return base + random.choice(SUF_EVENT)

def generate_fake(name: str, kind: str) -> str:
    return {
        "person": gen_person,
        "place":  gen_place,
        "org":    gen_org,
        "tech":   gen_tech,
        "event":  gen_event,
    }.get(kind, gen_place)()

# --- эвристическое определение типа по контексту ---
CTX_PERSON = [r"\b(мистер|миссис|доктор|агент|капитан|лор[дт])\b"]
CTX_PLACE  = [r"\b(город|страна|столиц[аы]|тюрьм[ае]|баз[ае]|земл[яеи])\b"]
CTX_ORG    = [r"\b(организаци|корпораци|агентств|институт|департамент|орден|легион)\w*\b"]
CTX_TECH   = [r"\b(камень|щит|жезл|перчатка|клинок|доспех|устройств|дрон|молот)\w*\b"]
CTX_EVENT  = [r"\b(битв|войн|вторжен|противостояни|инцидент|осад)\w*\b"]

def guess_type(name: str, text_sample: str) -> str:
    ctx = text_sample.lower()
    for rx in CTX_PERSON:
        if re.search(rx, ctx): return "person"
    for rx in CTX_PLACE:
        if re.search(rx, ctx):  return "place"
    for rx in CTX_ORG:
        if re.search(rx, ctx):  return "org"
    for rx in CTX_TECH:
        if re.search(rx, ctx):  return "tech"
    for rx in CTX_EVENT:
        if re.search(rx, ctx):  return "event"
    if len(name.split())==2 and name.split()[0][0].isupper() and name.split()[1][0].isupper():
        return "person"
    if any(t in name.lower() for t in ["битва","война","вторжение","инцидент","осада"]):
        return "event"
    if any(t in name.lower() for t in ["институт","корпорация","агентство","орден","легион","тюрьма"]):
        return "org"
    if any(t in name.lower() for t in ["камень","перчатка","жезл","щит","молот"]):
        return "tech"
    return "place"

# --- извлечение кандидатов из текста ---
MARVEL_HINTS = [
    r"\bЩИТ\b", r"\bГидра\b", r"\bМ\.Е\.Ч\.?\b", r"\bS\.W\.O\.R\.D\.?\b",
    r"\bКамн(ь|и)\b", r"\bКамень (разума|времени|пространства|реальности|силы|души)\b",
    r"\bВибраниум\b", r"\bКвинджет\b",
    r"\bМстител[ьи]\b", r"\bВойна Бесконечности\b", r"\bГражданская война\b", r"\bБитва\b",
    r"\bAvengers\b", r"\bSokovia\b", r"\bWakanda\b", r"\bAsgard\b", r"\bInfinity\b",
]

def extract_candidates(text: str):
    cands = Counter()
    for rx in MARVEL_HINTS:
        for m in re.finditer(rx, text, flags=re.I | re.U):
            cands[m.group(0)] += 1
    for m in re.finditer(r'\b([A-ZА-ЯЁ][a-zа-яё]+(?:[- ][A-ZА-ЯЁa-zа-яё]+)+)\b', text, flags=re.U):
        s = m.group(1).strip()
        cands[s] += 1
    for m in re.finditer(r'\b(Мстители|ЩИТ|Гидра|Рафт|Асгард|Ваканда|Соковия|Камень|Камни|Вибраниум)\b', text, flags=re.U|re.I):
        cands[m.group(1)] += 1
    return [k for k, _ in cands.items() if len(k) >= 2]

def generate_aliases(mapping: dict):
    first_count, last_count = Counter(), Counter()
    pairs = []
    for k, v in mapping.items():
        ks, vs = k.split(), v.split()
        if len(ks)==2 and len(vs)==2 and ks[0][0].isupper() and ks[1][0].isupper():
            pairs.append((ks,vs))
            first_count[ks[0]] += 1
            last_count[ks[1]]  += 1
    new = {}
    for (ks,vs) in pairs:
        f,l = ks; df,dl = vs
        if first_count[f]==1 and f not in mapping:
            new[f] = df
        if last_count[l]==1 and l not in mapping:
            new[l] = dl
    return new

def build_mapping_for_folder(in_dir: Path, seed: dict, fuzzy_threshold: float):
    mapping = dict(seed) if seed else {}
    seen_new = {}

    for p in sorted(in_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        cands = extract_candidates(text)
        for cand in cands:
            bk = base_key(cand)
            found_key = None
            for k in mapping.keys():
                if base_key(k) == bk: found_key = k; break
            if not found_key:
                sim_hits = [k for k in mapping.keys() if similarity(base_key(k), bk) >= fuzzy_threshold]
                if sim_hits:
                    found_key = sorted(sim_hits, key=lambda k: -similarity(base_key(k), bk))[0]
            if found_key:
                continue
            m = re.search(re.escape(cand), text)
            ctx = text[max(0, m.start()-80): m.end()+80] if m else ""
            kind = guess_type(cand, ctx)
            fake = generate_fake(cand, kind)
            sequel = re.search(SEQUEL_SUFFIX_RE, cand)
            if sequel:
                base = re.sub(SEQUEL_SUFFIX_RE, '', cand).rstrip()
                bk2 = base_key(base)
                base_existing = None
                for k in mapping.keys():
                    if base_key(k)==bk2:
                        base_existing = k; break
                if base_existing:
                    fake = mapping[base_existing] + sequel.group(0)
            mapping[cand] = fake
            seen_new[cand] = fake

    aliases = generate_aliases(mapping)
    for k,v in aliases.items():
        mapping.setdefault(k, v)

    groups = defaultdict(set)
    for k,v in mapping.items():
        groups[base_key(k)].add(v)
    for bk, vs in groups.items():
        if len(vs) > 1:
            print(f"[WARN] Базовая форма {bk!r} мапится в разные значения: {sorted(vs)}", file=sys.stderr)

    return mapping, seen_new

# ---- Замена текста (тот же движок, что и в replace_names.py) ----
def prepare_entries(mapping: dict):
    entries = []
    for src, dst in mapping.items():
        src_tokens = re.split(r'[ \-]+', src.strip())
        dst_tokens = re.split(r'[ \-]+', dst.strip())
        src_lemmas = [lemmatize(t.lower()) for t in src_tokens]
        entries.append({"src":src,"dst":dst,
                        "src_tokens":src_tokens,"dst_tokens":dst_tokens,
                        "src_lemmas":src_lemmas})
    entries.sort(key=lambda e: len(e["src_tokens"]), reverse=True)
    return entries

def transfer_caps(src_word: str, dst_word: str) -> str:
    if src_word.isupper(): return dst_word.upper()
    if src_word.istitle(): return dst_word[:1].upper() + dst_word[1:]
    return dst_word

def heuristic_inflect(dst_base: str, src_word: str) -> str:
    low = dst_base.lower()
    p = morph.parse(src_word)[0]
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

def inflect_like(source: str, target_base: str) -> str:
    p_src = morph.parse(source)[0]
    p_tgt = morph.parse(target_base)[0]
    need = set()
    if getattr(p_src.tag, "case", None):   need.add(p_src.tag.case)
    if getattr(p_src.tag, "number", None): need.add(p_src.tag.number)
    if getattr(p_tgt.tag, "gender", None) and getattr(p_src.tag, "gender", None):
        need.add(p_src.tag.gender)
    inflected = None
    try:
        inflected = p_tgt.inflect(need) if need else None
    except Exception:
        inflected = None
    form = inflected.word if inflected else heuristic_inflect(target_base, source)
    return transfer_caps(source, form)

def match_phrase(tokens, i, entry):
    lemmas = entry["src_lemmas"]; j = i; matched = []
    for need in lemmas:
        while j < len(tokens) and (is_space(tokens[j]) or is_hyphen(tokens[j])): j += 1
        if j >= len(tokens) or not is_word(tokens[j]): return None
        if lemmatize(tokens[j].lower()) != need: return None
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
    for a,b in zip(pos_list, pos_list[1:]):
        seps.append("".join(tokens[a+1:b]))
    rebuilt = []
    for i, w in enumerate(out_words):
        rebuilt.append(w)
        if i < len(seps): rebuilt.append(seps[i])
    if tail: rebuilt.append(tail)
    first, last = pos_list[0], pos_list[-1]
    tokens[first] = "".join(rebuilt)
    for t in range(first+1, last+1):
        tokens[t] = ""

def replace_text(text: str, mapping: dict) -> str:
    entries = prepare_entries(mapping)
    tokens = tokenize(text); i = 0
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
        i = k
    return "".join(tokens)

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Сбор словаря по папке и массовая замена (Marvel → выдуманное).")
    ap.add_argument("--in-dir", required=True, help="Папка с .txt файлами")
    ap.add_argument("--seed-mapping", default=None, help="Начальный JSON-словарь (опционально)")
    ap.add_argument("--out-mapping", required=True, help="Куда сохранить итоговый объединённый словарь")
    ap.add_argument("--out-diff", required=True, help="Куда сохранить diff новых пар")
    ap.add_argument("--replace-out-dir", default=None, help="Куда класть заменённые файлы (если указано)")
    ap.add_argument("--fuzzy-threshold", type=float, default=0.9, help="Порог похожести для объединения/проверок")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    seed = load_seed(args.seed_mapping) if args.seed_mapping else {}

    mapping, new_pairs = build_mapping_for_folder(in_dir, seed, args.fuzzy_threshold)

    Path(args.out_mapping).write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_diff).write_text(json.dumps(new_pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ mapping → {args.out_mapping}")
    print(f"✓ new pairs → {args.out_diff}")

    if args.replace_out_dir:
        out_dir = Path(args.replace_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(in_dir.glob("*.txt")):
            txt = p.read_text(encoding="utf-8", errors="ignore")
            replaced = replace_text(txt, mapping)
            (out_dir / p.name).write_text(replaced, encoding="utf-8")
        print(f"✓ replaced files → {out_dir}")

def load_seed(path_str: str):
    return load_seed_mapping(path_str)

if __name__ == "__main__":
    main()
