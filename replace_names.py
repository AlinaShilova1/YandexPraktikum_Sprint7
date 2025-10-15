# -*- coding: utf-8 -*-
import argparse, json, os, re, sys, random, warnings, time
from pathlib import Path
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from functools import lru_cache

# ---------------- Morph init: prefer pymorphy3 (fast, Py3.12-friendly) ----------------
morph = None
USING_MORPH3 = False
t0 = time.time()
try:
    from pymorphy3 import MorphAnalyzer as MorphAnalyzer3
    morph = MorphAnalyzer3()
    USING_MORPH3 = True
    print(f"[DEBUG] Morph: pymorphy3 initialized in {time.time()-t0:.2f}s", file=sys.stderr)
except Exception as e:
    print(f"[WARN] pymorphy3 init failed: {e!r}. Falling back to pymorphy2…", file=sys.stderr)
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
    t1 = time.time()
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    print(f"[DEBUG] Morph: pymorphy2 initialized in {time.time()-t1:.2f}s", file=sys.stderr)
# -------------------------------------------------------------------------------------

random.seed(42)

SEQUEL_SUFFIX_RE = r'(?:\s*[:\-]?\s*(?:\d+|[IVX]{1,4}))$'
TOKEN_RE = re.compile(r'([A-Za-zА-Яа-яЁё0-9]+|[\-–—]|[^\w\s]|[\s]+)', re.U)
def tokenize(text: str): return TOKEN_RE.findall(text)
def is_word(tok: str) -> bool: return bool(re.match(r'^[A-Za-zА-Яа-яЁё0-9]+$', tok))
def is_space(tok: str) -> bool: return tok.isspace()
def is_hyphen(tok: str) -> bool: return tok in ('-','–','—')

# ---- лемматизация с кэшем (ускоряет ×2–×10) ----
@lru_cache(maxsize=200000)
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
        print(f"[WARN] seed-мэппинг пустой: {p}.", file=sys.stderr)
        return {}
    cleaned = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    cleaned = re.sub(r"(?m)^\s*//.*?$", "", cleaned)
    try:
        data = json.loads(cleaned)
        return data
    except json.JSONDecodeError as e:
        head = cleaned[:160].replace("\n", "↩")
        print(f"[ERROR] Некорректный JSON в {p}: {e}. Начало: {head}", file=sys.stderr)
        raise

def base_key(s: str) -> str:
    s2 = s.strip().lower()
    s2 = re.sub(r'[\s\-–—]+', ' ', s2)
    s2 = re.sub(SEQUEL_SUFFIX_RE, '', s2)
    return s2

# ---------------- генерация выдуманных названий (нейтральный стиль) ----------------
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

# ---------------- строгий извлекатель кандидатов ----------------
RU_STOP = {
    "и","или","но","а","как","во","в","на","к","от","до","по","за","для",
    "что","чтобы","это","тот","эта","эти","его","ее","их","бы","ли","же",
    "был","была","были","будет","есть","нет","со","об","из","при","над","под",
}
EVENT_PREPS = {"за","в","на","под","при"}  # допускаем в названиях событий

TRIGGER_HEADS = (
    r"(?:Камень|Камни|Перчатка|Скипетр|ЩИТ|Гидра|Квинджет|Вибраниум|Башня|Институт|Рафт|М\.Е\.Ч\.|S\.W\.O\.R\.D\.)"
)
EVENT_HEADS = r"(?:Битва|Война|Осада|Вторжение|Противостояние|Инцидент)"

# Прописные последовательности: 1–4 слова, каждое с заглавной
CAP_SEQ = re.compile(r"\b([A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+){0,3})\b", re.U)
# Дефисные имена: Человек-паук, Улисс-Кло (после дефиса допускаем нижний регистр)
HYPHEN_NAME = re.compile(r"\b([A-ZА-ЯЁ][a-zа-яё]+-[A-Za-zА-Яа-яЁё]{2,})\b", re.U)
# Акронимы/сокращения
ACRONYM = re.compile(r"\b(ЩИТ|Гидра|М\.Е\.Ч\.?|S\.W\.O\.R\.D\.?)\b", re.U | re.I)
# Технологии/объекты (голова с прописной + «хвост» со строчной)
TECH_OBJ = re.compile(rf"\b({TRIGGER_HEADS}\s+[a-zа-яё\-]{{2,}})\b", re.U | re.I)
# События: «Битва за Соковию», «Война в Ваканде»
EVENT_PAT = re.compile(
    rf"\b({EVENT_HEADS})\s+({'|'.join(EVENT_PREPS)})\s+([A-ZА-ЯЁ][a-zа-яё]+)\b", re.U
)

def _has_verb(token: str) -> bool:
    p = morph.parse(token)[0]
    pos = str(p.tag.POS)
    return pos in ("VERB", "INFN")

def _bad_inside(name: str) -> bool:
    if re.search(r"[^\w\s\-–—\.]", name, flags=re.U):
        return True
    if len(name) > 40: 
        return True
    if len(re.split(r"\s+", name.strip())) > 4:
        return True
    return False

def filter_candidate(name: str) -> bool:
    """Жёсткая фильтрация кандидата, чтобы не тащить предложения целиком."""
    if _bad_inside(name):
        return False
    # акронимы/сокращения — ок
    if ACRONYM.fullmatch(name):
        return True
    # события с предлогом — ок
    if EVENT_PAT.search(name):
        return True
    # обычные ФИО/названия: без глаголов и стоп-слов
    tokens = [t for t in re.split(r"[\s\-–—]+", name) if t]
    for t in tokens:
        if "." in t:   # пропускаем части типа «М.Е.Ч.»
            continue
        if t.lower() in RU_STOP:
            return False
        if _has_verb(t):
            return False
    return True

def extract_candidates(text: str):
    cands = set()

    # 1) акронимы/сокращения
    for m in ACRONYM.finditer(text):
        cands.add(m.group(1))

    # 2) тех/объекты от триггеров
    for m in TECH_OBJ.finditer(text):
        cand = m.group(1).strip()
        if filter_candidate(cand):
            cands.add(cand)

    # 3) события
    for m in EVENT_PAT.finditer(text):
        cand = f"{m.group(1)} {m.group(2)} {m.group(3)}".strip()
        if filter_candidate(cand):
            cands.add(cand)

    # 4) Capitalized-последовательности
    for m in CAP_SEQ.finditer(text):
        cand = m.group(1).strip()
        if filter_candidate(cand):
            cands.add(cand)

    # 5) дефисные имена
    for m in HYPHEN_NAME.finditer(text):
        cand = m.group(1).strip()
        if filter_candidate(cand):
            cands.add(cand)

    cands = [c for c in cands if len(c) >= 2]
    return sorted(cands, key=str.lower)

# ---------------- алиасы «Имя»/«Фамилия» (если уникальны) ----------------
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

# ---------------- построение словаря по папке ----------------
def build_mapping_for_folder(in_dir: Path, seed: dict, fuzzy_threshold: float):
    mapping = dict(seed) if seed else {}
    seen_new = {}

    files = sorted(in_dir.glob("*.txt"))
    print(f"[DEBUG] найдено файлов: {len(files)} в {in_dir}", file=sys.stderr)

    for idx, p in enumerate(files, 1):
        print(f"[DEBUG] [{idx}/{len(files)}] читаю: {p.name}", file=sys.stderr)
        text = p.read_text(encoding="utf-8", errors="ignore")
        print(f"[DEBUG] длина текста: {len(text):,} символов", file=sys.stderr)

        cands = extract_candidates(text)
        print(f"[DEBUG] кандидат(ов) извлечено: {len(cands)}", file=sys.stderr)
        if len(cands) <= 30:
            print(f"[DEBUG] кандидаты: {', '.join(cands)}", file=sys.stderr)

        for cand in cands:
            bk = base_key(cand)
            found_key = None
            for k in mapping.keys():
                if base_key(k) == bk: 
                    found_key = k; break
            if not found_key:
                sim_hits = [k for k in mapping.keys() if similarity(base_key(k), bk) >= fuzzy_threshold]
                if sim_hits:
                    found_key = sorted(sim_hits, key=lambda k: -similarity(base_key(k), bk))[0]
            if found_key:
                continue

            m = re.search(re.escape(cand), text)
            ctx = text[max(0, m.start()-80): m.end()+80] if m else ""
            # простая эвристика типа (можно дообучать, но мы сильно сузили кандидатов)
            kind = "person"
            if EVENT_PAT.search(cand): kind = "event"
            elif TECH_OBJ.search(cand): kind = "tech"
            elif ACRONYM.fullmatch(cand): kind = "org"
            elif "-" in cand: kind = "person"
            elif len(cand.split())>=2 and cand.split()[0][0].isupper() and cand.split()[1][0].isupper():
                kind = "person"
            else:
                kind = "place"

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

        print(f"[DEBUG] словарь на шаге: {len(mapping):,} пар (+{len(seen_new):,} новых)", file=sys.stderr)

    # алиасы (Имя/Фамилия), если уникальны
    aliases = generate_aliases(mapping)
    before = len(mapping)
    for k,v in aliases.items():
        mapping.setdefault(k, v)
    added_aliases = len(mapping) - before
    print(f"[DEBUG] добавлено алиасов: {added_aliases}", file=sys.stderr)

    # проверка согласованности
    groups = defaultdict(set)
    for k,v in mapping.items():
        groups[base_key(k)].add(v)
    inconsistent = {bk: vs for bk,vs in groups.items() if len(vs) > 1}
    if inconsistent:
        print(f"[WARN] найдены неоднозначные базы: {len(inconsistent)}", file=sys.stderr)
        for bk, vs in list(inconsistent.items())[:10]:
            print(f"   - {bk!r}: {sorted(vs)}", file=sys.stderr)

    return mapping, seen_new

# ---------------- замена текста ----------------
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

    print(f"[DEBUG] start; cwd={Path.cwd()}", file=sys.stderr)
    print(f"[DEBUG] args={vars(args)}", file=sys.stderr)

    in_dir = Path(args.in_dir)
    print(f"[DEBUG] in-dir resolved: {in_dir.resolve()}", file=sys.stderr)

    seed = load_seed_mapping(args.seed_mapping) if args.seed_mapping else {}
    print(f"[DEBUG] seed pairs: {len(seed)}", file=sys.stderr)

    print("[DEBUG] строю объединённый словарь…", file=sys.stderr)
    tA = time.time()
    mapping, new_pairs = build_mapping_for_folder(in_dir, seed, args.fuzzy_threshold)
    print(f"[DEBUG] словарь собран за {time.time()-tA:.2f}s: всего {len(mapping):,} пар; новых {len(new_pairs):,}", file=sys.stderr)

    print(f"[DEBUG] запись словарей → {args.out_mapping} ; diff → {args.out_diff}", file=sys.stderr)
    Path(args.out_mapping).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_mapping).write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_diff).write_text(json.dumps(new_pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.replace_out_dir:
        out_dir = Path(args.replace_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] начинаю замену во всех файлах → {out_dir}", file=sys.stderr)
        tR = time.time()
        files = sorted(in_dir.glob("*.txt"))
        for idx, p in enumerate(files, 1):
            txt = p.read_text(encoding="utf-8", errors="ignore")
            print(f"[DEBUG] [{idx}/{len(files)}] заменяю: {p.name} (len={len(txt):,})", file=sys.stderr)
            replaced = replace_text(txt, mapping)
            (out_dir / p.name).write_text(replaced, encoding="utf-8")
        print(f"[DEBUG] замена завершена за {time.time()-tR:.2f}s", file=sys.stderr)

    print("[OK] done.", file=sys.stderr)

if __name__ == "__main__":
    main()
