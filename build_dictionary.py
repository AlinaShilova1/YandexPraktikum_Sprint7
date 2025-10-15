# -*- coding: utf-8 -*-
import argparse, json, re, sys, csv, random, warnings
from pathlib import Path
from collections import Counter, defaultdict
from functools import lru_cache

# ---------------- Morph init: prefer pymorphy3 (fast), fallback to pymorphy2 ----------------
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
# ------------------------------------------------------------------------------------------

random.seed(42)

SEQUEL_SUFFIX_RE = r'(?:\s*[:\-]?\s*(?:\d+|[IVX]{1,4}))$'
LATIN_RE = re.compile(r'^[A-Za-z]+(?:[-\s][A-Za-z]+)*$')
DOT_ACRONYM_RE = re.compile(r'^[A-Za-zА-ЯЁ]\.(?:[A-Za-zА-ЯЁ]\.)+$')

def is_english_string(s: str) -> bool:
    return bool(LATIN_RE.fullmatch(s.strip()))

def is_dotted_acronym(tok: str) -> bool:
    return bool(DOT_ACRONYM_RE.fullmatch(tok))

@lru_cache(maxsize=200000)
def lemma(w: str) -> str:
    w = w.lower()
    try:
        return morph.parse(w)[0].normal_form
    except Exception:
        return w

# --- стоп-листы и разрешённые одиночные ---
RU_STOP = {
    "и","или","но","а","как","во","в","на","к","от","до","по","за","для",
    "что","чтобы","это","тот","эта","эти","его","ее","их","бы","ли","же",
    "был","была","были","будет","есть","нет","со","об","из","при","над","под",
    "он","она","они","оно","бросив","более","единственный","единственная","больше",
    "когда","после","вообще","затем","однажды","тогда","потом","далее","сначала","итак","таким","между","кроме",
}
EN_STOP = {
    "He","She","It","They","We","I","You","The","A","An","And","Or","But","If",
    "When","While","Because","After","Before","During","More","Most","Only",
    "As","At","By","From","In","On","Of","For","With","Not","Over","Into","Between",
    "Then","After","Later","Next","Meanwhile","However","Moreover","Thus","Therefore","Once"
}
COUNTRIES_RU = {
    "америка","сша","россия","беларусь","украина","венгрия","польша","германия","франция","италия",
    "испания","великобритания","англия","шотландия","уэльс","ирландия","норвегия","швеция","финляндия",
    "китай","япония","индия","бразилия","канада","австралия","египет","турция","израиль",
}
COUNTRIES_EN = {
    "usa","america","russia","belarus","ukraine","hungary","poland","germany","france","italy","spain",
    "united kingdom","england","scotland","wales","ireland","norway","sweden","finland","china","japan",
    "india","brazil","canada","australia","egypt","turkey","israel",
}
ALLOWED_SINGLE_RU = {"Мстители","ЩИТ","Гидра","Рафт","Асгард","Ваканда","Соковия","Вибраниум","Квинджет"}
ALLOWED_SINGLE_EN = {"Marvel","Avengers","S.H.I.E.L.D","SWORD","Hydra","Asgard","Wakanda","Sokovia"}

EVENT_PREPS = {"за","в","на","под","при"}
TRIGGER_HEADS = r"(?:Камень|Камни|Перчатка|Скипетр|ЩИТ|Гидра|Квинджет|Вибраниум|Башня|Институт|Рафт|М\.Е\.Ч\.|S\.W\.O\.R\.D\.)"
EVENT_HEADS   = r"(?:Битва|Война|Осада|Вторжение|Противостояние|Инцидент)"

CAP_SEQ     = re.compile(r"\b([A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+){0,3})\b", re.U)
HYPHEN_NAME = re.compile(r"\b([A-ZА-ЯЁ][a-zа-яё]+-[A-Za-zА-Яа-яЁё]{2,})\b", re.U)
ACRONYM     = re.compile(r"\b(ЩИТ|Гидра|М\.Е\.Ч\.?|S\.W\.O\.R\.D\.?)\b", re.U | re.I)
TECH_OBJ    = re.compile(rf"\b({TRIGGER_HEADS}\s+[a-zа-яё\-]{{2,}})\b", re.U | re.I)
EVENT_PAT   = re.compile(rf"\b({EVENT_HEADS})\s+({'|'.join(EVENT_PREPS)})\s+([A-ZА-ЯЁ][a-zа-яё]+)\b", re.U)

def _has_verb(token: str) -> bool:
    try:
        return str(morph.parse(token)[0].tag.POS) in ("VERB","INFN")
    except Exception:
        return False

def _bad_inside(name: str) -> bool:
    if re.search(r"[^\w\s\-–—\.]", name, flags=re.U): return True
    if len(name) > 40: return True
    if len(re.split(r"\s+", name.strip())) > 4: return True
    return False

def _is_proper_single_ru(tok: str) -> bool:
    if tok in ALLOWED_SINGLE_RU: return True
    if tok.lower() in COUNTRIES_RU: return False
    p = morph.parse(tok)[0]; tag = p.tag
    grams = getattr(tag, "grammemes", set())
    if any(g in grams or g in str(tag) for g in ("Name","Surn","Patr","Orgn","Abbr")):
        return True
    if "Geox" in grams or "Geox" in str(tag): return False
    return False

def _is_proper_single_en(tok: str) -> bool:
    if tok in ALLOWED_SINGLE_EN: return True
    if tok.lower() in COUNTRIES_EN: return False
    return False

def filter_candidate(name: str) -> bool:
    if _bad_inside(name): return False
    if ACRONYM.fullmatch(name): return True
    if EVENT_PAT.search(name):  return True

    tokens = [t for t in re.split(r"[\s\-–—]+", name) if t]
    if not tokens: return False
    if len(tokens) >= 2 and (tokens[0].lower() in RU_STOP or tokens[0] in EN_STOP):
        return False

    if len(tokens) == 1:
        t = tokens[0]
        if is_dotted_acronym(t):  # «М.Е.Ч.»/«S.W.O.R.D.» — ok
            return True
        if is_english_string(t):  return _is_proper_single_en(t)
        else:                     return _is_proper_single_ru(t)

    for t in tokens:
        if "." in t: continue
        if t.lower() in RU_STOP: return False
        if _has_verb(t):         return False
    return True

def split_base_suffix(s: str):
    m = re.search(SEQUEL_SUFFIX_RE, s.strip())
    if not m:
        return s.strip(), ""
    base = re.sub(SEQUEL_SUFFIX_RE, "", s.strip()).rstrip()
    return base, m.group(0)

def extract_candidates(text: str) -> Counter:
    cands = Counter()
    for m in ACRONYM.finditer(text):
        cands[m.group(1)] += 1
    for m in TECH_OBJ.finditer(text):
        cand = m.group(1).strip()
        if filter_candidate(cand): cands[cand] += 1
    for m in EVENT_PAT.finditer(text):
        cand = f"{m.group(1)} {m.group(2)} {m.group(3)}".strip()
        if filter_candidate(cand): cands[cand] += 1
    for m in CAP_SEQ.finditer(text):
        cand = m.group(1).strip()
        first = cand.split()[0]
        if first.lower() in RU_STOP or first in EN_STOP:  # «Тогда Танос»
            continue
        if filter_candidate(cand): cands[cand] += 1
    for m in HYPHEN_NAME.finditer(text):
        cand = m.group(1).strip()
        if filter_candidate(cand): cands[cand] += 1
    return cands

def lemma_key(s: str) -> str:
    base, _ = split_base_suffix(s)
    parts = re.split(r'[\s\-–—]+', base)
    out = []
    for t in parts:
        if not t: continue
        if is_dotted_acronym(t): out.append(t.lower())
        elif is_english_string(t): out.append(t.lower())
        else: out.append(lemma(t))
    return " ".join(out)

def canonical_src(s: str) -> str:
    parts = re.split(r'[\s\-–—]+', s.strip())
    out = []
    for t in parts:
        if not t: continue
        if is_dotted_acronym(t): out.append(t.upper())
        elif is_english_string(t): out.append(t.capitalize())
        else: out.append(lemma(t).capitalize())
    return " ".join(out)

# ---------- генерация вымышленных названий ----------
# RU
SYL_FIRST = ["Ли","За","Ка","Но","Ри","Та","Се","Эй","Де","Ар","Ви","Ма","Ко","Эл","Исо","Те","Ха","На","Ла","Ша"]
SYL_MID   = ["ни","ро","ва","кси","лор","ви","тра","мен","нек","вар","мор","кал","сен","вер","дар","ли","нон","хал","сал"]
SYL_LAST  = ["н","с","р","ль","в","кс","рр","м","рд","т","нн","сс","рт","нд","льд"]
SUF_PERSON_LAST = ["ов","ев","ин","ик","нер","вик","кроу","драк","восс","кальд","лорн","вирт","мидсон","стерн"]
SUF_PLACE  = ["ия","ана","ория","нвель","бург","хейм","стан","град","поль","вилль"]
SUF_ORG    = [" Альянса"," Орден"," Синдикат"," Корпорация"," Консорциум"," Легион"]
SUF_TECH   = [" Кристалл"," Сфера"," Жезл"," Рукавица"," Барьер"," Модуль"]
SUF_EVENT  = [" Война"," Сражение"," Осада"," Инцидент"," Противостояние"]

def gen_person_ru(): return f"{random.choice(SYL_FIRST)+random.choice(SYL_MID)} {random.choice(SYL_FIRST)+random.choice(SYL_MID)+random.choice(SUF_PERSON_LAST)}"
def gen_place_ru():  return random.choice(SYL_FIRST)+random.choice(SYL_MID)+random.choice(SUF_PLACE)
def gen_org_ru():    return random.choice(SYL_FIRST)+random.choice(SYL_MID)+random.choice(SYL_LAST)+random.choice(SUF_ORG)
def gen_tech_ru():   return random.choice(SYL_FIRST)+random.choice(SYL_MID)+random.choice(SYL_LAST)+random.choice(SUF_TECH)
def gen_event_ru():  return random.choice(SYL_FIRST)+random.choice(SYL_MID)+random.choice(SYL_LAST)+random.choice(SUF_EVENT)

# EN
EN_FIRST = ["Aiden","Evan","Noah","Mason","Liam","Logan","Caleb","Owen","Lucas","Jared","Rowan","Kade","Dylan","Kellan","Talon"]
EN_LAST  = ["Harwick","Voss","Kincaid","Merrin","Cross","Verran","Drake","Hollis","Varner","Kaldor","Wexler","Crowe","Holden","Vyrn"]
EN_PLACE = ["Eldoria","Valeron","Norvel","Haldengrad","Craneheim","Neoville","Ardell","Vestra","Ardenfall","Orviston"]
EN_ORG   = ["Aurex Corporation","Vector Guard","Chronos Order","Oris Syndicate","Cosmos Watch","Aurex Expo"]
EN_TECH  = ["Mind Crystal","Time Sphere","Chaos Codex","Aegis Module","Entropy Veil","Eon Gauntlet"]
EN_EVENT = ["Eon War","Alliance Rift","Siege of Westglen","Battle of Eridia","Shadow Front","Incident"]

def gen_person_en(): return f"{random.choice(EN_FIRST)} {random.choice(EN_LAST)}"
def gen_place_en():  return random.choice(EN_PLACE)
def gen_org_en():    return random.choice(EN_ORG)
def gen_tech_en():   return random.choice(EN_TECH)
def gen_event_en():  return random.choice(EN_EVENT)

def generate_fake(kind: str, english: bool) -> str:
    if english:
        return {
            "person": gen_person_en,
            "place":  gen_place_en,
            "org":    gen_org_en,
            "tech":   gen_tech_en,
            "event":  gen_event_en,
        }.get(kind, gen_place_en)()
    else:
        return {
            "person": gen_person_ru,
            "place":  gen_place_ru,
            "org":    gen_org_ru,
            "tech":   gen_tech_ru,
            "event":  gen_event_ru,
        }.get(kind, gen_place_ru)()

def guess_kind(sample: str) -> str:
    if EVENT_PAT.search(sample): return "event"
    if TECH_OBJ.search(sample):  return "tech"
    if ACRONYM.fullmatch(sample):return "org"
    if "-" in sample:            return "person"
    parts = sample.split()
    if len(parts) >= 2 and parts[0][0].isupper() and parts[1][0].isupper():
        return "person"
    return "org" if is_english_string(sample) else "place"

# ---------- seed mapping loader ----------
def load_seed_mapping(path: str) -> dict:
    if not path: return {}
    p = Path(path)
    if not p.exists(): return {}
    raw = p.read_text(encoding="utf-8-sig")
    cleaned = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    cleaned = re.sub(r"(?m)^\s*//.*?$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"[ERROR] Некорректный JSON: {path}", file=sys.stderr); raise
    # нормализуем ключи seed по лемме для наследования
    return {k.strip(): v.strip() for k, v in data.items() if k.strip()}

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Сбор словаря с автогенерацией вымышленных значений.")
    ap.add_argument("--in-dir", required=True, help="Папка с .txt")
    ap.add_argument("--out-json", default="terms_map.json", help="Выходной JSON со значениями")
    ap.add_argument("--out-csv", default="candidates_counts.csv", help="CSV (диагностика частот)")
    ap.add_argument("--seed-mapping", default=None, help="Опционально: готовый словарь для переиспользования имён")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    files = sorted(in_dir.glob("*.txt"))
    if not files:
        print(f"[WARN] В папке {in_dir} нет .txt файлов", file=sys.stderr)

    seed = load_seed_mapping(args.seed_mapping)
    seed_by_lemma = {}
    for sk, sv in seed.items():
        base_sk, _ = split_base_suffix(sk)
        seed_by_lemma.setdefault(lemma_key(base_sk), sv)

    # агрегируем формы по (lemma_base, suffix)
    groups = defaultdict(lambda: {"forms_no_suffix": Counter(), "forms_by_suffix": defaultdict(Counter)})
    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        cands = extract_candidates(text)
        for cand, cnt in cands.items():
            base, suf = split_base_suffix(cand)
            lbk = lemma_key(base)
            if suf:
                groups[lbk]["forms_by_suffix"][suf][cand] += cnt
            else:
                groups[lbk]["forms_no_suffix"][cand] += cnt

    # собираем финальный словарь
    final_map = {}
    rows = []
    for lbk, data in sorted(groups.items(), key=lambda kv: sum(data["forms_no_suffix"].values()) + sum(sum(v.values()) for v in data["forms_by_suffix"].values()), reverse=True):
        base_forms = data["forms_no_suffix"]
        suffix_forms = data["forms_by_suffix"]

        # ключ-«база» (если нет форм без суффикса, возьмём самую частую из любых и отрежем суф.)
        if base_forms:
            best_base_form, _ = base_forms.most_common(1)[0]
        else:
            # найдём самый частотный среди всех и уберём суффикс визуально
            best_any_suf = max(((suf, cnts.most_common(1)[0]) for suf, cnts in suffix_forms.items()),
                               key=lambda x: x[1][1], default=(None, (None, 0)))
            best_form = best_any_suf[1][0] if best_any_suf[1][0] else None
            best_base_form = split_base_suffix(best_form)[0] if best_form else lbk

        base_key = canonical_src(best_base_form)
        english = all(is_english_string(t) or is_dotted_acronym(t) for t in re.split(r'[\s\-–—]+', base_key) if t)

        # определить тип по базе
        kind = guess_kind(best_base_form)

        # взять из seed, если есть; иначе сгенерировать
        fake_base = seed_by_lemma.get(lemma_key(base_key))
        if not fake_base:
            fake_base = generate_fake(kind, english)

        # записать базу
        final_map[base_key] = fake_base

        # записать все сиквелы (если есть)
        for suf, cnts in suffix_forms.items():
            best_form_suf, _ = cnts.most_common(1)[0]
            key_with_suf = canonical_src(best_form_suf)  # уже содержит суффикс
            final_map[key_with_suf] = fake_base + suf

        # статистика
        total_count = sum(base_forms.values()) + sum(sum(v.values()) for v in suffix_forms.values())
        raw_forms = []
        raw_forms += [f"{f}×{c}" for f,c in base_forms.most_common()]
        for suf, cnts in suffix_forms.items():
            raw_forms += [f"{f}×{c}" for f,c in cnts.most_common()]
        rows.append({
            "lemma_key": lbk,
            "chosen_base": base_key,
            "fake_base": fake_base,
            "total_count": total_count,
            "raw_forms": "; ".join(raw_forms)
        })

    # сохранить
    Path(args.out_json).write_text(json.dumps(final_map, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["lemma_key","chosen_base","fake_base","total_count","raw_forms"])
        wr.writeheader()
        wr.writerows(rows)

    print(f"✓ terms_map.json → {args.out_json} (пар: {len(final_map)})")
    print(f"✓ candidates_counts.csv → {args.out_csv}")

if __name__ == "__main__":
    main()
