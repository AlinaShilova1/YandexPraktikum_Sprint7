# -*- coding: utf-8 -*-
import argparse, json, re, sys
from pathlib import Path

# --- Набор падежных окончаний, которые пытаемся ловить у исходных слов ---
ENDINGS = [
    'ами','ями','ого','его','ому','ему','ыми','ими','ою','ею',
    'ая','яя','ой','ей','ам','ям','ах','ях','ов','ев','ью',
    'ом','ем','у','ю','а','я','е','и','ы','о'
]

def detect_suffix(base: str, word_with_suffix: str) -> str:
    """Определяет суффикс (флексию), добавленный к base, учитывая е/ё."""
    if word_with_suffix.startswith(base):
        return word_with_suffix[len(base):]
    base_alt = base.replace("ё", "е").replace("Ё", "Е")
    word_alt = word_with_suffix.replace("ё", "е").replace("Ё", "Е")
    if word_alt.startswith(base_alt):
        return word_with_suffix[len(base):]
    return ""

def inflect_word(dst_base: str, suffix: str) -> str:
    """
    Эвристически «склоняет» выдуманное слово под суффикс исходного.
    Это не морфологический анализатор, но покрывает распространённые случаи.
    """
    if not suffix:
        return dst_base

    # аббревиатуры/капсом и короткие — не трогаем
    if dst_base.isupper() and len(dst_base) <= 6:
        return dst_base

    low = dst_base.lower()

    # суффикс небуквенный — просто приписываем (например, пунктуация)
    if all(not ch.isalpha() for ch in suffix):
        return dst_base + suffix

    # Fem на -а
    if low.endswith('а'):
        stem = dst_base[:-1]
        repl = {
            'ы':'ы','и':'и','е':'е','у':'у','ю':'ю','ой':'ой','ей':'ей'
        }
        for k,v in repl.items():
            if suffix.startswith(k):
                return stem + (v.upper() if dst_base[-1].isupper() else v)
        return stem + suffix

    # Fem на -я
    if low.endswith('я'):
        stem = dst_base[:-1]
        repl = {'и':'и','е':'е','ю':'ю','ей':'ей'}
        for k,v in repl.items():
            if suffix.startswith(k):
                return stem + (v.upper() if dst_base[-1].isupper() else v)
        return stem + suffix

    # Слова на -ия (Валерия → Валерии/Валерию/Валерией)
    if low.endswith('ия'):
        stem = dst_base[:-2]
        repl = {'и':'ии','е':'ии','ю':'ию','ей':'ией'}
        if suffix in repl:
            v = repl[suffix]
            return stem + (v.upper() if dst_base[-1].isupper() else v)
        return dst_base + suffix

    # Мягкий знак (в т.ч. муж.род): -ь → -я/-ю/-ем/-е/-и
    if low.endswith('ь'):
        stem = dst_base[:-1]
        if suffix in ('я','ю','ем','е','и'):
            return stem + (suffix.upper() if dst_base[-1].isupper() else suffix)
        return dst_base + suffix

    # По умолчанию просто приписываем суффикс
    return dst_base + suffix

def build_one_word_rule(src_word: str, dst_word: str):
    endings_re = '(?:' + '|'.join(sorted(set(map(re.escape, ENDINGS)), key=len, reverse=True)) + ')?'
    pat = re.compile(r'(?<!\w)(' + re.escape(src_word) + r')(' + endings_re + r')(?!\w)')
    def repl(m):
        full = m.group(1) + m.group(2)
        suf = detect_suffix(src_word, full)
        return inflect_word(dst_word, suf)
    return pat, repl

def build_multiword_rule(src_phrase: str, dst_phrase: str):
    """
    Универсальное правило для фраз из N слов.
    Берём суффикс каждого исходного токена и переносим его на соответствующий/последний токен замены.
    """
    src_tokens = src_phrase.split()
    dst_tokens = dst_phrase.split()
    endings_re = '(?:' + '|'.join(sorted(set(map(re.escape, ENDINGS)), key=len, reverse=True)) + ')?'
    parts = []
    for tok in src_tokens:
        parts.append('(' + re.escape(tok) + ')(' + endings_re + ')')
    pattern = r'(?<!\w)' + r'\s+'.join(parts) + r'(?!\w)'
    pat = re.compile(pattern)

    def repl(m):
        suffixes = []
        for i in range(len(src_tokens)):
            full = m.group(2*i+1) + m.group(2*i+2)
            suf = detect_suffix(src_tokens[i], full)
            suffixes.append(suf)
# пытаемся «распределить» суффиксы по токенам назначения;
        # если токенов замены меньше — используем суффикс последнего
        out_tokens = []
        for j, dtok in enumerate(dst_tokens):
            suf = suffixes[min(j, len(suffixes)-1)] if suffixes else ""
            out_tokens.append(inflect_word(dtok, suf))
        return " ".join(out_tokens)

    return pat, repl

def compile_rules(mapping: dict):
    # Сортируем по длине исходной фразы, чтобы длинные матчились первыми
    items = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
    rules = []
    for src, dst in items:
        if ' ' in src:
            pat, repl = build_multiword_rule(src, dst)
        else:
            pat, repl = build_one_word_rule(src, dst)
        rules.append((pat, repl))
    return rules

def apply_rules(text: str, rules):
    out = text
    for pat, repl in rules:
        out = pat.sub(repl, out)
    return out

def main():
    ap = argparse.ArgumentParser(description="Замена имён/терминов на выдуманные с учётом склонений (эвристика).")
    ap.add_argument("--input", required=True, help="Путь к исходному TXT")
    ap.add_argument("--mapping", required=True, help="JSON с парами {оригинал: выдуманное}")
    ap.add_argument("--output", required=True, help="Куда сохранить результат TXT")
    args = ap.parse_args()

    src_path = Path(args.input)
    map_path = Path(args.mapping)
    out_path = Path(args.output)

    text = src_path.read_text(encoding="utf-8")

    # Загружаем маппинг
    mapping = json.loads(map_path.read_text(encoding="utf-8"))

    # Компилим правила и применяем
    rules = compile_rules(mapping)
    result = apply_rules(text, rules)

    out_path.write_text(result, encoding="utf-8")
    print(f"✓ Готово: {out_path}")

if name == "main":
    main()