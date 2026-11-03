r"""Convert a TEI edition (Corpus Corporum / Mirabile, e.g. the SISMEL Mussato editions) to a .txt
that ingest.py reads as format A (pages): one "## p. N (#k) ####" marker per <pb n="N"/>, so chunks
are cited by the printed page of the edition.

    .venv\Scripts\python.exe tei_to_txt.py Traditio_civitate_Paduae.xml [more.xml ...] [--out-dir DIR]

Writes <name>.txt next to each XML (or in --out-dir) and prints a report of every change.
The first line is "%%NO-OCR", so ingest.py does not flag the chunks as OCR.

Layout:
  - teiHeader and <front> are skipped; only <body> is converted.
  - Sections become inline markers at the start of a paragraph: "[§ 40]" when the <div2>s carry n=
    (Traditio), "[lib. IV, § 31]" / "[prol., § 1]" when <div2>s are books and <milestone n=> are the
    sections (De gestis). A <div2>'s <head> that is only its number is dropped (the marker replaces it).
  - <lb/> is a space, <hi> is joined without a space ("XIV<hi>or</hi>" -> "XIVor"), <l> lines are joined
    with " / ".
  - A page whose next <pb> skips numbers gets a range label ("165-167"): its text runs over those pages.

Cleanup (this edition only, never ingest.py's general text cleanup), each change listed in the report:
  - footnote calls glued to a word ("veterum8" -> "veterum"), and the free-standing ones in KNOWN_NOTE_CALLS;
    other free-standing numbers are kept (De gestis VII 54 has real ones) and only listed for review;
  - a space where removed markup joined two words ("adiciebaturTartarus", "cives,abbatem").
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "{http://www.tei-c.org/ns/1.0}"
NO_OCR = "%%NO-OCR"                 # must match ingest.TEXT_NO_OCR
PB = ""                       # page-break placeholder inside paragraph text
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV"]

# Free-standing footnote calls: (text before, call, text after), as the text reads after whitespace is collapsed.
KNOWN_NOTE_CALLS = [
    ("lacrimatus est.", "98", "Sicque"),        # Traditio 67
    ("egisset,", "269", "tamen"),               # Traditio 256
    ("Post quam,", "339", "conclamata"),        # Traditio 323
]
GLUED_CALL = re.compile(r"([^\W\d_])(\d+)(?!\w)")
CAMEL_JOIN = re.compile(r"([a-zß-ÿ])([A-ZÀ-Þ])")
PUNCT_JOIN = re.compile(r"([,;:.!?])([^\W\d_])")
FREE_NUMBER = re.compile(r"(?<!\w)\d+(?!\w)")


def local(tag) -> str:
    return tag.replace(NS, "") if isinstance(tag, str) else ""


def nonws(s: str) -> int:
    return len(re.sub(r"\s", "", s))


def context(s: str, start: int, end: int, width: int = 25) -> str:
    return (s[max(0, start - width):start] + "⟦" + s[start:end] + "⟧" + s[end:end + width]).replace(PB, "¦")


class Emitter:
    """Paragraphs of text with PB placeholders; the page labels go in self.pages, in order."""

    def __init__(self) -> None:
        self.paras: list[str] = []
        self.cur: list[str] = []
        self.pages: list[str] = []
        self.marker_chars = 0           # non-space chars added by section markers and " / "

    def text(self, s: str) -> None:
        self.cur.append(s)

    def para(self) -> None:
        p = "".join(self.cur)
        if p.strip():
            self.paras.append(p)
        self.cur = []

    def page(self, n: str, tail: str) -> None:
        """A <pb> between two letters splits a word: mark it glued, to be moved after the word."""
        self.pages.append(n)
        before = "".join(self.cur)[-1:]
        self.cur.append(PB if before.isalpha() and tail[:1].isalpha() else f" {PB} ")

    def section(self, label: str) -> None:
        self.para()
        marker = f"[{label}] "
        self.marker_chars += nonws(marker)
        self.text(marker)


def convert(xml: Path, out: Path) -> None:
    root = ET.parse(xml).getroot()
    body = root.find(f".//{NS}body")
    if body is None:
        sys.exit(f"{xml.name}: no TEI <body>")
    div2s = list(body.iter(f"{NS}div2"))
    numbered = bool(div2s) and all(d.get("n") for d in div2s)   # Traditio: one div2 per section
    book_of = {} if numbered else {d: i for i, d in enumerate(div2s)}
    em = Emitter()
    skipped_heads = []
    state = {"book": None}

    def walk(el) -> None:
        tag = local(el.tag)
        skip_text = False
        if tag == "pb":
            em.page(el.get("n") or "", el.tail or "")
        elif tag == "lb":
            em.text(" ")
        elif tag == "milestone":
            b = state["book"]
            em.section(f"prol., § {el.get('n')}" if b == 0 else f"lib. {ROMAN[b]}, § {el.get('n')}")
        elif tag == "div2":
            if numbered:
                em.section(f"§ {el.get('n')}")
            else:
                em.para()
                state["book"] = book_of[el]
        elif tag == "head":
            parent_n = numbered and el in heads_of_div2
            if parent_n and (el.text or "").strip() == heads_of_div2[el]:
                skipped_heads.append(el)
                skip_text = True
            else:
                em.para()
        elif tag == "p":
            em.para()
        if el.text and not skip_text:
            em.text(el.text)
        prev = None
        for ch in el:
            if local(ch.tag) == "l" and prev == "l":
                em.text(" / ")
                em.marker_chars += 1
            walk(ch)
            if ch.tail:
                em.text(ch.tail)
            if local(ch.tag) != "lb":
                prev = local(ch.tag)
        if tag in ("p", "div2") or (tag == "head" and not skip_text):
            em.para()

    heads_of_div2 = {h: d.get("n") for d in div2s for h in d.findall(f"{NS}head")}
    walk(body)
    em.para()
    source_chars = nonws("".join(body.itertext())) - sum(nonws(h.text or "") for h in skipped_heads)

    paras = [" ".join(p.split()) for p in em.paras]
    report: dict[str, list[str]] = {k: [] for k in (
        "glued note calls removed", "known note calls removed", "known note calls NOT found",
        "spaces inserted (lowercase+Capital)", "spaces inserted (after punctuation)",
        "page breaks moved to the end of a split word", "free-standing numbers kept (review)")}
    removed = 0

    for before, call, after in KNOWN_NOTE_CALLS:
        pat = f"{before} {call} {after}"
        hits = sum(p.count(pat) for p in paras)
        if hits == 1:
            paras = [p.replace(pat, f"{before} {after}") for p in paras]
            removed += len(call)
            report["known note calls removed"].append(pat)
        elif hits > 1:
            sys.exit(f"{xml.name}: known note call {pat!r} occurs {hits} times")
        elif any(before in p and after in p for p in paras):
            report["known note calls NOT found"].append(pat)

    def sub(pattern, repl, key, p):
        def f(m):
            report[key].append(context(p, m.start(), m.end()))
            return repl(m)
        return pattern.sub(f, p)

    out_paras = []
    for p in paras:
        n_before = nonws(p)
        p = sub(GLUED_CALL, lambda m: m.group(1), "glued note calls removed", p)
        removed += n_before - nonws(p)
        p = sub(CAMEL_JOIN, lambda m: f"{m.group(1)} {m.group(2)}", "spaces inserted (lowercase+Capital)", p)
        p = sub(PUNCT_JOIN, lambda m: f"{m.group(1)} {m.group(2)}", "spaces inserted (after punctuation)", p)
        p = sub(re.compile(rf"([^\W\d_]){PB}(\S+)"), lambda m: f"{m.group(1)}{m.group(2)} {PB}",
                "page breaks moved to the end of a split word", p)   # <pb> inside a word
        for m in FREE_NUMBER.finditer(p):
            if re.search(r"\[[^\]]*$", p[:m.start()]) and "]" in p[m.end():m.end() + 2]:
                continue                                    # inside a "[§ 40]" marker
            report["free-standing numbers kept (review)"].append(context(p, m.start(), m.end()))
        out_paras.append(p)

    # Page labels: a page whose next <pb> skips numbers covers the whole range.
    labels, gaps = [], []
    for i, n in enumerate(em.pages):
        nxt = em.pages[i + 1] if i + 1 < len(em.pages) else None
        if n.isdigit() and nxt and nxt.isdigit() and int(nxt) > int(n) + 1:
            labels.append(f"{n}–{int(nxt) - 1}")
            gaps.append(f"p. {n} -> next marker p. {nxt}: labelled {labels[-1]}")
        else:
            labels.append(n)
        if nxt and n.isdigit() and nxt.isdigit() and int(nxt) <= int(n):
            gaps.append(f"p. {n} -> next marker p. {nxt}: NOT increasing (check)")

    # Paragraphs before the first <pb> (De gestis: the Corpus Corporum title) go on the first page.
    lead = 0
    while lead < len(out_paras) and PB not in out_paras[lead]:
        lead += 1
    if 0 < lead < len(out_paras):
        report["moved onto the first page (were before its <pb>)"] = out_paras[:lead]
        first_pb = out_paras[lead].replace(PB, "", 1)
        out_paras = [PB] + out_paras[:lead] + [first_pb] + out_paras[lead + 1:]
    lines = [NO_OCR]
    k = 0
    for p in out_paras:
        segs = p.split(PB)
        lines.append(segs[0].strip())
        for seg in segs[1:]:
            k += 1
            lines.append(f"## p. {labels[k - 1]} (#{k}) ####")
            lines.append(seg.strip())
        lines.append("")
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"
    first = next(l for l in text.split("\n")[1:] if l.strip())
    if not first.startswith("## p. "):
        sys.exit(f"{xml.name}: text before the first page break: {first[:60]!r}")
    out.write_text(text, encoding="utf-8", newline="\n")

    marks = re.findall(r"^## p\. .*$", text, re.M)
    body_chars = nonws(text) - nonws(NO_OCR) - sum(nonws(m) for m in marks)
    expected = source_chars - removed + em.marker_chars
    print(f"\n=== {xml.name} -> {out.name}")
    print(f"structure: {'one <div2 n> per section' if numbered else f'{len(div2s)} books with <milestone> sections'}; "
          f"{len(out_paras)} paragraphs; {len(em.pages)} page markers (p. {em.pages[0]}-{em.pages[-1]})")
    print(f"coverage: source {source_chars} chars - {removed} note-call digits + {em.marker_chars} marker chars "
          f"= {expected}; written {body_chars} {'OK' if expected == body_chars else 'MISMATCH'}")
    print(f"page gaps: {gaps or 'none'}")
    for key, rows in report.items():
        print(f"{key}: {len(rows)}")
        for r in rows:
            print(f"    {r}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("xml", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path)
    opts = ap.parse_args()
    for xml in opts.xml:
        out = (opts.out_dir / xml.name if opts.out_dir else xml).with_suffix(".txt")
        convert(xml, out)


if __name__ == "__main__":
    main()
