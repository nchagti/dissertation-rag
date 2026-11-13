r"""Convert a DraCor-style TEI drama (RomDraCor Seneca, the user's NeoLatDraCor Ecerinis) to a .txt that
ingest.py reads as format B (verse): chunks are cited by section and verse range, e.g.
"Act III, vv. 530-561", "Act I, Scene 2, vv. 93-119", "Scene 4, vv. 363-390".

    .venv\Scripts\python.exe drama_tei_to_txt.py seneca_xmls\*.xml [--out-dir DIR]

Writes <name>.txt (in --out-dir, default seneca_txt\) and prints a report of everything it had to decide.

Layout (format B, as the MQDQ files):
  - a header block, ended by a line of 60 "=";
  - "## Act I" / "## Scene 1" / "## Prelude" for a top-level division, "### Scene 2" for a scene inside an act.
    Acts are Roman, scenes Arabic. An act or scene without n= is numbered by its order in the file;
  - a speaker line (full name from particDesc, e.g. "Clytemestra", not the edition's "Clyt.") before each speech;
  - one "[n] text" line per verse, n from the edition's <milestone unit="line" n=>, suffix kept ("[392a]", "[5b]");
    a verse split between speakers (<l part="M|F">) repeats the number, as in MQDQ;
  - <stage> as "(text)"; <gap/> as "[...]"; <note> dropped (listed in the report); <pb>, castList,
    "card" milestones and <head> (the section heading replaces it) are not output.

Repairs, each listed in the report:
  - verses with no milestone (Oedipus 202-220) are numbered by interpolation, only when the count fits the gap
    exactly; otherwise the file fails;
  - a second milestone with the same number right after a split verse, in the same speech, continues that
    verse (Agamemnon 795: "Helena ubi est Troiam" + "puto.") and is joined onto it.
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "{http://www.tei-c.org/ns/1.0}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
HEADER_END = "=" * 60                 # must match ingest.TEXT_HEADER_END
VERSE = re.compile(r"^\[(\d+)[a-z0-9]*\] ")
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def local(e) -> str:
    return e.tag.replace(NS, "") if isinstance(e.tag, str) else ""


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def nonws(s: str) -> int:
    return len(re.sub(r"\s", "", s))


def num_part(n: str) -> int | None:
    m = re.match(r"\d+", n)
    return int(m.group()) if m else None


class Converter:
    def __init__(self, path: Path):
        self.path = path
        self.root = ET.parse(path).getroot()
        self.out: list[str] = []
        self.report: list[str] = []
        self.names = {p.get(XML_ID): norm("".join(p.find(f"{NS}persName").itertext()))
                      for p in self.root.iter(f"{NS}person") if p.find(f"{NS}persName") is not None}
        self.cur_n: str | None = None        # number of the current verse
        self.fresh = False                   # a line milestone came since the last <l>
        self.dup_of_part = False             # that milestone repeats the number of a split verse
        self.last_part: str | None = None    # part= of the last <l>
        self.last_sp = None
        self.pending: list[tuple[int, bool]] = []  # (index in self.out, starts a new verse) waiting for a number
        self.pending_after: str | None = None
        self.l_chars = 0                     # non-space chars of all verse text kept
        self.notes = 0

    # ---- header
    def header(self) -> list[str]:
        h = self.root.find(f"{NS}teiHeader")
        title_el = h.find(f".//{NS}titleStmt/{NS}title")
        title, abbr = norm(title_el.text), title_el.get("n")
        auth = h.find(f".//{NS}titleStmt/{NS}author/{NS}persName")
        author = norm(" ".join(auth.itertext())) if auth is not None else ""
        orig = h.find(f".//{NS}bibl[@type='originalSource']/{NS}title")
        digital = h.find(f".//{NS}bibl/{NS}idno[@type='URL']")
        lic = h.find(f".//{NS}publicationStmt//{NS}licence")
        lic_text = norm(" ".join(lic.itertext())) if lic is not None else ""
        pub = norm(h.findtext(f".//{NS}publicationStmt/{NS}publisher") or "")
        tid = self.root.get(XML_ID)
        lines = [f"{author}, {title}" + (f" ({abbr})" if abbr else ""),
                 f"Source: TEI-XML file {self.path.name}" + (f" ({pub}{', ' + tid if tid else ''})" if pub else ""),
                 f"Base text: {norm(' '.join(orig.itertext()))}" if orig is not None else "Base text: (not given)",
                 f"Digital source: {norm(digital.text)}" if digital is not None else "",
                 f"Rights: {lic_text}" if lic_text else "",
                 "Format: [n] = verse number of the base text (from the TEI line milestones; a letter suffix such as "
                 "[392a] is the edition's own). A verse divided between speakers appears as separate lines with the "
                 "same number. Lines without numbers are speaker names and (stage directions). "
                 "## / ### lines are acts and scenes.",
                 HEADER_END]
        return [l for l in lines if l]

    # ---- body
    def run(self) -> str:
        body = self.root.find(f".//{NS}body")
        self.top_count = {"act": 0, "scene": 0}
        self.walk_children(body, depth=0)
        self.flush_pending(None)
        return "\n".join(self.header() + self.out) + "\n"

    def walk_children(self, el, depth: int):
        counts = {}
        for c in el:
            t = local(c)
            if t == "div":
                kind = c.get("type") or ""
                counts[kind] = counts.get(kind, 0) + 1
                heading = self.div_heading(c, kind, counts[kind], depth)
                if heading:
                    self.out += ["", heading]
                self.walk_children(c, depth + (1 if heading else 0))
            elif t == "sp":
                self.speech(c)
            elif t == "milestone":
                self.milestone(c)
            elif t == "stage":
                self.stage(c, None)
            elif t == "speaker":                         # cast line outside a speech (Phaedra II)
                self.out.append(f"({norm(''.join(c.itertext()))})")
                self.report.append(f"<speaker> outside a speech output as a stage line: {norm(''.join(c.itertext()))!r}")
            elif t == "l":
                self.verse(c, None)
            elif t in ("castList", "head", "pb", "lb", ""):
                continue
            else:
                self.report.append(f"ignored <{t}> outside a speech")

    def div_heading(self, div, kind: str, ordinal: int, depth: int) -> str | None:
        if not div.findall(f".//{NS}l"):
            return None                                  # title page, cast list
        n = div.get("n")
        num = int(n) if n and n.isdigit() else ordinal
        if n is None and kind in ("act", "scene"):
            self.report.append(f"{kind} without n= numbered {num} by order")
        if kind == "act":
            return f"{'#' * (depth + 2)} Act {ROMAN[num]}"
        if kind == "scene":
            return f"{'#' * (depth + 2)} Scene {num}"
        if kind == "prelude":
            return f"{'#' * (depth + 2)} Prelude"
        self.report.append(f"div type={kind!r} given heading {kind.title()!r}")
        return f"{'#' * (depth + 2)} {kind.title() or 'Section'}"

    def speech(self, sp):
        who = (sp.get("who") or "").split()
        name = " / ".join(self.names.get(w.lstrip("#"), w.lstrip("#")) for w in who)
        if not name:
            name = norm("".join(sp.find(f"{NS}speaker").itertext())) if sp.find(f"{NS}speaker") is not None else ""
            self.report.append(f"speech without who=, labelled {name!r} from <speaker>")
        self.out.append(name)
        for c in sp:
            t = local(c)
            if t == "l":
                self.verse(c, sp)
            elif t == "milestone":
                self.milestone(c)
            elif t == "stage":
                self.stage(c, sp)
            elif t in ("speaker", "pb", "lb", ""):
                continue
            else:
                self.report.append(f"ignored <{t}> in a speech")

    def stage(self, st, sp):
        """A stage direction, or narrative verses encoded inside <stage> (Ecerinis 86-90): those stay verses."""
        if st.find(f"{NS}l") is None:
            self.out.append(f"({norm(''.join(st.itertext()))})")
            return
        self.report.append(f"verses inside <stage> kept as numbered verses (v. {st.find(f'{NS}milestone').get('n')}...)")
        for c in st:
            t = local(c)
            if t == "l":
                self.verse(c, sp)
            elif t == "milestone":
                self.milestone(c)
            elif t not in ("pb", "lb", "") and norm("".join(c.itertext())):
                sys.exit(f"{self.path.name}: <{t}> with text inside a verse <stage>")
        if norm(st.text or "") or any(norm(c.tail or "") for c in st):
            sys.exit(f"{self.path.name}: loose text inside a verse <stage>")

    def milestone(self, m):
        if m.get("unit") != "line":
            return
        n = norm(m.get("n"))
        if n != m.get("n"):
            self.report.append(f"line number {m.get('n')!r} trimmed to {n!r}")
        self.flush_pending(n)
        self.dup_of_part = n == self.cur_n and self.last_part is not None
        self.cur_n, self.fresh = n, True

    def flush_pending(self, next_n: str | None):
        if not self.pending:
            return
        a, b = num_part(self.pending_after or "0"), num_part(next_n) if next_n else None
        new = sum(starts for _, starts in self.pending)
        if b is None or b - a - 1 != new:
            sys.exit(f"{self.path.name}: {new} unnumbered verses after {self.pending_after} "
                     f"do not fit before {next_n}")
        k = a
        for idx, starts in self.pending:
            k += starts
            self.out[idx] = f"[{k}] {self.out[idx]}"
        self.report.append(f"vv. {a + 1}-{b - 1} had no line milestones: numbered by interpolation")
        self.pending = []

    def text_of(self, l) -> str:
        parts = [l.text or ""]
        for c in l:
            t = local(c)
            if t == "gap":
                parts.append(" [...] ")
            elif t == "note":
                self.notes += 1
                self.report.append(f"note dropped at v. {self.cur_n}: {norm(''.join(c.itertext()))!r}")
            elif t not in ("pb", "lb"):
                parts.append("".join(c.itertext()))
                self.report.append(f"<{t}> inside a verse kept as text")
            parts.append(c.tail or "")
        return norm("".join(parts))

    def verse(self, l, sp):
        text = self.text_of(l)
        self.l_chars += nonws(text)
        part = l.get("part")
        printed = l.get("n")
        if self.fresh and self.dup_of_part and part is None and sp is not None and sp is self.last_sp:
            prev = max(i for i, s in enumerate(self.out) if VERSE.match(s))
            self.out[prev] += " " + text
            self.report.append(f"v. {self.cur_n}: second milestone after a split verse; {text!r} joined to it")
            self.fresh = self.dup_of_part = False
            self.last_sp = sp
            return
        if self.fresh:
            self.out.append(f"[{self.cur_n}] {text}")
        elif part in ("M", "F") and not self.pending:
            self.out.append(f"[{self.cur_n}] {text}")
        else:                                        # no number yet: numbered when the next milestone comes
            if not self.pending:
                self.pending_after = self.cur_n
            self.pending.append((len(self.out), part not in ("M", "F")))
            self.out.append(text)
        if printed and self.fresh and norm(printed) != self.cur_n:
            self.report.append(f"<l n={printed!r}> under milestone {self.cur_n!r}")
        self.fresh = self.dup_of_part = False
        self.last_part, self.last_sp = part, sp


def check(conv: Converter, txt: str) -> list[str]:
    """Every verse's text is in the output; numbers in order apart from the edition's transpositions."""
    body = txt.split(HEADER_END + "\n", 1)[1].split("\n")
    verses = [l for l in body if VERSE.match(l)]
    got = sum(nonws(VERSE.sub("", l)) for l in verses)
    out = [f"verse lines: {len(verses)}, <l> elements: {sum(1 for _ in conv.root.iter(f'{NS}l'))}",
           f"verse chars: {got} in output vs {conv.l_chars} in <l> (minus notes) "
           f"{'OK' if got == conv.l_chars else 'MISMATCH'}"]
    nums = [num_part(VERSE.match(l).group(0)[1:]) for l in verses]
    back = [(a, b) for a, b in zip(nums, nums[1:]) if b < a]
    out.append(f"places where the number goes backwards (transpositions in the edition): {len(back)}"
               + (f", e.g. {back[:6]}" if back else ""))
    suffixed = sorted({VERSE.match(l).group(0)[1:-2] for l in verses if not re.match(r"^\[\d+\] ", l)},
                      key=lambda s: (num_part(s), s))
    if suffixed:
        out.append(f"suffixed numbers: {', '.join(suffixed)}")
    out.append(f"sections: {', '.join(l.lstrip('# ') for l in body if l.startswith('## ') or l.startswith('### '))}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("seneca_txt"))
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)
    for path in args.xml:
        conv = Converter(path)
        txt = conv.run()
        dest = args.out_dir / (path.name.removesuffix(".xml").removesuffix(".tei") + ".txt")
        dest.write_text(txt, encoding="utf-8", newline="\n")
        print(f"===== {path.name} -> {dest}")
        for line in check(conv, txt) + sorted(set(conv.report), key=conv.report.index):
            print("  " + line)


if __name__ == "__main__":
    main()
