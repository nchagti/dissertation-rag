"""Ingestion for the dissertation RAG: Zotero collection -> Chroma + SQLite FTS5 index.

Read-only with respect to Zotero: only GET requests to the local API, and every
file (PDF, snapshot, .txt) is opened for reading only.

Storage (all under <project>\\db):
    Chroma collection "zotero_chunks" (cosine), with bge-m3 embeddings
    keyword.sqlite: chunks, chunks_fts (FTS5), items, sources (fingerprints)

Usage:
    python ingest.py                     # index new/changed sources, remove deleted ones
    python ingest.py --rebuild           # start from scratch
    python ingest.py --items K1 K2 ...   # only these items (no deletion sweep)
    python ingest.py --items K1 --force  # re-index these items even if unchanged
    python ingest.py --dry-run           # extract + count chunks, write nothing
    python ingest.py --benchmark         # embedding speed on GPU vs CPU
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pymupdf
from bs4 import BeautifulSoup
from lingua import Language, LanguageDetectorBuilder
from tqdm import tqdm

from audit_zotero import PROJECT_DIR, ZoteroAPI, collect_collections, resolve_path, year_of

DB_DIR = PROJECT_DIR / "db"
KEYWORD_DB = DB_DIR / "keyword.sqlite"
LOG_FILE = DB_DIR / "ingest.log"
OCR_MANIFEST = PROJECT_DIR / "ocr_manifest.csv"
ADOBE_CSV = PROJECT_DIR / "embedded_annotations.csv"
COLLECTION_NAME = "zotero_chunks"

MODEL_NAME = "BAAI/bge-m3"
MAX_SEQ_LENGTH = 512
CHUNK_TOKENS = 450
OVERLAP_TOKENS = 60
MIN_CHUNK_CHARS = 25          # non-whitespace chars; pages with less (blank, page number only) get no chunk
FLUSH_CHUNKS = 256            # chunks embedded and committed together
CHUNKER_VERSION = "4"      # bump when extraction/chunking changes, to force re-embedding

LANGUAGES = [Language.ENGLISH, Language.FRENCH, Language.ITALIAN, Language.GERMAN, Language.LATIN]
PUBLICATION_FIELDS = ["publicationTitle", "bookTitle", "proceedingsTitle", "encyclopediaTitle",
                      "dictionaryTitle", "websiteTitle", "blogTitle", "forumTitle", "programTitle"]
PRINTED_PAGE_TYPES = {"journalArticle", "bookSection"}

# Printed-page labels to use instead of HathiTrust's, per .txt attachment: {attachment key: {scan: label}}.
# Hathi labels these scans "186" (after the plates in App. I); their running heads read 187 and 188.
# Overridden scans are exempt from the repeated-label skip. Scan 247 (a plate Hathi labels "187") stays
# skipped. After changing this dict, re-index the item with: ingest.py --items <item key> --force
HATHI_LABEL_OVERRIDES = {
    "DFL6RV27": {243: "187", 244: "188"},          # SEX42XLF, Redazione parmense degli Annales patavini
}

log = logging.getLogger("ingest")

# Once the model is cached, don't contact the Hugging Face Hub on every run.
if (Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-m3").is_dir():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


# ------------------------------------------------------------------ helpers

def fp_hash(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def file_fp(path: Path) -> str:
    st = path.stat()
    return f"{path}|{st.st_size}|{st.st_mtime_ns}|v{CHUNKER_VERSION}"


def names(creators: list[dict], types: set[str]) -> list[str]:
    out = []
    for c in creators:
        if c.get("creatorType") in types:
            out.append(c["name"] if "name" in c else
                       ", ".join(p for p in (c.get("lastName", ""), c.get("firstName", "")) if p))
    return out


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"([^\W\d_])[-¬­]\s*\n\s*([^\W\d_])")


def clean_text(text: str) -> str:
    """Join words hyphenated across lines, collapse whitespace, keep paragraph breaks.

    Numbers (including margin/verse line numbers) are deliberately left alone.
    """
    text = _CONTROL.sub("", text).replace(" ", " ")
    text = _HYPHEN_BREAK.sub(lambda m: m.group(1) + m.group(2) if m.group(2).islower() else m.group(0), text)
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paras if p)


def short_title(d: dict) -> str:
    """Zotero's Short Title if it is really a short form of the title, else the title up to ':' or '. '.

    (Some chapters have the book's title in Short Title, which would cite the wrong work.)
    """
    words = lambda s: set(re.findall(r"\w+", s.lower()))  # noqa: E731
    title = d.get("title", "") or ""
    st = (d.get("shortTitle") or "").strip()
    if st and words(st) <= words(title):
        return st
    return re.split(r"[:.]\s", title, maxsplit=1)[0]


def enough_text(text: str) -> bool:
    return len(re.sub(r"\s", "", text)) >= MIN_CHUNK_CHARS


BLOCK_TAGS = ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "td", "th",
              "dd", "dt", "figcaption", "section", "article", "br", "tr", "table", "ul", "ol"]


def html_to_text(raw: str | bytes, main_only: bool) -> str:
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript", "template", "svg", "nav", "iframe", "form"]):
        tag.decompose()
    for tag in soup.select('[role="navigation"], [aria-hidden="true"]'):
        tag.decompose()
    root = soup
    if main_only:
        for tag in soup(["header", "footer"]):
            tag.decompose()
        articles = soup.find_all("article")
        root = (soup.find("main") or
                (max(articles, key=lambda a: len(a.get_text())) if articles else None) or
                soup.body or soup)
    # Hyphenation joining needs a line break, so mark block boundaries with blank lines.
    for tag in root.find_all(BLOCK_TAGS):
        tag.insert_before("\n\n")
        tag.insert_after("\n\n")
    return clean_text(root.get_text())


# ----------------------------------------------------------------- chunking

class Chunker:
    def __init__(self) -> None:
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def ntokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def _units(self, text: str) -> list[tuple[str, int]]:
        """Paragraphs, falling back to sentences, then word windows, each <= CHUNK_TOKENS."""
        units = []
        for para in text.split("\n\n"):
            n = self.ntokens(para)
            if n <= CHUNK_TOKENS:
                units.append((para, n))
                continue
            for sent in re.split(r"(?<=[.!?;:])\s+", para):
                n = self.ntokens(sent)
                if n <= CHUNK_TOKENS:
                    units.append((sent, n))
                    continue
                words, buf = sent.split(), []
                for w in words:
                    buf.append(w)
                    if len(buf) % 20 == 0 and self.ntokens(" ".join(buf)) > CHUNK_TOKENS - 40:
                        units.append((" ".join(buf), self.ntokens(" ".join(buf))))
                        buf = []
                if buf:
                    units.append((" ".join(buf), self.ntokens(" ".join(buf))))
        return units

    def _tail(self, units: list[tuple[str, int]]) -> list[tuple[str, int]]:
        """Trailing units totalling <= OVERLAP_TOKENS; if none fits, the last ~60 tokens of words."""
        tail, total = [], 0
        for u in reversed(units):
            if total + u[1] > OVERLAP_TOKENS:
                break
            tail.insert(0, u)
            total += u[1]
        if tail:
            return tail
        words = units[-1][0].split()
        take = []
        for w in reversed(words):
            take.insert(0, w)
            if len(take) % 5 == 0 and self.ntokens(" ".join(take)) >= OVERLAP_TOKENS:
                break
        text = " ".join(take)
        return [(text, self.ntokens(text))]

    def split(self, text: str) -> list[str]:
        if self.ntokens(text) <= CHUNK_TOKENS:
            return [text]
        chunks, cur, total = [], [], 0
        for unit in self._units(text):
            if cur and total + unit[1] > CHUNK_TOKENS:
                chunks.append("\n\n".join(u[0] for u in cur))
                cur = self._tail(cur)
                total = sum(u[1] for u in cur)
                if total + unit[1] > CHUNK_TOKENS:
                    cur, total = [], 0
            cur.append(unit)
            total += unit[1]
        if cur:
            chunks.append("\n\n".join(u[0] for u in cur))
        return chunks


# ------------------------------------------------------------ printed pages

COVER_WORDS = [
    "stable url", "author(s):", "published by", "for additional information about this article",
    "access provided", "your use of the jstor archive", "request date", "ill number", "ill #",
    "patron:", "lender string", "borrower:", "call number", "call#", "supplied by", "tn:",
    "reproduced with permission of copyright owner", "copyright and use", "core metadata",
    "follow this and additional works", "recommended citation", "interlibrary", "odyssey:",
    "this material may be protected by copyright",
]


def cover_score(doc: pymupdf.Document, i: int, median_size: tuple[float, float]) -> int:
    page = doc[i]
    text = page.get_text().lower()
    score = 2 * sum(w in text for w in COVER_WORDS)
    w, h = page.rect.width, page.rect.height
    if abs(w - median_size[0]) / median_size[0] > 0.02 or abs(h - median_size[1]) / median_size[1] > 0.02:
        score += 2
    if len(re.sub(r"\s", "", text)) < 150:
        score += 1
    return score


def parse_page_range(pages: str) -> tuple[int, int] | None:
    ranges = re.findall(r"(\d+)\s*[-–—]\s*(\d+)", pages or "")
    if len(ranges) != 1:
        return None
    s, e = ranges[0]
    if int(e) < int(s) and len(e) < len(s):          # "141-53" -> 141-153
        e = s[: len(s) - len(e)] + e
    s, e = int(s), int(e)
    return (s, e) if e >= s else None


def safe_label(page: pymupdf.Page) -> str:
    """Page label, or "" for pages before the first label rule (malformed label tables).

    Some PDFs store the label prefix as a hex UTF-16 string that comes through undecoded,
    e.g. "<FEFF0043006F007600650072>" for "Cover" or "<FEFF>ii" for "ii".
    """
    try:
        label = page.get_label()
    except IndexError:
        return ""
    return re.sub(r"<FEFF((?:[0-9A-Fa-f]{4})*)>",
                  lambda m: bytes.fromhex(m.group(1)).decode("utf-16-be", "replace"), label)


def printed_pages(doc: pymupdf.Document, item: dict) -> tuple[list[str], str]:
    """One printed page label per PDF page ("" when unknown), and how it was derived.

    Page labels from the PDF win, unless they merely count 1..n (many publisher PDFs
    define those, and they say nothing about the printed page). Otherwise the Zotero
    Pages range is mapped onto the PDF only when the page counts match (allowing one
    cover sheet at either end). Text on the page is never used to read a page number.
    """
    n = doc.page_count
    if doc.get_page_labels():
        labels = [safe_label(doc[i]) for i in range(n)]
        if labels != [str(i + 1) for i in range(n)]:
            return labels, "pdf page labels"
        how_labels = "trivial pdf labels 1..n ignored; "
    else:
        how_labels = ""
    labels, how = _pages_field(doc, item)
    return labels, how_labels + how


def _pages_field(doc: pymupdf.Document, item: dict) -> tuple[list[str], str]:
    n = doc.page_count
    data = item["data"]
    rng = parse_page_range(data.get("pages", ""))
    if data.get("itemType") not in PRINTED_PAGE_TYPES or rng is None:
        return [""] * n, "none"
    start, end = rng
    span = end - start + 1
    if n == span:
        return [str(start + i) for i in range(n)], f"pages field {data['pages']} (exact match)"
    if n == span + 1 and n >= 3:
        mid = [doc[i].rect for i in range(1, n - 1)]
        median_size = (statistics.median(r.width for r in mid), statistics.median(r.height for r in mid))
        first, last = cover_score(doc, 0, median_size), cover_score(doc, n - 1, median_size)
        if first >= 2 and first > last:
            return [""] + [str(start + i) for i in range(n - 1)], f"pages field {data['pages']} (cover first)"
        if last >= 2 and last > first:
            return [str(start + i) for i in range(n - 1)] + [""], f"pages field {data['pages']} (cover last)"
        return [""] * n, f"pages field {data['pages']}: extra page, cover end unclear"
    return [""] * n, f"pages field {data['pages']}: {span} pages vs PDF {n}"


# --------------------------------------------------------- plain-text files

TEXT_PAGE_MARK = re.compile(r"^## p\. ?(\S*) \(#(\d+)\) #+[ \t]*$", re.M)   # HathiTrust: "## p. 7 (#47) ####"
TEXT_HEADER_END = "=" * 60                                                # MQDQ: end of the header block
TEXT_VERSE = re.compile(r"^\[(\d+)\] ")
TEXT_HEADING = re.compile(r"^(#{2,3}) (.+)$")
TEXT_FOLIO = re.compile(r"\[foglio ([^\]]*)\]")
TEXT_ANNO = re.compile(r"^\s*Anno (?:itaque )?Domini \d{4}")
TEXT_WORD = re.compile(r"[^\W\d_]{3,}")
TEXT_PATHS = {"pages": "page path (A)", "verse": "verse path (B)", "chronicle": "sectioned prose, chronicle (C)",
              "folio": "sectioned prose, folios (D)", "plain": "plain paragraphs (FLAGGED: no known format)",
              "paragraphs": "numbered paragraphs (E)"}
GIBBERISH_SCORE = 0.2          # pages scoring lower are skipped (manuscript plates); see page_vocab_scores
GIBBERISH_REVIEW = 0.35        # pages scoring lower, but kept, are listed in the dry-run report
GIBBERISH_MIN_CHARS = 200      # shorter pages are not scored
LABEL_CARRY_CHARS = 40         # a speaker label up to this length is repeated at the top of a chunk
LOCATION_CHARS = 70

# Format E (numbered paragraphs; padoue_scrape/make_openedition_txt.py writes it from an OpenEdition scrape)
TEXT_PARA_SENTINEL = "%%OPENEDITION-PARAGRAPHS v1"
TEXT_PARA_CHAPTER = re.compile(r"^%%CHAPTER (.*)$")
TEXT_PARA_ATTR = re.compile(r'(\w+)=(?:"([^"]*)"|(\S*))')
TEXT_PARA_NUM = re.compile(r"^\[¶(\d+)\] ?")
TEXT_PARA_CALL = re.compile(r"\[\^(\d+)-(\d+)\]")                       # [^568-12]: chapter 568, note 12
TEXT_PARA_DEF = re.compile(r"^\[\^(\d+)-(\d+)\]: ?")
TEXT_PARA_FIGURE = re.compile(r"^\[FIGURE: (.*) — image file: [^\]]*\]$")
TEXT_PARA_HEAD = re.compile(r"^(#{1,6}) (.+)$")
TEXT_PARA_SEP = "* * *"
TEXT_PARA_LEADIN_CHARS = 80    # short unpunctuated line right before a [¶N] (e.g. a consul's name) leads into it
# A period after these does not end a sentence ("cf. Bortolami", "op. cit. p. 12", "t. II. ").
FR_ABBREV = {
    "p", "pp", "cf", "fol", "fols", "ch", "chap", "n", "nn", "no", "nos", "t", "vol", "vols", "éd", "éds", "ed", "edd",
    "dir", "op", "cit", "ca", "s.v", "sv", "ibid", "id", "art", "doc", "docs", "ms", "mss", "fasc", "col", "coll",
    "loc", "trad", "suppl", "sq", "sqq", "ss", "st", "ste", "mgr", "av", "apr", "env", "corp", "soppr", "dipl",
    "proc", "reg", "cod", "fig", "pl", "tav", "tab", "cap", "lib", "can", "sec", "saec", "not", "vv", "cfr",
    "réf", "rééd", "mm", "mme", "mlle", "dr", "prof", "s.d", "s.l", "a.c", "j.-c", "ann", "app",
}
FR_ROMAN = re.compile(r"^(?!(?:Le|De|Ce|Me|Mi|Di|Ci|Li|Vi|Mille)$)[IVXLCDM]+(?:er|re|e)?$")
FR_BOUNDARY = re.compile(r"[.!?…]+[»”\"’)\]]*\s+(?=[«“\"(\[]?\s?[A-ZÀ-ÖØ-Þ])")


def md_plain(s: str) -> str:
    """Format E chunk text: no footnote calls, no Markdown emphasis, \\* \\_ \\\\ unescaped."""
    s = re.sub(r"\[\^[\w-]+\]", "", s)
    s = re.sub(r"(?<!\\)\*", "", s)
    s = re.sub(r"\\([\\*_])", r"\1", s)
    return " ".join(s.split())


def figure_title(caption: str) -> str:
    """The real title(s) of a [FIGURE: ...] caption; '' for "Image 3.jpg", "(à suivre) Note" and the like."""
    keep = []
    for part in caption.split(" — "):
        t = md_plain(part)
        rest = re.sub(r"\(à suivre\)|\bNote\b", "", t).strip(" .,;:")
        if rest and not re.fullmatch(r"(?:Image\s+)?(?:img[-_]?\d+|\d+)\.jpe?g", rest, re.I):
            keep.append(t)
    return " — ".join(keep)


def fr_sentence_ends(text: str, inside_quotes: bool) -> list[int]:
    """Offsets where a French sentence ends (start of the following whitespace), skipping abbreviations,
    initials ("S. Bortolami", "C. G. Mor"), ordinals ("Ier.", "XIIe.") and, unless inside_quotes, «...»."""
    ends = []
    for m in FR_BOUNDARY.finditer(text):
        punct_end = m.start() + len(m.group(0).rstrip())
        if text[m.start()] == ".":
            prev = re.search(r"(\S+)$", text[:m.start()])
            word = prev.group(1).lstrip("([«“\"") if prev else ""
            w = word.lower().rstrip(".")
            if (len(w) == 1 and w.isalpha()) or w in FR_ABBREV or re.fullmatch(r"(?:\w\.)+\w", w) \
                    or re.fullmatch(r"\w\.?-\w", w) or FR_ROMAN.match(word):
                continue
        if not inside_quotes and text.count("«", 0, punct_end) > text.count("»", 0, punct_end):
            continue
        ends.append(punct_end)
    return ends


def fr_sentences(text: str, inside_quotes: bool = False) -> list[str]:
    cuts = [0] + fr_sentence_ends(text, inside_quotes) + [len(text)]
    return [s for s in (text[a:b].strip() for a, b in zip(cuts, cuts[1:])) if s]


def para_location(label: str, a: int, b: int, suffix: str = "") -> str:
    """"<label>, ¶a–b<suffix>" within LOCATION_CHARS: the label is shortened, never the ¶ range."""
    rng = (f"¶{a}" if a == b else f"¶{a}–{b}") if a else ""
    def loc(lab: str) -> str:
        return ", ".join(p for p in (lab, rng) if p) + suffix
    while len(loc(label)) > LOCATION_CHARS and " " in label:
        label = label.rstrip("…").rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
    return loc(label)


def nonws(s: str) -> int:
    return len(re.sub(r"\s", "", s))


def is_xml(a: dict) -> bool:
    return ((a.get("contentType") or "").lower() in ("text/xml", "application/xml")
            or (a.get("filename") or "").lower().endswith(".xml"))


def is_text(a: dict) -> bool:
    ctype = (a.get("contentType") or "").lower()
    return ctype == "text/plain" or (not ctype and (a.get("filename") or "").lower().endswith(".txt"))


def overlap_len(a: str, b: str) -> int:
    """Length of the longest suffix of a that is also a prefix of b (the overlap between two chunks)."""
    probe = b[:30]
    i = a.find(probe, max(0, len(a) - 4000))
    while i != -1:
        if b.startswith(a[i:]):
            return len(a) - i
        i = a.find(probe, i + 1)
    return 0


def _mostly_upper(s: str) -> bool:
    """Rubric text: >= 2 words, starting with two capitals, at most one lowercase letter per 15 (OCR slips)."""
    letters = [c for c in s if c.isalpha()]
    return (len(letters) >= 4 and len(s.split()) >= 2 and s[:2].isupper()
            and sum(c.islower() for c in letters) <= max(1, len(letters) // 15))


def find_rubrics(lines: list[str]) -> list[tuple[int, int, str, bool]]:
    """Chapter rubrics "UPPERCASE RUBRIC - text": (first line, last line, rubric, text follows on the last line).

    The rubric may wrap onto a second line, with the " - " either inside it or at its start ("- Anno ...").
    """
    dash = re.compile(r"^(.*?)\s-(?:\s|$)(.*)")
    out, i = [], 0
    while i < len(lines):
        m = dash.match(lines[i])
        if m and _mostly_upper(m.group(1)):
            out.append((i, i, m.group(1).strip(), bool(m.group(2).strip())))
            i += 1
            continue
        if not m and _mostly_upper(lines[i]) and i + 1 < len(lines):
            nxt = lines[i + 1]
            m2 = re.match(r"^-(?:\s|$)(.*)", nxt)
            if m2:
                out.append((i, i + 1, lines[i].strip(), bool(m2.group(1).strip())))
                i += 2
                continue
            m2 = dash.match(nxt)
            if m2 and _mostly_upper(m2.group(1)):
                out.append((i, i + 1, f"{lines[i].strip()} {m2.group(1).strip()}", bool(m2.group(2).strip())))
                i += 2
                continue
        i += 1
    return out


def detect_text_format(text: str) -> str:
    lines = text.split("\n")
    if lines[0].rstrip() == TEXT_PARA_SENTINEL:          # explicit marker only, never a heuristic
        return "paragraphs"
    if TEXT_PAGE_MARK.search(text):
        return "pages"
    if TEXT_HEADER_END in lines and any(TEXT_VERSE.match(l) for l in lines[lines.index(TEXT_HEADER_END) + 1:]):
        return "verse"
    if TEXT_FOLIO.search(text):
        return "folio"
    if len(find_rubrics(lines)) >= 10 or sum(bool(TEXT_ANNO.match(l)) for l in lines) >= 5:
        return "chronicle"
    return "plain"


def page_vocab_scores(texts: list[str]) -> list[float]:
    """Per page: share of its words (3+ letters) that occur on at least 3 other pages of the same file.

    OCR of manuscript plates is gibberish that shares little vocabulary with the edition's pages.
    (lingua's confidence cannot tell: it is relative across the five languages, so gibberish still
    scores ~1.0 for one of them.)
    """
    words = [TEXT_WORD.findall(t.lower()) for t in texts]
    df = Counter(w for ws in words for w in set(ws))
    return [sum(df[w] - 1 >= 3 for w in ws) / len(ws) if ws else 0.0 for ws in words]


def clean_chronicle_noise(lines: list[str]) -> tuple[list[str], Counter]:
    """Drop "/*" lines (paragraph breaks), append lone "."/"," to the previous line, rejoin "[" and "]"
    split onto their own lines or line edges: "expletis / [ / quinquaginta octo / ] annis"."""
    out: list[str] = []
    noise: Counter = Counter()
    carry = ""
    for line in lines:
        s = line.strip()
        if s == "/*":
            noise['"/*" line (paragraph break)'] += 1
            out.append("")
            continue
        if s in (".", ",", "]") and out:
            noise[f'lone "{s}" line'] += 1
            out[-1] = out[-1].rstrip() + s
            continue
        if s == "[":
            noise['lone "[" line'] += 1
            carry = "["
            continue
        if s.startswith("]") and out:
            noise['"]" at line start'] += 1
            out[-1] = out[-1].rstrip() + "]"
            line = s[1:].lstrip()
        if carry:
            line, carry = carry + line.lstrip(), ""
        if line.rstrip().endswith(" ["):
            noise['"[" at line end'] += 1
            line, carry = line.rstrip()[:-1].rstrip(), "["
        out.append(line)
    return out, noise


def verse_sections(body: list[str]) -> list[tuple[str, list[str]]]:
    """MQDQ body -> [(section label, lines)]; "## a" / "### b" headings give "a", "a, b" or "b"."""
    sections, label, parent, cur = [], "", "", []
    for line in body:
        line = line.rstrip()
        if not line.strip():
            continue
        h = TEXT_HEADING.match(line)
        if h:
            if cur:
                sections.append((label, cur))
            cur = []
            name = h.group(2).strip()
            if len(h.group(1)) == 2:
                parent = label = name
            else:
                label = f"{parent}, {name}" if parent else name
            continue
        cur.append(line)
    if cur:
        sections.append((label, cur))
    return sections


def rubric_location(year: int | None, rubric: str | None) -> str:
    loc = f"a. {year}" if year else ""
    if rubric:
        r = rubric.strip().rstrip(".").strip()
        loc = ", ".join(p for p in (loc, r[:1].upper() + r[1:].lower()) if p)
    return loc if len(loc) <= LOCATION_CHARS else loc[:LOCATION_CHARS - 1].rstrip(" ,") + "…"


# ------------------------------------------------------------ source model

@dataclass
class Chunk:
    id: str
    text: str
    meta: dict
    detect_lang: bool = True      # False for generated text (tag-only notes)


@dataclass
class Source:
    id: str
    item_key: str
    content_fp: str
    meta_fp: str
    build: Callable[[], list[Chunk]]
    chunks: list[Chunk] = field(default_factory=list)


@dataclass
class ItemInfo:
    item: dict
    collections: list[str]
    attachments: list[dict] = field(default_factory=list)
    notes: list[dict] = field(default_factory=list)

    @property
    def key(self) -> str:
        return self.item["key"]

    @property
    def data(self) -> dict:
        return self.item["data"]

    def base_meta(self) -> dict:
        d = self.data
        return dict(
            item_key=self.key,
            attachment_key="",
            authors="; ".join(names(d.get("creators", []), {"author"})),
            year=year_of(self.item),
            title=d.get("title", ""),
            item_type=d.get("itemType", ""),
            publication=next((d[f] for f in PUBLICATION_FIELDS if d.get(f)), ""),
            collections="; ".join(sorted(self.collections)),
            tags="; ".join(sorted(t["tag"] for t in d.get("tags", []))),
            source_type="",
            pdf_page=0,
            printed_page="",
            ocr=False,
            language="",
            my_annotations_on_page=0,
            has_ink_on_page=False,
            page_annotation_tags="",
            annotation_tags="",
            url="",
        )

    def fp(self) -> str:
        return fp_hash(self.data.get("dateModified"), sorted(self.collections))


# --------------------------------------------------------------- gathering

class Library:
    """Live snapshot of the collection from the Zotero local API (GET only)."""

    def __init__(self, item_filter: set[str] | None) -> None:
        self.api = ZoteroAPI()
        self.api.check_connection()
        col_names = collect_collections(self.api)
        items: dict[str, dict] = {}
        for ck in col_names:
            for it in self.api.get_all(f"collections/{ck}/items/top"):
                items.setdefault(it["key"], it)
        log.info("Zotero: %d top-level items in %d collections", len(items), len(col_names))
        if item_filter:
            missing = item_filter - items.keys()
            if missing:
                sys.exit(f"Not in the collection: {', '.join(sorted(missing))}")
            items = {k: v for k, v in items.items() if k in item_filter}
        self.items: dict[str, ItemInfo] = {}
        for key, it in tqdm(items.items(), desc="Reading Zotero items", unit="item"):
            info = ItemInfo(it, [col_names[c] for c in it["data"].get("collections", []) if c in col_names])
            itype = it["data"]["itemType"]
            if itype == "attachment":
                info.attachments.append(it)
            elif itype == "note":
                info.notes.append(it)
            elif it.get("meta", {}).get("numChildren", 0):
                for child in self.api.get_all(f"items/{key}/children"):
                    ctype = child["data"]["itemType"]
                    if ctype == "attachment":
                        info.attachments.append(child)
                    elif ctype == "note":
                        info.notes.append(child)
            self.items[key] = info
        att_keys = {a["key"] for i in self.items.values() for a in i.attachments}
        self.annotations: dict[str, list[dict]] = defaultdict(list)
        for ann in self.api.get_all("items", itemType="annotation"):
            parent = ann["data"].get("parentItem")
            if parent in att_keys:
                self.annotations[parent].append(ann["data"])
        self.adobe: dict[str, list[dict]] = defaultdict(list)
        with open(ADOBE_CSV, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row["attachment_key"] in att_keys:
                    self.adobe[row["attachment_key"]].append(row)
        self.ocr: dict[str, Path] = {}
        if OCR_MANIFEST.exists():
            with open(OCR_MANIFEST, encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    self.ocr[row["attachment_key"]] = PROJECT_DIR / row["ocr_pdf"]
        log.info("Zotero: %d attachments, %d annotations, %d Adobe annotations (in collection)",
                 len(att_keys), sum(map(len, self.annotations.values())), sum(map(len, self.adobe.values())))


# ---------------------------------------------------------------- builders

def ann_page(ann: dict) -> int:
    try:
        return int(json.loads(ann.get("annotationPosition") or "{}").get("pageIndex")) + 1
    except (TypeError, ValueError):
        return 0


def norm_ann(text: str) -> str:
    return re.sub(r"\W+", "", (text or "").lower())


def tag_list(data: dict) -> list[str]:
    return sorted({t["tag"] for t in data.get("tags", []) if t.get("tag")}, key=str.lower)


def has_content(ann: dict) -> bool:
    """Text, a comment or tags: anything worth a chunk (tag-only notes hold tags on purpose)."""
    return bool((ann.get("annotationText") or "").strip() or (ann.get("annotationComment") or "").strip()
                or tag_list(ann))


def annotation_text(highlight: str, comment: str, tags: list[str] = (), where: str = "") -> str:
    """Chunk text for one annotation; a note with only tags becomes "Tagged note on p. X: a; b"."""
    parts = []
    if highlight.strip():
        parts.append(f"Highlighted: {highlight.strip()}")
    if comment.strip():
        parts.append(f"My comment: {comment.strip()}")
    if tags:
        parts.append(f"Tags: {'; '.join(tags)}" if parts else
                     f"Tagged note{' on ' + where if where else ''}: {'; '.join(tags)}")
    return "\n".join(parts)


def page_ref(printed: str, pdf_page: int) -> str:
    return f"p. {printed}" if printed else (f"PDF p. {pdf_page}" if pdf_page else "")


class Builder:
    def __init__(self, lib: Library, chunker: Chunker) -> None:
        self.lib = lib
        self.chunker = chunker
        self.detector = LanguageDetectorBuilder.from_languages(*LANGUAGES).build()
        self.page_info_cache: dict[str, tuple[list[str], str]] = {}
        self.text_report: dict[str, dict] = {}     # per .txt attachment, filled when its source is built
        self.ignored_xml = 0

    # -- language
    def set_languages(self, chunks: list[Chunk]) -> None:
        todo = [c for c in chunks if c.detect_lang and len(re.findall(r"[^\W\d_]", c.text)) >= 20]
        if not todo:
            return
        for c, lang in zip(todo, self.detector.detect_languages_in_parallel_of([c.text for c in todo])):
            c.meta["language"] = lang.name.title() if lang else ""

    # -- context header
    def header(self, c: Chunk) -> str:
        """e.g. "[Bisanti 1994 · Albertino Mussato e l'"Octavia" · p. 391]"; no page for snapshots/metadata.

        Text files give their location instead when they have one: "vv. 1-14", "a. 1208, ...", "fol. I recto".
        """
        d = self.lib.items[c.meta["item_key"]].data
        creators = d.get("creators", [])
        first = next((x for x in creators if x.get("creatorType") == "author"), creators[0] if creators else None)
        surname = (first.get("lastName") or first.get("name", "")) if first else ""
        title = short_title(d)
        words = title.split()
        title = " ".join(words[:8]) + ("…" if len(words) > 8 else "")
        parts = [" ".join(p for p in (surname, c.meta["year"]) if p), title]
        if c.meta.get("location"):
            parts.append(c.meta["location"])
        elif c.meta["source_type"] not in ("snapshot", "metadata"):
            parts.append(page_ref(c.meta["printed_page"], c.meta["pdf_page"]))
        return "[" + " · ".join(p for p in parts if p) + "]"

    # -- files
    def pdf_path(self, att: dict) -> tuple[Path | None, bool]:
        key = att["key"]
        if key in self.lib.ocr:
            if self.lib.ocr[key].is_file():
                return self.lib.ocr[key], True
            log.warning("%s: OCR copy %s missing; using original", key, self.lib.ocr[key])
        path = resolve_path({"data": att})
        return (path if path and path.is_file() else None), False

    def printed(self, info: ItemInfo, att: dict) -> tuple[list[str], str]:
        key = att["key"]
        if key not in self.page_info_cache:
            path, _ = self.pdf_path(att)
            if path is None:
                self.page_info_cache[key] = ([], "no file")
            else:
                with pymupdf.open(path) as doc:
                    self.page_info_cache[key] = printed_pages(doc, info.item)
        return self.page_info_cache[key]

    def adobe_rows(self, att_key: str) -> list[dict]:
        """Adobe annotations, minus those Zotero already has (same page and text)."""
        zot = {(ann_page(a), norm_ann(a.get("annotationText") or a.get("annotationComment")))
               for a in self.lib.annotations.get(att_key, [])}
        return [r for r in self.lib.adobe.get(att_key, [])
                if (int(r["page_number"]), norm_ann(r["highlighted_text"] or r["comment"])) not in zot]

    # -- sources
    def sources(self) -> list[Source]:
        out: list[Source] = []
        for info in self.lib.items.values():
            out.append(self.meta_source(info))
            for note in info.notes:
                out.append(self.note_source(info, note))
            for att in info.attachments:
                a = att["data"]
                ctype = (a.get("contentType") or "").lower()
                if is_xml(a):
                    self.ignored_xml += 1          # same text as the item's .txt; indexing both would duplicate it
                    continue
                if is_text(a):
                    s = self.text_source(info, a)
                    if s:
                        out.append(s)
                elif ctype == "application/pdf":
                    s = self.pdf_source(info, a)
                    if s:
                        out.append(s)
                    out.extend(self.adobe_sources(info, a))
                elif ctype in ("text/html", "application/xhtml+xml"):
                    s = self.snapshot_source(info, a)
                    if s:
                        out.append(s)
                out.extend(self.annotation_sources(info, a, ctype == "application/pdf"))
        return out

    def meta_source(self, info: ItemInfo) -> Source:
        def build() -> list[Chunk]:
            d = info.data
            m = info.base_meta()
            lines = [f"Title: {m['title']}"]
            if m["authors"]:
                lines.append(f"Authors: {m['authors']}")
            eds = names(d.get("creators", []), {"editor", "seriesEditor"})
            if eds:
                lines.append(f"Editors: {'; '.join(eds)}")
            if m["year"]:
                lines.append(f"Year: {m['year']}")
            lines.append(f"Item type: {m['item_type']}")
            if m["publication"]:
                lines.append(f"Publication: {m['publication']}")
            if d.get("abstractNote", "").strip():
                lines.append(f"Abstract: {' '.join(d['abstractNote'].split())}")
            if m["tags"]:
                lines.append(f"Tags: {m['tags']}")
            m.update(source_type="metadata")
            return [Chunk(f"{info.key}:meta", "\n".join(lines), m)]
        return Source(f"meta:{info.key}", info.key, fp_hash(info.data.get("dateModified"), CHUNKER_VERSION),
                      info.fp(), build)

    def note_source(self, info: ItemInfo, note: dict) -> Source:
        n = note["data"]

        tags = tag_list(n)

        def build() -> list[Chunk]:
            text = html_to_text(n.get("note", ""), main_only=False)
            m = dict(info.base_meta(), source_type="child_note", annotation_tags="; ".join(tags))
            if not enough_text(text):
                # A note that exists only to hold tags still gets a chunk; a truly empty one does not.
                return [Chunk(f"{n['key']}:c1", annotation_text("", "", tags), m, detect_lang=False)] if tags else []
            suffix = f"\nTags: {'; '.join(tags)}" if tags else ""
            return [Chunk(f"{n['key']}:c{i}", t + suffix, dict(m))
                    for i, t in enumerate(self.chunker.split(text), 1)]
        return Source(f"note:{n['key']}", info.key, fp_hash(n.get("dateModified"), tags, CHUNKER_VERSION),
                      info.fp(), build)

    def snapshot_source(self, info: ItemInfo, a: dict) -> Source | None:
        path = resolve_path({"data": a})
        if not path or not path.is_file():
            return None

        def build() -> list[Chunk]:
            text = html_to_text(path.read_bytes(), main_only=True)
            if not enough_text(text):
                return []
            m = dict(info.base_meta(), source_type="snapshot", attachment_key=a["key"],
                     url=a.get("url") or info.data.get("url", ""))
            return [Chunk(f"{a['key']}:s:c{i}", t, dict(m)) for i, t in enumerate(self.chunker.split(text), 1)]
        return Source(f"snapshot:{a['key']}", info.key, file_fp(path), info.fp(), build)

    # -- plain-text files (.txt): editions of primary sources
    def text_source(self, info: ItemInfo, a: dict) -> Source | None:
        path = resolve_path({"data": a})
        if not path or not path.is_file():
            log.warning("%s: text file not found; skipped", a["key"])
            return None

        def build() -> list[Chunk]:
            text = path.read_bytes().decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
            fmt = detect_text_format(text)
            report = {"attachment_key": a["key"], "item_key": info.key, "file": path.name, "format": fmt,
                      "skipped": [], "review": [], "details": {}}
            pieces, expected, repeated = getattr(self, f"_text_{fmt}")(text, report)
            base = dict(info.base_meta(), source_type="text_file", attachment_key=a["key"], ocr=fmt == "pages",
                        location="", chunk_seq=0)
            # A piece is (text, location, printed) or, for format E, (text, location, printed, extra meta).
            chunks = [Chunk(f"{a['key']}:t:c{seq + 1}", t, dict(base, location=loc, printed_page=printed,
                                                                chunk_seq=seq, **(extra[0] if extra else {})))
                      for seq, (t, loc, printed, *extra) in enumerate(pieces)]
            got = sum(nonws(c.text) for c in chunks) - repeated
            report.update(chunks=len(chunks), expected_chars=expected, chunk_chars=got,
                          diff_pct=100 * (got - expected) / expected if expected else 0.0)
            self.text_report[a["key"]] = report
            log.info("%s (%s): %s, %d chunks", a["key"], path.name, TEXT_PATHS[fmt], len(chunks))
            if abs(report["diff_pct"]) > 1:
                log.warning("%s: coverage check failed: %d chars in chunks vs %d expected (%+.2f%%)",
                            a["key"], got, expected, report["diff_pct"])
            return chunks
        return Source(f"text:{a['key']}", info.key, file_fp(path), info.fp(), build)

    def _split(self, text: str, location: str, printed: str = "") -> tuple[list[tuple[str, str, str]], int]:
        """Chunker pieces of one page/section, and the non-space chars their overlaps repeat."""
        parts = self.chunker.split(text)
        repeated = sum(nonws(b[:overlap_len(p, b)]) for p, b in zip(parts, parts[1:]))
        return [(t, location, printed) for t in parts], repeated

    def _text_pages(self, text: str, report: dict):
        """Format A (HathiTrust): one page = one or more chunks, cited by the marker's printed label."""
        marks = list(TEXT_PAGE_MARK.finditer(text))
        pages = [("", 0, text[:marks[0].start()])] if text[:marks[0].start()].strip() else []
        pages += [(m.group(1), int(m.group(2)), text[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(text)])
                  for i, m in enumerate(marks)]
        cleaned = [clean_text(body) for _, _, body in pages]
        scores = page_vocab_scores(cleaned)
        expected = nonws(text) - sum(nonws(m.group(0)) for m in marks)
        overrides = HATHI_LABEL_OVERRIDES.get(report["attachment_key"], {})
        pieces, repeated, prev_label = [], 0, None
        for (hathi_label, scan, body), clean, score in zip(pages, cleaned, scores):
            label = overrides.get(scan, hathi_label)
            long = nonws(clean) > GIBBERISH_MIN_CHARS
            # Repeats are judged on Hathi's own labels, so the plates after an overridden page still count.
            reason = ("repeated label" if hathi_label and hathi_label == prev_label and scan not in overrides else
                      "too little text" if not enough_text(clean) else
                      "gibberish" if long and score < GIBBERISH_SCORE else None)
            prev_label = hathi_label
            entry = {"label": label, "scan": scan, "chars": nonws(clean), "score": round(score, 3) if long else None}
            if reason:
                if enough_text(clean):
                    cv = self.detector.compute_language_confidence_values(clean)
                    entry["lingua"] = f"{cv[0].language.name.title()} {cv[0].value:.3f}" if cv else ""
                report["skipped"].append(dict(entry, reason=reason))
                expected -= nonws(body)
                continue
            if long and score < GIBBERISH_REVIEW:
                report["review"].append(entry)
            p, r = self._split(clean, "", label)
            pieces += p
            repeated += r
        report["details"]["pages"] = f"{len(pages)} ({len(pages) - len(report['skipped'])} indexed)"
        return pieces, expected, repeated

    def _text_verse(self, text: str, report: dict):
        """Format B (MQDQ): whole lines packed into chunks within sections, located by verse range."""
        lines = text.split("\n")
        body = lines[lines.index(TEXT_HEADER_END) + 1:]
        headings = [l for l in body if TEXT_HEADING.match(l.rstrip())]
        expected = sum(map(nonws, body)) - sum(map(nonws, headings))
        pieces, repeated, verse_lines, labels_carried = [], 0, 0, 0
        sections = verse_sections(body)
        for sec, sec_lines in sections:
            units = []                         # (text, tokens, verse number or None, source line index)
            for idx, line in enumerate(sec_lines):
                v = TEXT_VERSE.match(line)
                vno = int(v.group(1)) if v else None
                n = self.chunker.ntokens(line)
                for part in ([line] if n <= CHUNK_TOKENS else self.chunker.split(line)):
                    units.append((part, n if part is line else self.chunker.ntokens(part), vno, idx))
            groups, cur, total = [], [], 0
            for j, u in enumerate(units):
                if cur and total + u[1] > CHUNK_TOKENS:
                    groups.append(cur)
                    tail, t = [], 0
                    for k in reversed(cur):
                        if t + units[k][1] > OVERLAP_TOKENS:
                            break
                        tail.insert(0, k)
                        t += units[k][1]
                    if len(tail) == len(cur) or t + u[1] > CHUNK_TOKENS:
                        tail, t = [], 0
                    cur, total = tail, t
                cur.append(j)
                total += u[1]
            if cur:
                groups.append(cur)
            last = -1                          # highest unit index already in an earlier chunk
            for g in groups:
                texts = [units[k][0] for k in g]
                rep = sum(nonws(units[k][0]) for k in g if k <= last)
                if last >= 0 and units[g[0]][2] is not None:
                    # Chunk starts inside a speech: repeat the speaker's label (short label lines only).
                    lab = next((units[k][0] for k in range(g[0] - 1, -1, -1) if units[k][2] is None), None)
                    if lab and len(lab.strip()) <= LABEL_CARRY_CHARS:
                        texts.insert(0, lab)
                        rep += nonws(lab)
                        labels_carried += 1
                verse_lines += len({units[k][3] for k in g if k > last and units[k][2] is not None})
                nums = [units[k][2] for k in g if units[k][2] is not None]
                vv = f"vv. {min(nums)}-{max(nums)}" if nums else ""
                pieces.append(("\n".join(texts), ", ".join(p for p in (sec, vv) if p), ""))
                repeated += rep
                last = max(last, g[-1])
        report["details"].update(
            sections=len(sections), verse_lines_in_file=sum(bool(TEXT_VERSE.match(l)) for l in body),
            verse_lines_in_chunks=verse_lines, speaker_labels_carried_forward=labels_carried)
        return pieces, expected, repeated

    def _text_chronicle(self, text: str, report: dict):
        """Format C: prologue, rubricated chapters, then "Anno Domini NNNN" year sections."""
        raw = text.split("\n")
        lines, noise = clean_chronicle_noise(raw)
        expected = nonws(text) - 2 * noise['"/*" line (paragraph break)']
        rubrics = {r[0]: r for r in find_rubrics(lines)}
        in_chapter = {r[1] + 1 for r in rubrics.values() if not r[3]}   # "RUBRIC -" then "Anno Domini" below
        starts = []
        for i, line in enumerate(lines):
            if i in rubrics:
                starts.append((i, rubrics[i][2]))
            elif TEXT_ANNO.match(line) and i not in in_chapter:
                starts.append((i, None))
        if not starts or starts[0][0] > 0:
            starts.insert(0, (0, "prologus"))
        pieces, repeated, year, backwards, labels, long_secs = [], 0, None, [], [], []
        for j, (s, rubric) in enumerate(starts):
            e = starts[j + 1][0] if j + 1 < len(starts) else len(lines)
            sec = clean_text("\n".join(lines[s:e]))
            if not sec:
                continue
            if rubric == "prologus":
                loc = "prologus"
            else:
                m = re.search(r"Domini (\d{4})", sec) or re.search(r"\bAnno (\d{4})", sec[:120])
                if m:
                    if year and int(m.group(1)) < year:
                        backwards.append(f"line {s + 1}: {year} -> {m.group(1)}")
                    year = int(m.group(1))
                loc = rubric_location(year, rubric)
            labels.append(loc)
            ntok = self.chunker.ntokens(sec)
            if ntok > 1500:
                long_secs.append(f"{loc} ({ntok} tokens)")
            p, r = self._split(sec, loc)
            pieces += p
            repeated += r
        report["details"].update(
            sections=len(labels), rubrics=len(rubrics),
            noise_lines_cleaned=f"{sum(noise.values())} ({', '.join(f'{k}: {v}' for k, v in noise.items())})",
            years_going_backwards=backwards or "none", sections_over_1500_tokens=long_secs or "none",
            first_10_sections=labels[:10], last_10_sections=labels[-10:])
        return pieces, expected, repeated

    def _text_folio(self, text: str, report: dict):
        """Format D: one section per "[foglio ...]" marker; the marker becomes the location."""
        marks = list(TEXT_FOLIO.finditer(text))
        bounds = [(0, "")] + [(m.end(), "fol. " + " ".join(m.group(1).replace(",", "").split())) for m in marks]
        ends = [m.start() for m in marks] + [len(text)]
        pieces, repeated = [], 0
        for (start, loc), end in zip(bounds, ends):
            sec = clean_text(text[start:end])
            if sec:
                p, r = self._split(sec, loc)
                pieces += p
                repeated += r
        report["details"]["folios"] = [loc for _, loc in bounds[1:]]
        return pieces, nonws(text) - sum(nonws(m.group(0)) for m in marks), repeated

    def _text_plain(self, text: str, report: dict):
        """No known format: chunked like a web snapshot (and flagged in the report)."""
        pieces, repeated = self._split(clean_text(text), "")
        return pieces, nonws(text), repeated

    def _text_paragraphs(self, text: str, report: dict):
        """Format E (numbered paragraphs, e.g. an OpenEdition book; see TEXT_PARA_SENTINEL).

        "%%CHAPTER id=.. part=.. label=".." kind=body|bibliography" lines delimit chapters. Whole [¶N]
        paragraphs are packed within sections (never across a heading or "* * *"), located as
        "<label>, ¶a–b"; the footnotes they call go into separate "notes" chunks with the same ¶ range.
        """
        lines = text.split("\n")
        heads = [i for i, l in enumerate(lines) if TEXT_PARA_CHAPTER.match(l)]
        st = dict(paragraphs=0, entries=0, notes=0, kinds=Counter(), chunks=Counter(), split=0, via_figure=0,
                  via_title=0, via_heading=0, leadins=[], unparagraphed=0, gaps=[], footnotes=[], other=[])
        samples: list[tuple[str, str, str]] = []
        pieces, repeated = [], 0
        for k, h in enumerate(heads):
            attrs = {m.group(1): m.group(2) if m.group(2) is not None else m.group(3)
                     for m in TEXT_PARA_ATTR.finditer(TEXT_PARA_CHAPTER.match(lines[h]).group(1))}
            body = lines[h + 1:heads[k + 1] if k + 1 < len(heads) else len(lines)]
            p, r = self._para_chapter(attrs, body, st, samples)
            pieces += p
            repeated += r
        report["details"].update(
            chapters=f"{len(heads)} ({', '.join(f'{v} {k}' for k, v in st['kinds'].items())})",
            paragraphs_parsed=st["paragraphs"],
            paragraphs_per_kind=f"body: {st['paragraphs']} numbered paragraphs; "
                                f"bibliography: {st['entries']} entries",
            notes_parsed=st["notes"],
            chunks_by_kind=dict(st["chunks"]),
            paragraphs_over_chunk_tokens_split=st["split"],
            notes_attached_via_figure_placeholders=st["via_figure"],
            notes_attached_via_chapter_titles=st["via_title"],
            notes_attached_via_subheadings=st["via_heading"],
            leadin_lines_attached_to_next_paragraph=f"{len(st['leadins'])} {st['leadins'][:5]}"
                                                    + (" ..." if len(st["leadins"]) > 5 else ""),
            unnumbered_blocks_without_a_paragraph=st["unparagraphed"],
            paragraph_numbering=st["gaps"] or "1..N with no gaps in every body chapter",
            footnote_calls_vs_definitions=st["footnotes"] or "equal in every chapter (each note called once)",
            other_problems=st["other"] or "none")
        report["split_samples"] = samples
        return pieces, self._para_expected(lines), repeated

    def _para_expected(self, lines: list[str]) -> int:
        """Non-space chars that format E content should put in chunks, computed from the raw lines:
        all text minus markers ([¶N], [^..], Markdown), headings (only carried as prefixes), separators
        and figure placeholders (except their real titles)."""
        total = 0
        for line in lines[1:]:
            s = line.strip()
            if not s or TEXT_PARA_CHAPTER.match(s) or TEXT_PARA_HEAD.match(s) or s in ("---", TEXT_PARA_SEP):
                continue
            f = TEXT_PARA_FIGURE.match(s)
            if f:
                total += nonws(figure_title(f.group(1)))
                continue
            s = TEXT_PARA_DEF.sub("", TEXT_PARA_NUM.sub("", re.sub(r"^>\s?", "", s)))
            total += nonws(md_plain(s))
        return total

    def _para_fit(self, text: str, budget: int, samples: list) -> list[tuple[str, int]]:
        """One over-long paragraph -> pieces <= budget: French sentences (outside «...» first), then
        clauses at ; and :, then word windows as a last resort."""
        ntok = self.chunker.ntokens
        out: list[tuple[str, int, str]] = []
        for s in fr_sentences(text):
            n = ntok(s)
            if n <= budget:
                out.append((s, n, "sentence"))
                continue
            for s2 in fr_sentences(s, inside_quotes=True):
                n = ntok(s2)
                if n <= budget:
                    out.append((s2, n, "sentence inside «»"))
                    continue
                for s3 in re.split(r"(?<=[;:])\s+", s2):
                    n = ntok(s3)
                    if n <= budget:
                        out.append((s3, n, "clause (; :)"))
                        continue
                    buf: list[str] = []
                    for w in s3.split():
                        buf.append(w)
                        if len(buf) % 20 == 0 and ntok(" ".join(buf)) > budget - 40:
                            out.append((" ".join(buf), ntok(" ".join(buf)), "word window"))
                            buf = []
                    if buf:
                        out.append((" ".join(buf), ntok(" ".join(buf)), "word window"))
        samples.extend((a[0], b[0], b[2]) for a, b in zip(out, out[1:]))
        return [(t, n) for t, n, _ in out]

    def _para_pack_split(self, texts: list[str], budget: int, samples: list) -> tuple[list[str], int]:
        """Over-long unit -> chunk texts with sentence overlap (Chunker._tail); returns (texts, repeated)."""
        ntok = self.chunker.ntokens
        units = [u for t in texts for u in ([(t, ntok(t))] if ntok(t) <= budget else self._para_fit(t, budget, samples))]
        chunks, cur, total, rep = [], [], 0, 0
        for u in units:
            if cur and total + u[1] > budget:
                chunks.append(cur)
                tail = self.chunker._tail(cur)
                t = sum(x[1] for x in tail)
                if t + u[1] > budget:
                    tail, t = [], 0
                rep += sum(nonws(x[0]) for x in tail)
                cur, total = list(tail), t
            cur.append(u)
            total += u[1]
        if cur:
            chunks.append(cur)
        return [" ".join(x[0] for x in c) for c in chunks], rep

    def _para_chapter(self, attrs: dict, body: list[str], st: dict, samples: list):
        ntok = self.chunker.ntokens
        cid, label, part = attrs.get("id", ""), attrs.get("label", ""), attrs.get("part", "")
        kind = attrs.get("kind", "body")
        st["kinds"][kind] += 1
        try:
            cut = body.index("---")
        except ValueError:
            cut = len(body)
        main, note_lines = body[:cut], body[cut + 1:]

        # -- footnote definitions (continuation lines are indented)
        notes: dict[str, str] = {}
        last = None
        for l in note_lines:
            if not l.strip():
                continue
            m = TEXT_PARA_DEF.match(l)
            if m:
                if m.group(1) != cid:
                    st["other"].append(f"{cid}: definition of a note of chapter {m.group(1)}: {l[:40]}")
                last = m.group(2)
                notes[last] = md_plain(l[m.end():])
            elif last:
                if not l.startswith("    "):
                    st["other"].append(f"{cid}: unindented line in the notes, joined to note {last}: {l[:40]}")
                notes[last] += " " + md_plain(l)
            else:
                st["other"].append(f"{cid}: text before the first note definition: {l[:40]}")
        st["notes"] += len(notes)

        # -- body -> segments (heading path, units); a unit = one [¶N] paragraph + attached blocks
        def calls_in(s: str) -> list[str]:
            out = []
            for m in TEXT_PARA_CALL.finditer(s):
                if m.group(1) != cid:
                    st["other"].append(f"{cid}: call of a note of chapter {m.group(1)}")
                out.append(m.group(2))
            return out

        nonblank = [l.strip() for l in main if l.strip()]
        segs: list[dict] = []
        stack: list[tuple[int, str]] = []
        title_seen = False
        pending: list[str] = []          # calls waiting for the next numbered paragraph (title, headings)
        fwd: list[tuple[str, list[str]]] = []   # blocks waiting for the next paragraph of this segment
        last_para = None                 # last numbered unit of the chapter (for figure calls)
        all_calls: list[str] = []
        nums: list[int] = []

        def new_seg() -> None:
            if segs and fwd:
                segs[-1]["units"].append(dict(nums=[], lines=[t for t, _ in fwd], calls=[c for _, cs in fwd for c in cs]))
                st["unparagraphed"] += 1
            fwd.clear()
            segs.append(dict(path=[t for lv, t in stack if lv > 1], units=[]))

        new_seg()
        for i, s in enumerate(nonblank):
            h = TEXT_PARA_HEAD.match(s)
            if h:
                level, calls = len(h.group(1)), calls_in(s)
                all_calls += calls
                pending += calls
                if level == 1 and not title_seen:
                    title_seen = True
                    st["via_title"] += len(calls)
                    continue
                st["via_heading"] += len(calls)
                stack = [x for x in stack if x[0] < level] + [(level, md_plain(h.group(2)))]
                new_seg()
                continue
            if s == TEXT_PARA_SEP:
                new_seg()
                continue
            seg = segs[-1]
            f = TEXT_PARA_FIGURE.match(s)
            if f:
                calls = calls_in(f.group(1))
                all_calls += calls
                st["via_figure"] += len(calls)
                if last_para is not None:
                    last_para["calls"] += calls
                else:
                    pending += calls
                title = figure_title(f.group(1))
                if not title:
                    continue
                s, calls = title, []
            m = TEXT_PARA_NUM.match(s)
            if m and kind == "body":
                calls = calls_in(s)
                all_calls += calls
                unit = dict(nums=[int(m.group(1))], lines=[t for t, _ in fwd] + [md_plain(s[m.end():])],
                            calls=[c for _, cs in fwd for c in cs] + pending + calls)
                fwd.clear()
                pending = []
                seg["units"].append(unit)
                last_para = unit
                nums.append(int(m.group(1)))
                st["paragraphs"] += 1
                continue
            if not f:
                calls = calls_in(s)
                all_calls += calls
                s = md_plain(TEXT_PARA_NUM.sub("", re.sub(r"^>\s?", "", s)))
                if not s:
                    continue
            if kind != "body":                       # bibliography: every block is an entry
                seg["units"].append(dict(nums=[], lines=[s], calls=pending + calls))
                pending = []
                st["entries"] += 1
                continue
            nxt = nonblank[i + 1] if i + 1 < len(nonblank) else ""
            leadin = (not f and not calls and len(s) <= TEXT_PARA_LEADIN_CHARS and s[-1] not in ".;:!?…"
                      and not re.match(r"(?:-|\d+\.) ", s) and TEXT_PARA_NUM.match(nxt))
            if leadin:
                st["leadins"].append(f"{cid}:{s[:30]}")
            if seg["units"] and not leadin:
                seg["units"][-1]["lines"].append(s)
                seg["units"][-1]["calls"] += calls
            else:
                fwd.append((s, calls))
        new_seg()

        # -- checks (reported, never fixed)
        if kind == "body" and nums != list(range(1, len(nums) + 1)):
            breaks = [f"{a}->{b}" for a, b in zip([0] + nums, nums) if b != a + 1]
            st["gaps"].append(f"{cid}: {breaks[:8]}")
        dup = sorted({c for c in all_calls if all_calls.count(c) > 1}, key=int)
        no_def = sorted(set(all_calls) - notes.keys(), key=int)
        no_call = sorted(notes.keys() - set(all_calls), key=int)
        if dup or no_def or no_call:
            st["footnotes"].append(f"{cid}: {len(all_calls)} calls, {len(notes)} definitions; "
                                   f"called but undefined {no_def}, defined but never called {no_call}, "
                                   f"called more than once {dup}")

        # -- pack into chunks
        pieces, rep = [], 0
        emitted: set[str] = set()
        meta0 = dict(chapter_id=cid, chapter_label=label, part=part)

        def notes_chunks(ids: list[str], a: int, b: int, section: str) -> None:
            nonlocal rep
            ids = [n for n in dict.fromkeys(ids) if n in notes and n not in emitted]
            emitted.update(ids)
            loc = para_location(label, a, b, " (notes)")
            entries: list[tuple[str, int, str]] = []
            for n in ids:
                line = f"Note {n}: {notes[n]}"
                rep += nonws(f"Note {n}:")
                if ntok(line) <= CHUNK_TOKENS:
                    entries.append((line, ntok(line), n))
                    continue
                parts, r = self._para_pack_split([notes[n]], CHUNK_TOKENS - 8, samples)
                rep += r + sum(nonws(f"Note {n} (suite):") for _ in parts[1:])
                entries += [(f"Note {n}{' (suite)' if j else ''}: {p}", ntok(p) + 8, n) for j, p in enumerate(parts)]
            cur: list[tuple[str, int, str]] = []
            for e in entries + [None]:
                if cur and (e is None or sum(x[1] for x in cur) + e[1] > CHUNK_TOKENS):
                    pieces.append(("\n".join(x[0] for x in cur), loc, "", dict(
                        meta0, chunk_kind="notes", section=section, para_start=a, para_end=b,
                        footnote_ids=",".join(dict.fromkeys(x[2] for x in cur)))))
                    st["chunks"]["notes"] += 1
                    cur = []
                if e is not None:
                    cur.append(e)

        def emit(units: list[dict], texts: list[str], prefix: str, section: str) -> None:
            nonlocal rep
            ns = [n for u in units for n in u["nums"]]
            a, b = (min(ns), max(ns)) if ns else (0, 0)
            calls = list(dict.fromkeys(c for u in units for c in u["calls"]))
            ck = "body" if kind == "body" else "bibliography"
            loc = para_location(label, a, b) if ck == "body" else label
            for t in texts:
                pieces.append((f"{prefix}\n{t}" if prefix else t, loc, "", dict(
                    meta0, chunk_kind=ck, section=section, para_start=a, para_end=b, footnote_ids=",".join(calls))))
                rep += nonws(prefix)
                st["chunks"][ck] += 1
            if calls:
                notes_chunks(calls, a, b, section)

        for seg in segs:
            if not seg["units"]:
                continue
            prefix = seg["path"][-1] if seg["path"] else ""
            section = " > ".join(seg["path"])
            budget = CHUNK_TOKENS - (ntok(prefix) + 1 if prefix else 0)
            group: list[dict] = []
            total = 0
            for u in seg["units"] + [None]:
                n = ntok("\n".join(u["lines"])) if u else 0
                if group and (u is None or n > budget or total + n + 1 > budget):
                    emit(group, ["\n\n".join("\n".join(x["lines"]) for x in group)], prefix, section)
                    group, total = [], 0
                if u is None:
                    break
                if n > budget:
                    texts, r = self._para_pack_split(u["lines"], budget, samples)
                    rep += r
                    st["split"] += bool(u["nums"])
                    emit([u], texts, prefix, section)
                    continue
                group.append(u)
                total += n + (1 if len(group) > 1 else 0)
        leftover = [n for n in notes if n not in emitted]
        if leftover:
            st["other"].append(f"{cid}: notes in no paragraph's chunk, emitted on their own: {leftover}")
            notes_chunks(leftover, 0, 0, "")
        return pieces, rep

    def page_flags(self, att_key: str, doc: pymupdf.Document) -> tuple[Counter, set[int], dict[int, set[str]]]:
        """Per page: count of my text/comment/tag annotations, pages with ink, and my annotation tags."""
        counts: Counter = Counter()
        ink: set[int] = set()
        tags: dict[int, set[str]] = defaultdict(set)
        for ann in self.lib.annotations.get(att_key, []):
            p = ann_page(ann)
            tags[p].update(tag_list(ann))
            if ann.get("annotationType") == "ink":
                ink.add(p)
            elif has_content(ann):
                counts[p] += 1
        for r in self.adobe_rows(att_key):
            counts[int(r["page_number"])] += 1
        for page in doc:
            if any(True for _ in page.annots(types=[pymupdf.PDF_ANNOT_INK])):
                ink.add(page.number + 1)
        return counts, ink, tags

    def pdf_source(self, info: ItemInfo, a: dict) -> Source | None:
        path, is_ocr = self.pdf_path(a)
        if path is None:
            log.warning("%s: PDF file not found; skipped", a["key"])
            return None
        anns = self.lib.annotations.get(a["key"], [])
        ann_fp = sorted((ann_page(x), x.get("annotationType"), x["key"], has_content(x), tag_list(x)) for x in anns)
        adobe_fp = sorted((r["page_number"], r["modified_date"]) for r in self.lib.adobe.get(a["key"], []))

        def build() -> list[Chunk]:
            labels, _how = self.printed(info, a)
            base = dict(info.base_meta(), source_type="pdf_text", attachment_key=a["key"], ocr=is_ocr)
            chunks = []
            with pymupdf.open(path) as doc:
                counts, ink, tags = self.page_flags(a["key"], doc)
                for page in doc:
                    pno = page.number + 1
                    blocks = [b[4] for b in page.get_text("blocks") if b[6] == 0]
                    text = clean_text("\n\n".join(blocks))
                    if not enough_text(text):
                        continue
                    m = dict(base, pdf_page=pno, printed_page=labels[page.number] if labels else "",
                             my_annotations_on_page=counts.get(pno, 0), has_ink_on_page=pno in ink,
                             page_annotation_tags="; ".join(sorted(tags.get(pno, ()), key=str.lower)))
                    for i, t in enumerate(self.chunker.split(text), 1):
                        chunks.append(Chunk(f"{a['key']}:p{pno}:c{i}", t, dict(m)))
            return chunks
        return Source(f"pdf:{a['key']}", info.key, file_fp(path),
                      fp_hash(info.fp(), ann_fp, adobe_fp), build)

    def annotation_sources(self, info: ItemInfo, a: dict, is_pdf: bool) -> list[Source]:
        out = []
        path, is_ocr = self.pdf_path(a) if is_pdf else (None, False)
        pdf_fp = file_fp(path) if path else ""
        for ann in self.lib.annotations.get(a["key"], []):
            atype = ann.get("annotationType", "")
            if atype in ("ink", "image"):
                continue          # no text; ink is recorded as has_ink_on_page on the PDF page

            tags = tag_list(ann)

            def build(ann=ann, atype=atype, tags=tags) -> list[Chunk]:
                if not has_content(ann):
                    return []          # neither text, comment nor tags
                stype = "zotero_highlight" if atype in ("highlight", "underline") else "zotero_note"
                m = dict(info.base_meta(), source_type=stype, attachment_key=a["key"], ocr=is_ocr,
                         annotation_tags="; ".join(tags))
                if is_pdf:
                    p = ann_page(ann)
                    labels, _ = self.printed(info, a)
                    m.update(pdf_page=p, printed_page=labels[p - 1] if 0 < p <= len(labels) else "")
                highlight, comment = ann.get("annotationText") or "", ann.get("annotationComment") or ""
                text = annotation_text(highlight, comment, tags, page_ref(m["printed_page"], m["pdf_page"]))
                return [Chunk(ann["key"], text, m, detect_lang=bool(highlight.strip() or comment.strip()))]
            out.append(Source(f"ann:{ann['key']}", info.key,
                              fp_hash(ann.get("dateModified"), tags, CHUNKER_VERSION),
                              fp_hash(info.fp(), pdf_fp), build))
        return out

    def adobe_sources(self, info: ItemInfo, a: dict) -> list[Source]:
        out = []
        path, is_ocr = self.pdf_path(a)
        pdf_fp = file_fp(path) if path else ""
        seen: set[str] = set()
        for r in self.adobe_rows(a["key"]):
            text = annotation_text(r["highlighted_text"], r["comment"])
            cid = f"{a['key']}:adobe:{fp_hash(r['page_number'], r['highlighted_text'], r['comment'])[:10]}"
            if not text or cid in seen:        # identical annotation twice on one page: index once
                continue
            seen.add(cid)

            def build(r=r, text=text, cid=cid) -> list[Chunk]:
                p = int(r["page_number"])
                labels, _ = self.printed(info, a)
                m = dict(info.base_meta(), source_type="adobe_annotation", attachment_key=a["key"], ocr=is_ocr,
                         pdf_page=p, printed_page=labels[p - 1] if 0 < p <= len(labels) else "")
                return [Chunk(cid, text, m)]
            out.append(Source(f"adobe:{cid}", info.key, fp_hash(r["modified_date"], text, CHUNKER_VERSION),
                              fp_hash(info.fp(), pdf_fp), build))
        return out


# ------------------------------------------------------------- embeddings

class Embedder:
    def __init__(self, device: str | None = None) -> None:
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu" and device is None:
            self.warn("CUDA is not available to PyTorch")
        self.model = SentenceTransformer(MODEL_NAME, device=self.device)
        self.model.max_seq_length = MAX_SEQ_LENGTH
        self.batch_size = 8
        if self.device == "cuda":
            try:
                self.encode(["GPU smoke test.", "Albertino Mussato, Ecerinis."])
                self.batch_size = self.calibrate()
            except Exception as e:  # noqa: BLE001
                self.fallback(e)

    def warn(self, reason: str) -> None:
        msg = f"WARNING: GPU not used ({reason}). Embedding on CPU, which is much slower."
        print("\n" + "!" * len(msg) + "\n" + msg + "\n" + "!" * len(msg), file=sys.stderr)
        log.warning(msg)

    def fallback(self, err: Exception) -> None:
        self.warn(f"{type(err).__name__}: {err}")
        self.device = "cpu"
        self.model.to("cpu")
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        self.batch_size = 8

    def encode(self, texts: list[str], batch_size: int | None = None):
        return self.model.encode(texts, batch_size=batch_size or self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)

    def calibrate(self) -> int:
        """Largest batch of full 512-token inputs that fits in free VRAM (no spill to shared RAM)."""
        torch = self.torch
        free, _total = torch.cuda.mem_get_info()
        budget = torch.cuda.memory_allocated() + free - 150 * 2**20
        text = "Albertino Mussato " * 400                     # truncated to 512 tokens
        best = 1
        for bs in (1, 2, 4, 8, 16, 32, 64):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                self.encode([text] * bs, batch_size=bs)
            except torch.cuda.OutOfMemoryError:
                break
            peak = torch.cuda.max_memory_reserved()
            if peak > budget:
                break
            best = bs
        torch.cuda.empty_cache()
        log.info("GPU batch size: %d (at %d tokens)", best, MAX_SEQ_LENGTH)
        return best

    def embed(self, texts: list[str]):
        while True:
            try:
                return self.encode(texts)
            except self.torch.cuda.OutOfMemoryError:
                self.torch.cuda.empty_cache()
                if self.batch_size == 1:
                    raise
                self.batch_size //= 2
                log.warning("CUDA out of memory; batch size reduced to %d", self.batch_size)
            except RuntimeError as e:
                if self.device != "cuda":
                    raise
                self.fallback(e)


# ----------------------------------------------------------------- storage

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    rowid INTEGER PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    source_id TEXT NOT NULL,
    item_key TEXT, attachment_key TEXT, source_type TEXT,
    pdf_page INTEGER, printed_page TEXT, language TEXT,
    text TEXT NOT NULL,
    meta TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source_id);
CREATE INDEX IF NOT EXISTS chunks_item ON chunks(item_key);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, tokenize = "unicode61 remove_diacritics 2"
);
CREATE TABLE IF NOT EXISTS items (
    item_key TEXT PRIMARY KEY,
    item_type TEXT, title TEXT, authors TEXT, editors TEXT, year TEXT, date TEXT,
    publication TEXT, volume TEXT, issue TEXT, pages TEXT, publisher TEXT, place TEXT,
    doi TEXT, isbn TEXT, issn TEXT, url TEXT, language TEXT, abstract TEXT, tags TEXT,
    collections TEXT, attachments TEXT, note_count INTEGER,
    date_added TEXT, date_modified TEXT, data_json TEXT
);
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    item_key TEXT, content_fp TEXT, meta_fp TEXT, chunk_ids TEXT, indexed_at TEXT
);
"""


class Store:
    def __init__(self, rebuild: bool) -> None:
        import chromadb
        from chromadb.config import Settings
        DB_DIR.mkdir(exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(DB_DIR), settings=Settings(anonymized_telemetry=False))
        self.db = sqlite3.connect(KEYWORD_DB)
        if rebuild:
            if COLLECTION_NAME in [c.name for c in self.client.list_collections()]:
                self.client.delete_collection(COLLECTION_NAME)
            self.db.executescript("DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS chunks_fts; "
                                  "DROP TABLE IF EXISTS items; DROP TABLE IF EXISTS sources;")
        self.col = self.client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine", "embedding_model": MODEL_NAME})
        self.db.executescript(SCHEMA)

    def known(self) -> dict[str, tuple[str, str, list[str]]]:
        return {sid: (c, m, json.loads(ids)) for sid, c, m, ids in
                self.db.execute("SELECT source_id, content_fp, meta_fp, chunk_ids FROM sources")}

    def _delete_sql(self, source_id: str) -> None:
        rows = [r[0] for r in self.db.execute("SELECT rowid FROM chunks WHERE source_id=?", (source_id,))]
        self.db.executemany("DELETE FROM chunks_fts WHERE rowid=?", [(r,) for r in rows])
        self.db.execute("DELETE FROM chunks WHERE source_id=?", (source_id,))

    def delete_source(self, source_id: str) -> None:
        self.col.delete(where={"source_id": source_id})
        self._delete_sql(source_id)
        self.db.execute("DELETE FROM sources WHERE source_id=?", (source_id,))

    def write(self, sources: list[Source], embeddings) -> None:
        """Replace these sources' chunks in both stores, then record them as indexed (one commit)."""
        chunks = [c for s in sources for c in s.chunks]
        for s in sources:
            self.col.delete(where={"source_id": s.id})
            self._delete_sql(s.id)
        if chunks:
            self.col.add(ids=[c.id for c in chunks], embeddings=embeddings,
                         documents=[c.text for c in chunks], metadatas=[c.meta for c in chunks])
        for c in chunks:
            self._insert_sql(c)
        self._record(sources)
        self.db.commit()

    def update_meta(self, sources: list[Source]) -> None:
        chunks = [c for s in sources for c in s.chunks]
        if chunks:
            self.col.update(ids=[c.id for c in chunks], metadatas=[c.meta for c in chunks])
        for c in chunks:
            self.db.execute("UPDATE chunks SET meta=?, printed_page=?, language=? WHERE chunk_id=?",
                            (json.dumps(c.meta, ensure_ascii=False), c.meta["printed_page"],
                             c.meta["language"], c.id))
        self._record(sources)
        self.db.commit()

    def _insert_sql(self, c: Chunk) -> None:
        cur = self.db.execute(
            "INSERT INTO chunks (chunk_id, source_id, item_key, attachment_key, source_type, pdf_page, "
            "printed_page, language, text, meta) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (c.id, c.meta["source_id"], c.meta["item_key"], c.meta["attachment_key"], c.meta["source_type"],
             c.meta["pdf_page"], c.meta["printed_page"], c.meta["language"], c.text,
             json.dumps(c.meta, ensure_ascii=False)))
        self.db.execute("INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)", (cur.lastrowid, c.text))

    def _record(self, sources: list[Source]) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.db.executemany(
            "INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?)",
            [(s.id, s.item_key, s.content_fp, s.meta_fp, json.dumps([c.id for c in s.chunks]), now)
             for s in sources])

    def write_items(self, lib: Library, sweep: bool) -> None:
        rows = []
        for info in lib.items.values():
            d = info.data
            m = info.base_meta()
            rows.append((
                info.key, d.get("itemType"), d.get("title", ""), m["authors"],
                "; ".join(names(d.get("creators", []), {"editor", "seriesEditor"})), m["year"], d.get("date", ""),
                m["publication"], d.get("volume", ""), d.get("issue", ""), d.get("pages", ""),
                d.get("publisher", ""), d.get("place", ""), d.get("DOI", ""), d.get("ISBN", ""),
                d.get("ISSN", ""), d.get("url", ""), d.get("language", ""), d.get("abstractNote", ""), m["tags"],
                m["collections"],
                json.dumps([{"key": a["key"], "contentType": a["data"].get("contentType", ""),
                             "linkMode": a["data"].get("linkMode", ""), "title": a["data"].get("title", "")}
                            for a in info.attachments], ensure_ascii=False),
                len(info.notes), d.get("dateAdded", ""), d.get("dateModified", ""),
                json.dumps(d, ensure_ascii=False)))
        self.db.executemany(f"INSERT OR REPLACE INTO items VALUES ({','.join('?' * 26)})", rows)
        if sweep:
            keep = set(lib.items)
            gone = [k for (k,) in self.db.execute("SELECT item_key FROM items") if k not in keep]
            self.db.executemany("DELETE FROM items WHERE item_key=?", [(k,) for k in gone])
        self.db.commit()


# -------------------------------------------------------------------- run

def extract(sources: list[Source], builder: Builder, desc: str) -> tuple[list[Source], list[str]]:
    ok, failed = [], []
    for s in tqdm(sources, desc=desc, unit="source"):
        try:
            s.chunks = s.build()
            for c in s.chunks:
                c.meta["source_id"] = s.id
            builder.set_languages(s.chunks)          # on the chunk's own text, before the header
            for c in s.chunks:
                c.text = f"{builder.header(c)}\n{c.text}"
            ok.append(s)
        except Exception:  # noqa: BLE001
            log.exception("Extraction failed for %s", s.id)
            failed.append(s.id)
    return ok, failed


def print_text_report(r: dict) -> None:
    ok = abs(r["diff_pct"]) <= 1
    print(f"\nText file {r['attachment_key']} (item {r['item_key']}) {r['file']}\n"
          f"    {TEXT_PATHS[r['format']]}, {r['chunks']} chunks\n"
          f"    coverage: {r['chunk_chars']} non-space chars in chunks vs {r['expected_chars']} expected "
          f"({r['diff_pct']:+.2f}%) {'OK' if ok else 'MISMATCH (over 1%)'}")
    for k, v in r["details"].items():
        print(f"    {k.replace('_', ' ')}: {v}")
    for p in r["skipped"]:
        print(f"    skipped p. {p['label'] or '(none)'} (#{p['scan']}): {p['reason']}; {p['chars']} chars"
              + (f", vocab score {p['score']}" if p["score"] is not None else "")
              + (f", lingua {p['lingua']}" if p.get("lingua") else ""))
    for p in r["review"]:
        print(f"    kept, score {GIBBERISH_SCORE}-{GIBBERISH_REVIEW}: p. {p['label']} (#{p['scan']}) "
              f"{p['chars']} chars, vocab score {p['score']}")


def run(args: argparse.Namespace) -> None:
    item_filter = set(args.items) if args.items else None
    lib = Library(item_filter)
    chunker = Chunker()
    builder = Builder(lib, chunker)
    sources = builder.sources()
    log.info("Sources in scope: %d", len(sources))

    if args.dry_run:
        done, failed = extract(sources, builder, "Extracting (dry run)")
        counts = Counter(c.meta["source_type"] for s in done for c in s.chunks)
        print("\nChunks by source_type:", dict(counts.most_common()), " total:", sum(counts.values()))
        print("Extraction failures:", len(failed))
        hows = Counter(re.sub(r"pages field [^:(]+", "pages field", how).strip()
                       for _, how in builder.page_info_cache.values())
        print("Printed pages, by PDF:", dict(hows.most_common()))
        print("Ignored XML attachments:", builder.ignored_xml)
        for r in builder.text_report.values():
            print_text_report(r)
        return

    store = Store(args.rebuild)
    known = store.known()
    new = [s for s in sources if args.force or s.id not in known or known[s.id][0] != s.content_fp]
    meta_only = [s for s in sources if s.id in known and known[s.id][0] == s.content_fp
                 and known[s.id][1] != s.meta_fp]
    current = {s.id for s in sources}
    gone = [sid for sid in known if sid not in current] if item_filter is None else []
    print(f"Sources: {len(sources)} in scope | {len(new)} new/changed | {len(meta_only)} metadata-only | "
          f"{len(gone)} deleted | {len(sources) - len(new) - len(meta_only)} unchanged")

    for sid in gone:
        store.delete_source(sid)
    store.db.commit()
    store.write_items(lib, sweep=item_filter is None)

    failed: list[str] = []
    if meta_only:
        done, f = extract(meta_only, builder, "Re-reading metadata")
        failed += f
        for i in range(0, len(done), 200):
            store.update_meta(done[i:i + 200])

    if new:
        todo, f = extract(new, builder, "Extracting text")
        failed += f
        total = sum(len(s.chunks) for s in todo)
        embedder = Embedder() if total else None
        if embedder:
            print(f"Embedding {total} chunks on {embedder.device.upper()} (batch size {embedder.batch_size})")
        bar = tqdm(total=total, desc="Embedding", unit="chunk", smoothing=0.05)
        batch: list[Source] = []

        def flush() -> None:
            texts = [c.text for s in batch for c in s.chunks]
            try:
                emb = embedder.embed(texts) if texts else []
                store.write(batch, emb.tolist() if len(texts) else [])
            except Exception:  # noqa: BLE001
                log.exception("Embedding/writing failed for %d sources starting %s", len(batch), batch[0].id)
                failed.extend(s.id for s in batch)
                store.db.rollback()
            bar.update(len(texts))
            batch.clear()

        for s in todo:
            batch.append(s)
            if sum(len(x.chunks) for x in batch) >= FLUSH_CHUNKS:
                flush()
        if batch:
            flush()
        bar.close()

    n_chunks = store.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"\nIndex now holds {n_chunks} chunks (Chroma: {store.col.count()}).")
    if failed:
        print(f"{len(failed)} sources failed; see {LOG_FILE}. They will be retried on the next run.")


def benchmark(n: int) -> None:
    """Embed the same n indexed chunks on GPU and on CPU and report the speed of each."""
    db = sqlite3.connect(KEYWORD_DB)
    texts = [t for (t,) in db.execute("SELECT text FROM chunks WHERE source_type='pdf_text' "
                                      "ORDER BY length(text) DESC LIMIT ?", (n,))]
    if not texts:
        sys.exit("No indexed pdf_text chunks to benchmark with.")
    chunker = Chunker()
    toks = [chunker.ntokens(t) for t in texts]
    print(f"Benchmark: {len(texts)} PDF chunks, mean {statistics.mean(toks):.0f} tokens")
    for device in ("cuda", "cpu"):
        emb = Embedder(device=device)
        if emb.device != device:
            continue
        emb.encode(texts[:2])                       # warm-up
        t0 = time.perf_counter()
        emb.embed(texts)
        dt = time.perf_counter() - t0
        print(f"  {device.upper():4}: {len(texts) / dt:6.2f} chunks/s  ({dt:.1f} s, batch size {emb.batch_size})")
        del emb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="delete the index and start from scratch")
    parser.add_argument("--items", nargs="+", metavar="KEY", help="only these Zotero item keys")
    parser.add_argument("--force", action="store_true",
                        help="with --items: re-index those items even if unchanged (e.g. after a code fix)")
    parser.add_argument("--dry-run", action="store_true", help="extract and count chunks; write nothing")
    parser.add_argument("--benchmark", type=int, nargs="?", const=48, metavar="N",
                        help="embedding speed on GPU vs CPU with N indexed chunks (default 48)")
    args = parser.parse_args()
    if args.force and not args.items:
        parser.error("--force needs --items (use --rebuild to redo everything)")

    DB_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8")])
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    log.addHandler(console)
    for noisy in ("httpx", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.benchmark:
        benchmark(args.benchmark)
    else:
        run(args)


if __name__ == "__main__":
    main()
