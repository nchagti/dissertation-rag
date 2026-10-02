"""Read-only extraction of annotations embedded in PDF files (step 2 of the RAG).

Annotations made in other PDF readers (mainly Adobe Acrobat) live inside the
PDF files themselves rather than in the Zotero database. This script reads
them from every attachment in audit.csv whose file_exists is "yes".

Each PDF is read into memory and opened from those bytes, so PyMuPDF never
holds a writable handle on the file. Nothing is ever saved or modified.

Usage:
    python extract_embedded_annotations.py              # full run
    python extract_embedded_annotations.py --limit 5 --out-stem embedded_annotations_test
    python extract_embedded_annotations.py --keys CXLSJM6P,QDCGJ3X6 --out-stem embedded_annotations_test
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import pymupdf

PROJECT_DIR = Path(r"C:\Users\nchag\Desktop\dissertation-rag")
AUDIT_CSV = PROJECT_DIR / "audit.csv"

SKIP_TYPES = {"Popup", "Link", "Widget"}
TEXT_MARKUP_TYPES = {"Highlight", "Underline", "StrikeOut", "Squiggly"}
MIN_WORD_OVERLAP = 0.5     # fraction of a word's box that must lie inside a quad
MIN_LINE_NUMBERS = 5       # aligned counting numbers needed to treat them as margin line numbers

COLUMNS = [
    "item_key", "attachment_key", "author", "year", "title",
    "page_number", "page_label", "annotation_type",
    "highlighted_text", "comment", "modified_date",
]


# ---------------------------------------------------------------- text cleaning

HYPHEN_BREAK = re.compile(r"[-\u00ad¬]\s*\n\s*")   # hyphen/soft hyphen/¬ at a line break
NOT_SIGN = re.compile(r"¬\s*")                      # ¬ is only ever a hyphenation mark


def clean_text(text: str) -> str:
    """Rejoin words hyphenated across line breaks and collapse whitespace."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = HYPHEN_BREAK.sub("", text)
    text = NOT_SIGN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


HEX_STRING = re.compile(r"<FEFF([0-9A-Fa-f]*)>")


def fix_label(label: str) -> str:
    """Decode label prefixes PyMuPDF returns as raw UTF-16 hex (e.g. <FEFF0043...>)."""
    def decode(m: re.Match) -> str:
        try:
            return bytes.fromhex(m.group(1)).decode("utf-16-be")
        except ValueError:
            return m.group(0)
    return HEX_STRING.sub(decode, label or "").replace("﻿", "").strip()


def pdf_date_to_iso(raw: str) -> str:
    """Convert a PDF date (D:YYYYMMDDHHmmSS+HH'mm') to ISO 8601; keep raw if unparseable."""
    m = re.match(r"^(?:D:)?(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?"
                 r"(?:([Zz+\-])(\d{2})?'?(\d{2})?'?)?", raw or "")
    if not m:
        return (raw or "").strip()
    year, mon, day, hh, mm, ss, tz, tzh, tzm = m.groups()
    iso = f"{year}-{mon or '01'}-{day or '01'}"
    if hh:
        iso += f"T{hh}:{mm or '00'}:{ss or '00'}"
        if tz in ("Z", "z"):
            iso += "Z"
        elif tz and tzh:
            iso += f"{tz}{tzh}:{tzm or '00'}"
    return iso


# ---------------------------------------------------------------- extraction

def find_line_numbers(words: list) -> set[int]:
    """Indexes of margin line numbers (as in line-numbered manuscripts).

    A line-number column is a set of digit-only words aligned on the same
    edge whose values mostly count up by one from top to bottom.
    """
    columns: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx, w in enumerate(words):
        if w[4].isdigit() and len(w[4]) <= 4:
            columns[("left", round(w[0] / 3))].append(idx)
            columns[("right", round(w[2] / 3))].append(idx)
    found: set[int] = set()
    for members in columns.values():
        if len(members) < MIN_LINE_NUMBERS:
            continue
        members.sort(key=lambda i: words[i][1])
        values = [int(words[i][4]) for i in members]
        steps = sum(1 for a, b in zip(values, values[1:]) if b == a + 1)
        if steps >= MIN_LINE_NUMBERS - 1 and steps >= 0.8 * (len(values) - 1):
            found.update(members)
    return found


def text_under_quads(annot: pymupdf.Annot, words: list, line_number_words: set[int]) -> str:
    """Words lying under the annotation's quads, in reading order.

    Quads keep the order the PDF stores them in (a highlight may run across
    columns), except that consecutive quads on the same visual line are
    sorted left to right, since readers sometimes store those out of order.
    Visual lines are joined with "\n" so clean_text can rejoin hyphenated
    words. Without quads, the annotation rect is used and words keep the
    PDF's own line order.
    """
    vertices = annot.vertices or []
    quads = [pymupdf.Quad(vertices[i:i + 4]).rect for i in range(0, len(vertices) - 3, 4)]
    has_quads = bool(quads)
    if not has_quads:
        quads = [annot.rect]

    # Group consecutive quads that share a visual line.
    quad_lines: list[list[pymupdf.Rect]] = []
    for q in quads:
        prev = quad_lines[-1][-1] if quad_lines else None
        overlap = min(q.y1, prev.y1) - max(q.y0, prev.y0) if prev else 0
        if prev and overlap > 0.5 * min(q.height, prev.height):
            quad_lines[-1].append(q)
        else:
            quad_lines.append([q])

    lines: list[list[int]] = []
    seen: set[int] = set()
    for qline in quad_lines:
        line: list[int] = []
        for qrect in qline:
            for idx, w in enumerate(words):
                if idx in seen:
                    continue
                wrect = pymupdf.Rect(w[:4])
                area = wrect.get_area()
                if area > 0 and (wrect & qrect).get_area() / area >= MIN_WORD_OVERLAP:
                    line.append(idx)
                    seen.add(idx)
        if has_quads:
            line.sort(key=lambda i: words[i][0])
            lines.append(line)
        else:
            for _, group in groupby(line, key=lambda i: (words[i][5], words[i][6])):
                lines.append(list(group))
    # Drop margin line numbers, unless they are all that was highlighted.
    if any(i not in line_number_words for line in lines for i in line):
        lines = [[i for i in line if i not in line_number_words] for line in lines]
    text = "\n".join(" ".join(words[i][4] for i in line) for line in lines if line)
    return clean_text(text)


def extract_pdf(path: str, meta: dict) -> list[dict]:
    """Return one row per annotation in the PDF (opened from bytes, read-only)."""
    data = Path(path).read_bytes()
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        if doc.needs_pass:
            raise RuntimeError("password-protected")
        has_labels = bool(doc.get_page_labels())
        rows = []
        for page in doc:
            annots = [a for a in page.annots() if a.type[1] not in SKIP_TYPES]
            if not annots:
                continue
            words = None
            line_number_words: set[int] = set()
            label = fix_label(page.get_label()) if has_labels else ""
            for annot in annots:
                atype = annot.type[1]
                highlighted = ""
                if atype in TEXT_MARKUP_TYPES:
                    if words is None:
                        words = page.get_text("words", sort=False)
                        line_number_words = find_line_numbers(words)
                    highlighted = text_under_quads(annot, words, line_number_words)
                info = annot.info
                rows.append({
                    **meta,
                    "page_number": page.number + 1,
                    "page_label": label,
                    "annotation_type": atype,
                    "highlighted_text": highlighted,
                    "comment": clean_text(info.get("content", "")),
                    "modified_date": pdf_date_to_iso(info.get("modDate", "")),
                    "_y": annot.rect.y0,
                    "_x": annot.rect.x0,
                })
        rows.sort(key=lambda r: (r["page_number"], r["_y"], r["_x"]))
        return rows
    finally:
        doc.close()


# ---------------------------------------------------------------- output

def write_csv(rows: list[dict], out: Path) -> None:
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def md_escape_quote(text: str) -> str:
    return text.replace("\n", " ")


def write_md(rows: list[dict], out: Path) -> None:
    by_item: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_item[r["item_key"]].append(r)

    def item_sort(key: str):
        r = by_item[key][0]
        return (r["author"].casefold(), r["year"], r["title"].casefold())

    lines = ["# Embedded PDF annotations", ""]
    for key in sorted(by_item, key=item_sort):
        item_rows = by_item[key]
        first = item_rows[0]
        heading = " ".join(p for p in [
            first["author"],
            f"({first['year']})" if first["year"] else "",
        ] if p)
        if first["title"]:
            heading = f"{heading}. *{first['title']}*" if heading else f"*{first['title']}*"
        lines += [f"## {heading or key}", ""]

        attachments = list(dict.fromkeys(r["attachment_key"] for r in item_rows))
        for att in attachments:
            if len(attachments) > 1:
                lines += [f"### Attachment {att}", ""]
            for r in (r for r in item_rows if r["attachment_key"] == att):
                page = f"p. {r['page_number']}"
                if r["page_label"] and r["page_label"] != str(r["page_number"]):
                    page += f" [{r['page_label']}]"
                lines.append(f"- **{page}** · {r['annotation_type']}")
                if r["highlighted_text"]:
                    lines.append(f"  > {md_escape_quote(r['highlighted_text'])}")
                if r["comment"]:
                    if r["highlighted_text"]:
                        lines.append("")
                    lines.append(f"  **Comment:** {r['comment']}")
                lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- main

def load_attachments(keys: set[str] | None, limit: int | None) -> list[dict]:
    with open(AUDIT_CSV, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f)
                if r["file_exists"] == "yes" and r["content_type"] == "PDF" and r["file_path"]]
    if keys:
        rows = [r for r in rows if r["attachment_key"] in keys or r["item_key"] in keys]
    if limit:
        rows = rows[:limit]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="only process the first N PDFs")
    ap.add_argument("--keys", help="comma-separated attachment or item keys to process")
    ap.add_argument("--out-stem", default="embedded_annotations",
                    help="output file name without extension (default: embedded_annotations)")
    args = ap.parse_args()

    keys = {k.strip() for k in args.keys.split(",")} if args.keys else None
    attachments = load_attachments(keys, args.limit)
    print(f"Processing {len(attachments)} PDF(s) from {AUDIT_CSV.name}...")

    all_rows: list[dict] = []
    pdfs_with_annots = 0
    errors: list[tuple[str, str]] = []
    for i, att in enumerate(attachments, 1):
        meta = {
            "item_key": att["item_key"],
            "attachment_key": att["attachment_key"],
            "author": att["first_author"],
            "year": att["year"],
            "title": att["title"],
        }
        try:
            rows = extract_pdf(att["file_path"], meta)
        except Exception as e:  # damaged, encrypted or unreadable PDF
            errors.append((att["attachment_key"], f"{type(e).__name__}: {e}"))
            continue
        if rows:
            pdfs_with_annots += 1
            all_rows.extend(rows)
        if i % 25 == 0:
            print(f"  {i}/{len(attachments)}")

    csv_out = PROJECT_DIR / f"{args.out_stem}.csv"
    md_out = PROJECT_DIR / f"{args.out_stem}.md"
    write_csv(all_rows, csv_out)
    write_md(all_rows, md_out)

    by_type = Counter(r["annotation_type"] for r in all_rows)
    with_comments = sum(1 for r in all_rows if r["comment"])
    print()
    print(f"PDFs processed:                 {len(attachments) - len(errors)}")
    print(f"PDFs with embedded annotations: {pdfs_with_annots}")
    print(f"Total annotations:              {len(all_rows)}")
    for atype, n in by_type.most_common():
        print(f"  {atype:<28}{n}")
    print(f"Annotations with comments:      {with_comments}")
    if errors:
        print(f"\nCould not read {len(errors)} PDF(s):")
        for key, msg in errors:
            print(f"  {key}: {msg}")
    print(f"\nWrote {csv_out.name} and {md_out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
