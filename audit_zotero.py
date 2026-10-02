"""Read-only audit of the Zotero dissertation collection (step 1 of the RAG).

Only issues GET requests to the Zotero local API and opens files read-only.
Nothing in Zotero (database, storage, settings) is modified.

Usage:
    python audit_zotero.py                 # full run -> audit.csv
    python audit_zotero.py --limit 10 --out audit_test.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf
import requests
from bs4 import BeautifulSoup
from lingua import Language, LanguageDetectorBuilder

API_BASE = "http://127.0.0.1:23119/api/users/0"
USER_AGENT = "dissertation-rag-audit/1.0"  # must not contain "Mozilla"
ROOT_COLLECTION = "UJ4JX5F6"
ZOTERO_DIR = Path(r"C:\Users\nchag\Zotero")
STORAGE_DIR = ZOTERO_DIR / "storage"
PROJECT_DIR = Path(r"C:\Users\nchag\Desktop\dissertation-rag")
EXPECTED_ITEMS = 389
PAGE_SIZE = 100

SAMPLE_PAGES = 5
MIN_PAGE_CHARS = 25        # non-whitespace chars for a page to count as having text
LANG_CHUNK_CHARS = 400     # text is split into chunks of about this size for language detection
MAX_LANG_CHARS = 30000     # cap on text fed to the language detector per attachment

COLUMNS = [
    "item_key", "attachment_key", "first_author", "year", "title", "item_type",
    "subcollections", "language_field", "content_type", "link_mode",
    "file_path", "file_exists", "page_count", "text_layer", "detected_languages",
    "annotation_count", "note_count", "has_abstract",
]


# ---------------------------------------------------------------- Zotero API

class ZoteroAPI:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.requests_made = 0

    def get(self, path: str, **params) -> requests.Response:
        self.requests_made += 1
        resp = self.session.get(f"{API_BASE}/{path}", params=params, timeout=30)
        resp.raise_for_status()
        return resp

    def check_connection(self) -> dict:
        try:
            resp = self.session.get(f"{API_BASE}/collections/{ROOT_COLLECTION}", timeout=10)
        except requests.exceptions.ConnectionError:
            sys.exit(
                "ERROR: Could not reach Zotero at http://127.0.0.1:23119.\n"
                "Please open Zotero and try again. (If it is already open, check that\n"
                "Settings > Advanced > 'Allow other applications on this computer to\n"
                "communicate with Zotero' is enabled.)"
            )
        if resp.status_code == 404:
            sys.exit(f"ERROR: Zotero is running, but collection {ROOT_COLLECTION} was not found.")
        if resp.status_code != 200:
            sys.exit(f"ERROR: Zotero local API returned HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def get_all(self, path: str, **params) -> list[dict]:
        """Paginate a listing endpoint with limit=100 until every result is retrieved."""
        results: list[dict] = []
        start = 0
        while True:
            resp = self.get(path, limit=PAGE_SIZE, start=start, **params)
            batch = resp.json()
            results.extend(batch)
            start += len(batch)
            total = resp.headers.get("Total-Results")
            if not batch or (start >= int(total) if total else len(batch) < PAGE_SIZE):
                break
        return results


# ------------------------------------------------------------ path handling

def find_base_attachment_path() -> Path | None:
    """Linked files may be stored relative to the 'Linked Attachment Base Directory'."""
    pattern = os.path.expandvars(r"%APPDATA%\Zotero\Zotero\Profiles\*\prefs.js")
    for prefs in glob.glob(pattern):
        text = Path(prefs).read_text(encoding="utf-8", errors="replace")
        m = re.search(r'user_pref\("extensions\.zotero\.baseAttachmentPath",\s*"(.*?)"\);', text)
        if m:
            return Path(m.group(1).encode().decode("unicode_escape"))
    return None


BASE_ATTACHMENT_PATH = find_base_attachment_path()


def resolve_path(att: dict) -> Path | None:
    data = att["data"]
    raw = data.get("path") or ""
    mode = data.get("linkMode", "")
    if mode in ("imported_file", "imported_url", "embedded_image"):
        if raw.startswith("storage:"):
            return STORAGE_DIR / data["key"] / raw[len("storage:"):]
        filename = data.get("filename")
        return STORAGE_DIR / data["key"] / filename if filename else None
    if mode == "linked_file" and raw:
        if raw.startswith("attachments:"):
            if BASE_ATTACHMENT_PATH is None:
                return Path(raw)  # unresolvable; will be flagged as missing
            return BASE_ATTACHMENT_PATH / raw[len("attachments:"):]
        return Path(raw)
    return None  # linked_url has no local file


def link_mode_label(mode: str) -> str:
    return {
        "imported_file": "imported",
        "imported_url": "imported",
        "linked_file": "linked",
        "linked_url": "linked (URL, no file)",
        "embedded_image": "embedded",
    }.get(mode, mode)


def content_type_label(att: dict) -> str:
    ct = (att["data"].get("contentType") or "").lower()
    if ct == "application/pdf":
        return "PDF"
    if ct in ("text/html", "application/xhtml+xml"):
        return "HTML snapshot"
    if ct == "application/epub+zip":
        return "EPUB"
    return f"other ({ct})" if ct else "other"


# ------------------------------------------------------------ file analysis

def sample_page_indices(n: int, k: int = SAMPLE_PAGES) -> list[int]:
    # Pages at 10%, 30%, 50%, 70%, 90%: spread out, and skips cover sheets on long docs.
    return sorted({min(n - 1, int(n * (i + 0.5) / k)) for i in range(k)})


def analyze_pdf(path: Path) -> tuple[str, str, str]:
    """Return (page_count, text_layer_status, sampled_text)."""
    try:
        with pymupdf.open(path) as doc:
            if doc.needs_pass or doc.is_encrypted:
                return str(doc.page_count), "error/encrypted", ""
            n = doc.page_count
            if n == 0:
                return "0", "error/encrypted", ""
            texts = [doc[i].get_text() for i in sample_page_indices(n)]
    except Exception:
        return "", "error/encrypted", ""
    with_text = sum(len(re.sub(r"\s", "", t)) >= MIN_PAGE_CHARS for t in texts)
    if with_text == len(texts):
        status = "text"
    elif with_text == 0:
        status = "none (needs OCR)"
    else:
        status = "mixed"
    return str(n), status, "\n".join(texts)


def html_text(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
        tag.decompose()
    return soup.get_text("\n")


LANGUAGES = [Language.ENGLISH, Language.FRENCH, Language.ITALIAN, Language.GERMAN, Language.LATIN]
_detector = None


def detect_languages(text: str) -> str:
    """Top two languages with rough proportions, weighted by chunk length."""
    global _detector
    text = re.sub(r"[ \t]+", " ", text)[:MAX_LANG_CHARS]
    chunks, buf = [], ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        buf = f"{buf} {line}" if buf else line
        if len(buf) >= LANG_CHUNK_CHARS:
            chunks.append(buf)
            buf = ""
    if len(buf) >= 40 or (buf and not chunks):
        chunks.append(buf)
    chunks = [c for c in chunks if len(re.findall(r"[^\W\d_]", c)) >= 20]
    if not chunks:
        return ""
    if _detector is None:
        _detector = LanguageDetectorBuilder.from_languages(*LANGUAGES).build()
    weights: Counter = Counter()
    for chunk, lang in zip(chunks, _detector.detect_languages_in_parallel_of(chunks)):
        if lang is not None:
            weights[lang.name.title()] += len(chunk)
    total = sum(weights.values())
    if not total:
        return ""
    return "; ".join(f"{name} {w / total:.0%}" for name, w in weights.most_common(2))


# -------------------------------------------------------------- item fields

def first_author(data: dict) -> str:
    creators = data.get("creators") or []
    if not creators:
        return ""
    c = next((c for c in creators if c.get("creatorType") == "author"), creators[0])
    if "name" in c:
        return c["name"]
    return ", ".join(p for p in (c.get("lastName", ""), c.get("firstName", "")) if p)


def year_of(item: dict) -> str:
    parsed = item.get("meta", {}).get("parsedDate", "")
    if parsed:
        return parsed[:4]
    m = re.search(r"\b(\d{4})\b", item["data"].get("date", "") or "")
    return m.group(1) if m else ""


def title_of(data: dict) -> str:
    if data.get("itemType") == "note":
        text = BeautifulSoup(data.get("note", ""), "html.parser").get_text("\n").strip()
        return text.splitlines()[0][:120] if text else "(empty note)"
    return data.get("title", "")


# ------------------------------------------------------------------- audit

def collect_collections(api: ZoteroAPI) -> dict[str, str]:
    """Map key -> path name for the root collection and all its descendants."""
    all_cols = api.get_all("collections")
    by_key = {c["key"]: c["data"] for c in all_cols}
    if ROOT_COLLECTION not in by_key:
        sys.exit(f"ERROR: collection {ROOT_COLLECTION} not found in collection listing.")
    children: dict[str, list[str]] = {}
    for key, data in by_key.items():
        parent = data.get("parentCollection")
        if parent:
            children.setdefault(parent, []).append(key)
    names: dict[str, str] = {}

    def walk(key: str, path: str) -> None:
        names[key] = path
        for child in sorted(children.get(key, []), key=lambda k: by_key[k]["name"].lower()):
            walk(child, f"{path} / {by_key[child]['name']}" if key != ROOT_COLLECTION
                 else by_key[child]["name"])

    walk(ROOT_COLLECTION, f"({by_key[ROOT_COLLECTION]['name']} root)")
    return names


def collect_annotations(api: ZoteroAPI) -> dict[str, list[dict]]:
    """All annotations in the library, grouped by parent attachment key.

    items/{key}/children does not return annotations on the local API, so they are
    fetched once via a filtered listing instead.
    """
    by_parent: dict[str, list[dict]] = {}
    for ann in api.get_all("items", itemType="annotation"):
        parent = ann["data"].get("parentItem")
        if parent:
            by_parent.setdefault(parent, []).append(ann)
    return by_parent


def attachment_row(base: dict, att: dict, annotations_by_parent: dict[str, list[dict]],
                   stats: Counter) -> dict:
    data = att["data"]
    row = dict(base)
    ctype = content_type_label(att)
    path = resolve_path(att)
    row.update(
        attachment_key=data["key"],
        content_type=ctype,
        link_mode=link_mode_label(data.get("linkMode", "")),
        file_path=str(path) if path else "",
    )
    exists = path is not None and path.is_file()
    row["file_exists"] = ("yes" if exists else "no") if path is not None else "n/a"
    if path is not None and not exists:
        stats["missing_files"] += 1

    annotations = annotations_by_parent.get(data["key"], [])
    row["annotation_count"] = len(annotations)
    stats["annotations"] += len(annotations)
    for ann in annotations:
        a = ann["data"]
        atype = a.get("annotationType", "")
        stats["ann_" + (atype if atype in ("highlight", "note") else "other")] += 1
        if atype not in ("highlight", "note"):
            stats[f"ann_other_{atype or 'unknown'}"] += 1
        if (a.get("annotationComment") or "").strip():
            stats["ann_with_comment"] += 1

    text = ""
    if ctype == "PDF":
        stats["pdfs"] += 1
        if exists:
            row["page_count"], row["text_layer"], text = analyze_pdf(path)
            stats[f"pdf_{row['text_layer']}"] += 1
        else:
            row["text_layer"] = "missing file"
    elif ctype == "HTML snapshot":
        stats["snapshots"] += 1
        if exists:
            text = html_text(path)
    elif ctype == "EPUB":
        stats["epubs"] += 1
    if text.strip():
        row["detected_languages"] = detect_languages(text)
    return row


def audit(limit: int | None) -> tuple[list[dict], Counter]:
    api = ZoteroAPI()
    root = api.check_connection()
    print(f"Connected to Zotero. Collection '{root['data']['name']}' found.")

    col_names = collect_collections(api)
    print(f"Collections: root + {len(col_names) - 1} subcollections (recursive).")

    items: dict[str, dict] = {}
    for col_key in col_names:
        for item in api.get_all(f"collections/{col_key}/items/top"):
            items.setdefault(item["key"], item)
    print(f"Top-level items found: {len(items)} (expected about {EXPECTED_ITEMS}).")
    if len(items) < EXPECTED_ITEMS * 0.9:
        print(f"WARNING: far fewer items than expected ({len(items)} vs ~{EXPECTED_ITEMS}). "
              "Pagination or collection traversal may be wrong.")

    item_list = list(items.values())
    if limit:
        item_list = item_list[:limit]
        print(f"TEST MODE: processing first {len(item_list)} items only.")

    annotations_by_parent = collect_annotations(api)
    library_annotations = sum(len(v) for v in annotations_by_parent.values())
    print(f"Annotations in whole library: {library_annotations} "
          f"(on {len(annotations_by_parent)} attachments).")

    rows: list[dict] = []
    stats: Counter = Counter()
    for i, item in enumerate(item_list, 1):
        data = item["data"]
        itype = data.get("itemType", "")
        subcols = [col_names[k] for k in data.get("collections", []) if k in col_names]
        base = {c: "" for c in COLUMNS}
        base.update(
            item_key=data["key"],
            first_author=first_author(data),
            year=year_of(item),
            title=title_of(data),
            item_type=itype,
            subcollections="; ".join(sorted(subcols)),
            language_field=data.get("language", ""),
            has_abstract="yes" if (data.get("abstractNote") or "").strip() else "no",
            content_type="none",
            annotation_count=0,
            note_count=0,
        )
        stats["items"] += 1

        if itype == "attachment":  # standalone attachment: it is its own attachment
            rows.append(attachment_row(base, item, annotations_by_parent, stats))
        else:
            children = (api.get_all(f"items/{data['key']}/children")
                        if item.get("meta", {}).get("numChildren", 1) else [])
            atts = [c for c in children if c["data"].get("itemType") == "attachment"]
            notes = [c for c in children if c["data"].get("itemType") == "note"]
            base["note_count"] = len(notes)
            stats["notes"] += len(notes)
            if itype == "note":
                stats["standalone_notes"] += 1
            if atts:
                rows.extend(attachment_row(base, a, annotations_by_parent, stats) for a in atts)
            else:
                stats["no_attachment"] += 1
                rows.append(base)
        if i % 25 == 0 or i == len(item_list):
            print(f"  processed {i}/{len(item_list)} items")
    stats["api_requests"] = api.requests_made
    stats["library_annotations"] = library_annotations
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--limit", type=int, help="only process the first N items (for testing)")
    parser.add_argument("--out", default=str(PROJECT_DIR / "audit.csv"), help="output CSV path")
    args = parser.parse_args()

    rows, s = audit(args.limit)
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_DIR / out
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out}")
    print("\n===== SUMMARY =====")
    print(f"Total items:              {s['items']}  (incl. {s['standalone_notes']} standalone notes)")
    print(f"Items with no attachment: {s['no_attachment']}")
    print(f"PDFs:                     {s['pdfs']}")
    print(f"  with text layer:        {s['pdf_text']}")
    print(f"  needing OCR:            {s['pdf_none (needs OCR)']}")
    print(f"  mixed:                  {s['pdf_mixed']}")
    print(f"  error/encrypted:        {s['pdf_error/encrypted']}")
    print(f"Missing files:            {s['missing_files']}")
    print(f"HTML snapshots:           {s['snapshots']}")
    print(f"EPUBs:                    {s['epubs']}")
    print(f"Total annotations:        {s['annotations']}  "
          f"(ignored {s['library_annotations'] - s['annotations']} outside this collection)")
    print(f"  highlights:             {s['ann_highlight']}")
    print(f"  notes:                  {s['ann_note']}")
    other = ", ".join(f"{k[len('ann_other_'):]} {v}" for k, v in sorted(s.items())
                      if k.startswith("ann_other_"))
    print(f"  other types:            {s['ann_other']}" + (f"  ({other})" if other else ""))
    print(f"  with a comment:         {s['ann_with_comment']}")
    print(f"Total notes (child):      {s['notes']}")
    print(f"(API GET requests made:   {s['api_requests']})")


if __name__ == "__main__":
    main()
