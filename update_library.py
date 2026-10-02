r"""Run the whole library pipeline: audit -> plan new OCR rows -> OCR -> ingest.

Zotero is only read (GET requests, files opened read-only); nothing in its storage
folder is written. Each file is handled on its own: a failure (error, crash, OCRmyPDF
non-zero exit, timeout) is recorded for that key and the run moves on to the next one.

Steps:
  1. preflight: Zotero reachable, Claude Desktop (incl. tray process) not running (not checked
     with --dry-run, which does not touch the index; re-checked before ingest.py)
  2. audit_zotero.py -> audit.csv
  3. new PDFs ("none (needs OCR)" / "mixed", key in neither ocr_plan.csv nor ocr_manifest.csv)
  4. "mixed" files: every page checked; only blank pages / plates without text -> "no OCR needed"
  5. languages of new rows: Tesseract (eng+fra+ita+deu+lat) on 3 sample pages + lingua, then confirm
  6. OCR every "OCR needed" plan row whose key is not in ocr_manifest.csv (so earlier failures are
     retried), one run_ocr.py process per file; a sidecar sample is shown after the first success
  7. ingest.py (incremental), also when some files failed
Failures are printed at the end and written to ocr_failures.csv (overwritten each run).

Usage:
    .venv\Scripts\python.exe update_library.py --dry-run   # steps 1-5 only; writes no plan/OCR/index
    .venv\Scripts\python.exe update_library.py             # full run, with confirmation prompts
    .venv\Scripts\python.exe update_library.py --yes       # full run, no prompts
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, deque
from pathlib import Path

import pymupdf
from lingua import Language, LanguageDetectorBuilder

from audit_zotero import MIN_PAGE_CHARS, STORAGE_DIR, ZoteroAPI
from run_ocr import MANIFEST, OUT_DIR, PLAN, ROOT, to_wsl

AUDIT_CSV = ROOT / "audit.csv"
FAILURES_CSV = ROOT / "ocr_failures.csv"
KEYWORD_DB = ROOT / "db" / "keyword.sqlite"
PLAN_FIELDS = ["item_key", "attachment_key", "author", "year", "title", "page_count", "text_layer",
               "textless_pages", "ocr_status", "notes", "languages", "file_path"]
FAILURE_FIELDS = ["attachment_key", "item_key", "author", "year", "title", "stage", "reason", "date"]
NEEDS_OCR, MIXED = "none (needs OCR)", "mixed"

LANGS = {Language.ENGLISH: "eng", Language.FRENCH: "fra", Language.ITALIAN: "ita",
         Language.GERMAN: "deu", Language.LATIN: "lat"}
ALL_TESS_LANGS = "eng+fra+ita+deu+lat"
SAMPLE_PAGES = 3
OCR_DPI = 300
TESSERACT_TIMEOUT = 300       # seconds per sample page
MIN_LANG_SHARE = 0.02         # a language must cover this share of the sample text to be used for OCR
BLANK_DARK_FRACTION = 0.002   # textless page with fewer dark pixels than this is blank
PLATE_MAX_WORDS = 20          # textless image page whose OCR yields fewer words is a plate, not text
SIDECAR_SAMPLE_CHARS = 1500

_detector = None


class StepFailed(Exception):
    """A whole pipeline step failed; the run stops."""


# ------------------------------------------------------------------ helpers

def say(msg: str = "") -> None:
    print(msg, flush=True)


def heading(title: str) -> None:
    say(f"\n{'=' * 8} {title} {'=' * max(4, 60 - len(title))}")


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        say("\n(no answer: input is not interactive; use --yes to skip prompts)")
        return "n"


def last_error_line(lines: list[str]) -> str:
    """The most informative last error line of a run_ocr.py run."""
    lines = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("Done:")]
    errs = [ln for ln in lines if re.search(r"error|failed|exception|traceback|not found|skipping", ln, re.I)]
    if not errs:
        return lines[-1] if lines else "no output"
    m = re.search(r"ocrmypdf failed with exit code (\d+)", errs[-1])
    if m and len(errs) > 1:
        return f"{errs[-2]} (ocrmypdf exit code {m.group(1)})"
    return errs[-1]


def assert_not_storage(path: Path) -> None:
    if STORAGE_DIR.resolve() in path.resolve().parents:
        raise RuntimeError(f"refusing to write inside Zotero storage: {path}")


# --------------------------------------------------------------- 1 preflight

def claude_desktop_processes() -> list[str]:
    """Claude Desktop processes (main window, tray, helpers); the Claude Code CLI does not count."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='claude.exe'\" | "
          "Select-Object ProcessId,ExecutablePath | ConvertTo-Json -Compress")
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise StepFailed(f"Could not list processes: {r.stderr.strip()[:300]}")
    procs = json.loads(r.stdout) if r.stdout.strip() else []
    if isinstance(procs, dict):
        procs = [procs]
    found = []
    for p in procs:
        exe = (p.get("ExecutablePath") or "").lower()
        if "\\.vscode\\extensions\\" in exe and "anthropic.claude-code" in exe:
            continue                              # Claude Code CLI, e.g. the one running this
        found.append(f"PID {p['ProcessId']}: {p.get('ExecutablePath') or '(path not readable)'}")
    return found


def preflight(dry_run: bool) -> None:
    heading("1. Preflight")
    root = ZoteroAPI().check_connection()        # exits with a clear message if Zotero is unreachable
    say(f"Zotero reachable at 127.0.0.1; collection '{root['data']['name']}' found.")
    if dry_run:
        say("Dry run: Claude Desktop check skipped (the index is not touched).")
        return
    desktop = claude_desktop_processes()
    if desktop:
        raise StepFailed(
            "Claude Desktop is running (it uses the index through the MCP server). Quit it completely,\n"
            "including the system tray icon (right-click > Quit), then run this again.\n  "
            + "\n  ".join(desktop[:5]) + (f"\n  ... and {len(desktop) - 5} more" if len(desktop) > 5 else ""))
    say("Claude Desktop is not running.")


# ------------------------------------------------------------------- 2 audit

def run_audit() -> None:
    heading("2. Audit (audit_zotero.py)")
    r = subprocess.run([sys.executable, str(ROOT / "audit_zotero.py")], cwd=ROOT)
    if r.returncode != 0:
        raise StepFailed(f"audit_zotero.py failed with exit code {r.returncode}")


# -------------------------------------------------------- 3-5 planning

def tesseract_page(page: pymupdf.Page, tmp: Path, tag: str) -> str:
    """OCR one rendered page with all five languages; the PDF itself is only read."""
    img = tmp / f"{tag}_p{page.number + 1}.png"
    page.get_pixmap(dpi=OCR_DPI, colorspace=pymupdf.csGRAY).save(img)
    try:
        r = subprocess.run(["wsl", "tesseract", to_wsl(img), "stdout", "-l", ALL_TESS_LANGS],
                           capture_output=True, timeout=TESSERACT_TIMEOUT)
    except subprocess.TimeoutExpired:
        subprocess.run(["wsl", "pkill", "-f", img.name], capture_output=True)
        raise RuntimeError(f"tesseract timed out after {TESSERACT_TIMEOUT}s on p.{page.number + 1}")
    finally:
        img.unlink(missing_ok=True)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(f"tesseract exit {r.returncode} on p.{page.number + 1}: {err[-1] if err else ''}")
    return r.stdout.decode("utf-8", "replace")


def words(text: str) -> int:
    return len(re.findall(r"[^\W\d_]{3,}", text))


def page_ranges(pages: list[int]) -> str:
    return ",".join(map(str, pages))


def classify_mixed(doc: pymupdf.Document, tmp: Path, key: str) -> dict:
    """Check every page. Returns textless_pages, ocr_status, notes, and the pages that need OCR."""
    short, empty = [], []
    for page in doc:
        n = len(re.sub(r"\s", "", page.get_text()))
        if n == 0:
            empty.append(page)
        elif n < MIN_PAGE_CHARS:
            short.append(page.number + 1)            # has a text layer, just little text (titles, dividers)
    blank, plates, needs, unchecked = [], [], [], []
    for page in empty:
        if needs:                                    # one text page without a text layer settles it
            unchecked.append(page.number + 1)
            continue
        pix = page.get_pixmap(dpi=40, colorspace=pymupdf.csGRAY)
        s = pix.samples
        dark = len(s) - len(s.translate(None, bytes(range(128))))
        if dark / max(1, len(s)) < BLANK_DARK_FRACTION:
            blank.append(page.number + 1)
            continue
        nw = words(tesseract_page(page, tmp, key))
        (plates if nw < PLATE_MAX_WORDS else needs).append((page.number + 1, nw))
    textless = sorted(short + [p.number + 1 for p in empty])
    parts = []
    if short:
        parts.append(f"pp.{page_ranges(short)} have a short text layer (<{MIN_PAGE_CHARS} chars)")
    if blank:
        parts.append(f"pp.{page_ranges(blank)} blank")
    if plates:
        parts.append(f"pp.{page_ranges([p for p, _ in plates])} image-only, OCR finds <{PLATE_MAX_WORDS} "
                     "words (plates)")
    if needs:
        p, nw = needs[0]
        parts.append(f"p.{p} has no text layer but OCR finds {nw} words"
                     + (f"; pp.{page_ranges(unchecked)} not checked" if unchecked else ""))
    return dict(textless_pages=page_ranges(textless),
                ocr_status="OCR needed" if needs else "no OCR needed",
                notes="auto: " + "; ".join(parts),
                ocr_pages=[p for p, _ in needs] + unchecked)


def sample_indices(pages: list[int], k: int = SAMPLE_PAGES) -> list[int]:
    """k pages spread over the list (at 1/6, 3/6, 5/6 for k=3), skipping covers on longer docs."""
    n = len(pages)
    return sorted({pages[min(n - 1, int(n * (i + 0.5) / k))] for i in range(k)}) if n else []


def detect_languages(text: str) -> tuple[list[tuple[str, float]], float, int]:
    """[(language, share)], mean confidence of the per-chunk detections, chars of text."""
    global _detector
    if _detector is None:
        _detector = LanguageDetectorBuilder.from_languages(*LANGS).build()
    chunks, buf = [], ""
    for line in re.sub(r"[ \t]+", " ", text).splitlines():
        line = line.strip()
        if line:
            buf = f"{buf} {line}" if buf else line
            if len(buf) >= 400:
                chunks.append(buf)
                buf = ""
    if buf:
        chunks.append(buf)
    chunks = [c for c in chunks if len(re.findall(r"[^\W\d_]", c)) >= 20]
    weights: Counter = Counter()
    conf_sum = 0.0
    for c, lang in zip(chunks, _detector.detect_languages_in_parallel_of(chunks)):
        if lang is not None:
            weights[lang] += len(c)
            conf_sum += _detector.compute_language_confidence(c, lang) * len(c)
    total = sum(weights.values())
    if not total:
        return [], 0.0, len(text)
    return [(lang, w / total) for lang, w in weights.most_common()], conf_sum / total, len(text)


def plan_new_row(a: dict, tmp: Path) -> dict:
    """Plan row for one new audit row (raises on failure; the caller records it)."""
    key = a["attachment_key"]
    path = Path(a["file_path"])
    if a.get("file_exists") != "yes" or not path.is_file():
        raise RuntimeError(f"PDF not found: {path}")
    row = dict(item_key=a["item_key"], attachment_key=key, author=a["first_author"], year=a["year"],
               title=a["title"], page_count=a["page_count"], text_layer=a["text_layer"],
               textless_pages="all", ocr_status="OCR needed", notes="", languages="", file_path=str(path))
    with pymupdf.open(path) as doc:              # read-only
        if a["text_layer"] == MIXED:
            m = classify_mixed(doc, tmp, key)
            row.update(textless_pages=m["textless_pages"], ocr_status=m["ocr_status"], notes=m["notes"])
            ocr_pages = m["ocr_pages"]
        else:
            ocr_pages = list(range(1, doc.page_count + 1))
        info = {}
        if row["ocr_status"] == "OCR needed":
            pages = sample_indices(ocr_pages)
            text = "\n".join(tesseract_page(doc[p - 1], tmp, key) for p in pages)
            langs, conf, nchars = detect_languages(text)
            if not langs:
                raise RuntimeError(f"no language detected in OCR of sample pages {page_ranges(pages)} "
                                   f"({nchars} chars of text)")
            used = [l for l, s in langs if s >= MIN_LANG_SHARE] or [langs[0][0]]
            row["languages"] = "+".join(LANGS[l] for l in used)
            info = dict(shares=", ".join(f"{l.name.title()} {s:.0%}" for l, s in langs),
                        confidence=conf, sample_pages=page_ranges(pages), chars=nchars)
    return dict(row=row, info=info)


def find_new_audit_rows(plan: list[dict], manifest: list[dict]) -> list[dict]:
    done = {r["attachment_key"] for r in plan} | {r["attachment_key"] for r in manifest}
    seen, out = set(), []
    for a in read_csv(AUDIT_CSV):
        k = a["attachment_key"]
        if (a["content_type"] == "PDF" and a["text_layer"] in (NEEDS_OCR, MIXED)
                and k not in done and k not in seen):
            seen.add(k)
            out.append(a)
    return out


def short(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


def show_plan_table(planned: list[dict]) -> None:
    say(f"\n{'#':>2}  {'key':8}  {'file':42} {'pp.':>5}  {'status':13}  {'detected (sample pages)':34}"
        f" {'conf':>5}  tesseract")
    for i, p in enumerate(planned, 1):
        r, inf = p["row"], p["info"]
        file = short(f"{r['author'].split(',')[0]} {r['year']} {r['title']}", 42)
        det = f"{inf['shares']} (pp.{inf['sample_pages']})" if inf else "-"
        conf = f"{inf['confidence']:.2f}" if inf else "-"
        say(f"{i:>2}  {r['attachment_key']:8}  {file:42} {r['page_count']:>5}  {r['ocr_status']:13}  "
            f"{short(det, 34):34} {conf:>5}  {r['languages'] or '-'}")
        if r["notes"]:
            say(f"{'':14}{r['notes']}")


def installed_tess_langs() -> set[str]:
    r = subprocess.run(["wsl", "tesseract", "--list-langs"], capture_output=True, text=True, timeout=60)
    return {ln.strip() for ln in r.stdout.splitlines()[1:] if ln.strip() and ln.strip() != "osd"}


def confirm_languages(planned: list[dict]) -> bool:
    ocr_rows = [p for p in planned if p["row"]["ocr_status"] == "OCR needed"]
    if not ocr_rows:
        return True
    valid = installed_tess_langs()
    while True:
        ans = ask("\nAccept these languages and run OCR? [y] yes  [n] no, stop here  "
                  "or '<#> <codes>' to change one file (e.g. '2 ita+lat'): ")
        if ans.lower() in ("y", "yes"):
            return True
        if ans.lower() in ("n", "no", ""):
            return False
        m = re.fullmatch(r"(\d+)\s+([a-z_]+(?:\+[a-z_]+)*)", ans)
        if not m or not 1 <= int(m.group(1)) <= len(planned):
            say("  Not understood. Type y, n, or a row number and codes like '2 ita+lat'.")
            continue
        p = planned[int(m.group(1)) - 1]
        bad = [c for c in m.group(2).split("+") if c not in valid]
        if p["row"]["ocr_status"] != "OCR needed":
            say(f"  Row {m.group(1)} is 'no OCR needed'; it gets no languages.")
        elif bad:
            say(f"  Not installed in Tesseract: {', '.join(bad)} (installed: {'+'.join(sorted(valid))})")
        else:
            p["row"]["languages"] = m.group(2)
            p["row"]["notes"] = "; ".join(x for x in (p["row"]["notes"], "languages set by user") if x)
            say(f"  Row {m.group(1)} ({p['row']['attachment_key']}): languages now {m.group(2)}")
            show_plan_table(planned)


def append_plan_rows(rows: list[dict]) -> None:
    """Append only; existing rows are never rewritten, reordered or deleted."""
    assert_not_storage(PLAN)
    with open(PLAN, encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    if header != PLAN_FIELDS:
        raise StepFailed(f"ocr_plan.csv has an unexpected header {header}; not appending")
    raw = PLAN.read_bytes()
    with open(PLAN, "a", encoding="utf-8", newline="") as f:
        if raw and not raw.endswith(b"\n"):
            f.write("\r\n")
        csv.DictWriter(f, fieldnames=PLAN_FIELDS).writerows(rows)


# --------------------------------------------------------------------- 6 OCR

def ocr_targets() -> list[dict]:
    done = {r["attachment_key"] for r in read_csv(MANIFEST)}
    return [r for r in read_csv(PLAN) if r["ocr_status"] == "OCR needed" and r["attachment_key"] not in done]


def kill_tree(proc: subprocess.Popen, key: str) -> None:
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    subprocess.run(["wsl", "pkill", "-f", f"ocrmypdf.*{key}"], capture_output=True)


def run_ocr_one(entry: dict, timeout: int) -> tuple[bool, str]:
    """One run_ocr.py process for one key, output streamed live. Returns (ok, reason)."""
    key = entry["attachment_key"]
    cmd = [sys.executable, "-u", str(ROOT / "run_ocr.py"), key]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    tail: deque[str] = deque(maxlen=200)

    def pump() -> None:
        for line in proc.stdout:
            print("    " + line, end="", flush=True)
            tail.append(line.rstrip())

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(proc, key)
        proc.wait()
        reader.join(10)
        return False, f"timed out after {timeout // 60} min"
    reader.join(10)
    in_manifest = key in {r["attachment_key"] for r in read_csv(MANIFEST)}
    if rc == 0 and in_manifest and (OUT_DIR / f"{key}.pdf").is_file():
        return True, ""
    if rc == 0:
        return False, last_error_line(list(tail)) if tail else "finished but produced no manifest entry"
    return False, last_error_line(list(tail))


def remove_partial_output(key: str) -> None:
    """A failed key must leave nothing in ocr_output that could pass for a finished file."""
    if key in {r["attachment_key"] for r in read_csv(MANIFEST)}:
        return                                       # registered earlier (e.g. by hand); leave it alone
    for p in OUT_DIR.glob(f"{key}.*"):
        assert_not_storage(p)
        p.unlink(missing_ok=True)
        say(f"    removed partial output {p.relative_to(ROOT)}")


def show_sidecar_sample(key: str) -> None:
    txt = OUT_DIR / f"{key}.txt"
    text = txt.read_text(encoding="utf-8", errors="replace") if txt.is_file() else ""
    pages = text.split("\f")
    filled = [i for i, p in enumerate(pages) if words(p) >= 20]
    say(f"\n---- Sidecar sample: {txt.relative_to(ROOT)} ({len(text)} chars, {len(pages)} pages, "
        f"{len(filled)} with text) ----")
    if not filled:
        say("(no page with text in the sidecar)")
        return
    i = filled[len(filled) // 2]
    sample = re.sub(r"\n{3,}", "\n\n", pages[i].strip())[:SIDECAR_SAMPLE_CHARS]
    say(f"[PDF p. {i + 1}]\n{sample}\n---- end of sample ----")


def run_ocr_step(targets: list[dict], args, failures: list[dict]) -> tuple[int, bool]:
    """Returns (number OCR'd, stopped_by_user)."""
    heading(f"6. OCR ({len(targets)} file(s), one run_ocr.py run each)")
    ok_count, sample_shown = 0, args.yes
    for n, entry in enumerate(targets, 1):
        key = entry["attachment_key"]
        pages = int(entry["page_count"] or 0)
        timeout = max(args.min_timeout * 60, args.timeout_per_page * pages)
        say(f"\n[{n}/{len(targets)}] {key}: {entry['author']} {entry['year']}, {pages} pp., "
            f"languages {entry['languages'] or '(empty)'}, timeout {timeout // 60} min")
        t0 = time.time()
        try:
            ok, reason = run_ocr_one(entry, timeout)
        except Exception as e:  # noqa: BLE001 -- e.g. the process could not be started
            ok, reason = False, f"{type(e).__name__}: {e}"
        if ok:
            ok_count += 1
            say(f"  OK: {key} in {(time.time() - t0) / 60:.1f} min")
        else:
            say(f"  FAILED: {key}: {reason}")
            failures.append(failure(entry, "OCR", reason))
            try:
                remove_partial_output(key)
            except Exception as e:  # noqa: BLE001
                say(f"    could not remove partial output for {key}: {e}")
        if ok and not sample_shown:
            sample_shown = True
            show_sidecar_sample(key)
            rest = len(targets) - n
            if rest and ask(f"\nContinue OCR with the remaining {rest} file(s)? [y/n]: ").lower() not in ("y", "yes"):
                say("Stopped at your request; the remaining files stay planned and will be OCR'd next run.")
                return ok_count, True
    return ok_count, False


# ------------------------------------------------------------------ 7 ingest

def run_ingest() -> int:
    heading("7. Ingest (ingest.py, incremental)")
    if claude_desktop_processes():
        say("Claude Desktop was started during the run; quit it (incl. tray) and run ingest.py yourself.")
        return 1
    return subprocess.run([sys.executable, str(ROOT / "ingest.py")], cwd=ROOT).returncode


def ingest_preview(ocr_keys: list[str]) -> None:
    """What a plain ingest.py run would index, from the Zotero API and a read-only look at the index."""
    import ingest
    lib = ingest.Library(None)
    sources = ingest.Builder(lib, None).sources()   # fingerprints only; no text extraction, no model
    known = {}
    if KEYWORD_DB.exists():
        con = sqlite3.connect(f"{KEYWORD_DB.as_uri()}?mode=ro", uri=True)
        known = {sid: (c, m) for sid, c, m in con.execute("SELECT source_id, content_fp, meta_fp FROM sources")}
        con.close()
    ocr_ids = {f"pdf:{k}" for k in ocr_keys}
    new = [s for s in sources if s.id in ocr_ids or s.id not in known or known[s.id][0] != s.content_fp]
    new_ids = {s.id for s in new}
    meta = [s for s in sources if s.id not in new_ids and known.get(s.id, ("", ""))[1] != s.meta_fp]
    current = {s.id for s in sources}
    gone = [sid for sid in known if sid not in current]
    say(f"ingest.py would see {len(sources)} sources: {len(new)} new/changed, {len(meta)} metadata-only, "
        f"{len(gone)} deleted, {len(sources) - len(new) - len(meta)} unchanged")
    if ocr_keys:
        say(f"  (counting the {len(ocr_keys)} file(s) above as re-indexed from their OCR copies)")
    by_item: dict[str, Counter] = {}
    for s in new:
        by_item.setdefault(s.item_key, Counter())[s.id.split(":")[0]] += 1
    for item_key, kinds in list(by_item.items())[:40]:
        d = lib.items[item_key].data
        cr = d.get("creators") or [{}]
        who = cr[0].get("lastName") or cr[0].get("name", "")
        say(f"  {item_key}  {short(f'{who} {ingest.year_of(lib.items[item_key].item)} {d.get('title', '')}', 70):70}"
            f"  {', '.join(f'{k} {v}' for k, v in kinds.items())}")
    if len(by_item) > 40:
        say(f"  ... and {len(by_item) - 40} more items")
    for sid in gone[:20]:
        say(f"  would remove: {sid}")


# ------------------------------------------------------------------ summary

def failure(row: dict, stage: str, reason: str) -> dict:
    return dict(attachment_key=row["attachment_key"], item_key=row["item_key"],
                author=row.get("author") or row.get("first_author", ""), year=row["year"], title=row["title"],
                stage=stage, reason=reason, date=datetime.date.today().isoformat())


def summary(ocr_ok: int, failures: list[dict], ingest_rc: int | None, write: bool) -> None:
    heading("Summary")
    say(f"OCR'd successfully: {ocr_ok}")
    say(f"Failures: {len(failures)}")
    for f in failures:
        say(f"  {f['attachment_key']}  [{f['stage']}]  {f['author']} {f['year']}, {short(f['title'], 60)}")
        say(f"      reason: {f['reason']}")
    if ingest_rc is not None:
        say("Ingest: " + ("ok" if ingest_rc == 0 else f"FAILED (exit code {ingest_rc}); see db\\ingest.log"))
    if write:
        assert_not_storage(FAILURES_CSV)
        with open(FAILURES_CSV, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FAILURE_FIELDS)
            w.writeheader()
            w.writerows(failures)
        say(f"Failure list written to {FAILURES_CSV.name}")
    else:
        say(f"(dry run: {FAILURES_CSV.name} not written)")


# --------------------------------------------------------------------- main

def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="steps 1-5 only: audit and planning; no ocr_plan.csv write, no OCR, no ingest")
    ap.add_argument("--yes", action="store_true", help="skip the language and sidecar-sample prompts")
    ap.add_argument("--timeout-per-page", type=int, default=30, metavar="SEC",
                    help="OCR timeout per file = max(--min-timeout, SEC x pages) (default 30)")
    ap.add_argument("--min-timeout", type=int, default=15, metavar="MIN", help="default 15 minutes")
    args = ap.parse_args()

    failures: list[dict] = []
    try:
        preflight(args.dry_run)
        run_audit()

        heading("3-5. Planning new OCR rows")
        plan, manifest = read_csv(PLAN), read_csv(MANIFEST)
        new = find_new_audit_rows(plan, manifest)
        say(f"PDFs in audit.csv needing OCR or mixed, not yet in ocr_plan.csv or ocr_manifest.csv: {len(new)}")
        planned = []
        with tempfile.TemporaryDirectory(prefix="update_library_") as tmp:
            for i, a in enumerate(new, 1):
                say(f"  [{i}/{len(new)}] {a['attachment_key']} ({a['text_layer']}, {a['page_count']} pp.): "
                    f"{short(a['first_author'] + ' ' + a['year'] + ' ' + a['title'], 70)}")
                try:
                    planned.append(plan_new_row(a, Path(tmp)))
                except Exception as e:  # noqa: BLE001 -- recorded; the row is not appended, so it is retried
                    reason = f"{type(e).__name__}: {e}"
                    say(f"      FAILED: {reason}")
                    failures.append(failure(a, "planning", reason))
        if planned:
            show_plan_table(planned)

        pending = [r for r in plan if r["ocr_status"] == "OCR needed"
                   and r["attachment_key"] not in {m["attachment_key"] for m in manifest}]
        will_ocr = pending + [p["row"] for p in planned if p["row"]["ocr_status"] == "OCR needed"]

        if args.dry_run:
            heading("Dry run: what would happen")
            say(f"Would append {len(planned)} row(s) to ocr_plan.csv"
                + (":" if planned else "."))
            for p in planned:
                r = p["row"]
                say(f"  {r['attachment_key']}  {r['ocr_status']:13}  {r['languages'] or '-':10}  "
                    f"{short(r['author'] + ' ' + r['year'] + ' ' + r['title'], 60)}")
            if will_ocr:
                say(f"Would OCR {len(will_ocr)} file(s) (after asking you to confirm the languages):")
                for r in will_ocr:
                    src = "earlier plan row, not in manifest (retry)" if r in pending else "new row"
                    say(f"  {r['attachment_key']}  {r['languages'] or '(empty)':10}  {r['page_count']:>5} pp.  "
                        f"{short(r['author'] + ' ' + r['year'] + ' ' + r['title'], 55):55}  [{src}]")
            else:
                say("Nothing needs OCR: steps 5-6 (language confirmation, OCR) would be skipped.")
            say("Then ingest.py (incremental) would run:")
            try:
                ingest_preview([r["attachment_key"] for r in will_ocr])
            except Exception as e:  # noqa: BLE001
                say(f"  (could not compute the ingest preview: {type(e).__name__}: {e})")
            summary(0, failures, None, write=False)
            return 1 if failures else 0

        if planned:
            if not args.yes and not confirm_languages(planned):
                say("Stopped before any change: nothing appended to ocr_plan.csv, no OCR, no ingest.")
                return 1
            append_plan_rows([p["row"] for p in planned])
            say(f"Appended {len(planned)} row(s) to ocr_plan.csv.")

        targets = ocr_targets()
        ocr_ok, stopped = 0, False
        if targets:
            ocr_ok, stopped = run_ocr_step(targets, args, failures)
        else:
            say("\nNothing needs OCR; skipping language confirmation and OCR.")
        if stopped:
            summary(ocr_ok, failures, None, write=True)
            return 1
        ingest_rc = run_ingest()
        summary(ocr_ok, failures, ingest_rc, write=True)
        return 1 if failures or ingest_rc else 0
    except StepFailed as e:
        say(f"\nSTOPPED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
