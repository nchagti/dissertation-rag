"""OCR the PDFs listed in ocr_plan.csv with OCRmyPDF running in WSL.

Inputs in Zotero storage are never touched: each is copied to a temp file
named <attachment key>.pdf and OCRmyPDF reads the copy.

Usage:
    python run_ocr.py D2586V7E          # specific attachment keys
    python run_ocr.py --all             # every row marked "OCR needed"
"""
from __future__ import annotations

import argparse
import csv
import datetime
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / "ocr_plan.csv"
OUT_DIR = ROOT / "ocr_output"
MANIFEST = ROOT / "ocr_manifest.csv"
MANIFEST_FIELDS = ["attachment_key", "item_key", "author", "year", "source_file",
                   "ocr_pdf", "sidecar_txt", "languages", "ocrmypdf_args", "date"]
COMMON_ARGS = ["--rotate-pages", "--deskew", "--jobs", "4"]


def to_wsl(path: Path) -> str:
    drive, rest = path.resolve().drive, path.resolve().as_posix()[2:]
    return f"/mnt/{drive[0].lower()}{rest}"


def load_plan() -> list[dict]:
    with open(PLAN, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def update_manifest(row: dict) -> None:
    rows = []
    if MANIFEST.exists():
        with open(MANIFEST, encoding="utf-8-sig", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["attachment_key"] != row["attachment_key"]]
    rows.append(row)
    with open(MANIFEST, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def resolve_source(entry: dict) -> Path | None:
    """The recorded path, or the sole PDF in the attachment folder if the file was renamed."""
    src = Path(entry["file_path"])
    if src.exists():
        return src
    pdfs = list(src.parent.glob("*.pdf"))
    if len(pdfs) == 1:
        print(f"{entry['attachment_key']}: recorded path not found; using {pdfs[0].name}")
        return pdfs[0]
    print(f"{entry['attachment_key']}: recorded path not found and {len(pdfs)} PDFs in folder; skipping")
    return None


def sidecar_pages(sidecar: str) -> tuple[int, list[int]]:
    """(page count, pages OCRmyPDF left alone under --skip-text), read from the sidecar: pages are
    separated by form feeds, and a run of untouched pages is one "[OCR skipped on page(s) N-M]" marker."""
    kept = []
    for m in re.finditer(r"\[OCR skipped on page\(s\) (\d+)(?:-(\d+))?\]", sidecar):
        kept.extend(range(int(m[1]), int(m[2] or m[1]) + 1))
    markers = len(re.findall(r"\[OCR skipped on page\(s\) ", sidecar))
    return sidecar.count("\f") + 1 + len(kept) - markers, kept


def page_ranges(pages: list[int]) -> str:
    """[1, 2, 3, 7] -> "1-3, 7"."""
    runs: list[list[int]] = []
    for p in pages:
        if runs and p == runs[-1][1] + 1:
            runs[-1][1] = p
        else:
            runs.append([p, p])
    return ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in runs)


def ocr_one(entry: dict, tmp: Path) -> bool:
    key, langs = entry["attachment_key"], entry["languages"].strip()
    if not langs:
        print(f"{key}: languages cell is empty -- run language detection first; skipping")
        return False
    src = resolve_source(entry)
    if src is None:
        return False
    tmp_in = tmp / f"{key}.pdf"
    shutil.copyfile(src, tmp_in)

    out_pdf, out_txt = OUT_DIR / f"{key}.pdf", OUT_DIR / f"{key}.txt"
    # OCRmyPDF writes into tmp; only a finished run is moved to OUT_DIR, so no partial output lands there.
    tmp_pdf, tmp_txt = tmp / f"{key}.ocr.pdf", tmp / f"{key}.ocr.txt"
    # --skip-text always: the audit only samples a few pages, so a file classed "none" can still
    # have text pages (e.g. Google Books notice pages), which would otherwise abort the whole file.
    args = ["-l", langs, *COMMON_ARGS, "--skip-text"]
    cmd = ["wsl", "ocrmypdf", *args, "--sidecar", to_wsl(tmp_txt), to_wsl(tmp_in), to_wsl(tmp_pdf)]
    print(f"\n=== {key} ({entry['author']} {entry['year']}, {entry['page_count']} pp.) ===")
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"{key}: ocrmypdf failed with exit code {result.returncode}")
        return False
    total, kept = sidecar_pages(tmp_txt.read_text(encoding="utf-8", errors="replace"))
    print(f"{key}: {total - len(kept)} page(s) OCR'd, {len(kept)} kept their "
          f"existing text{' (PDF p. ' + page_ranges(kept) + ')' if kept else ''}")
    shutil.move(tmp_txt, out_txt)
    shutil.move(tmp_pdf, out_pdf)

    update_manifest(dict(
        attachment_key=key, item_key=entry["item_key"], author=entry["author"], year=entry["year"],
        source_file=str(src), ocr_pdf=str(out_pdf.relative_to(ROOT)),
        sidecar_txt=str(out_txt.relative_to(ROOT)), languages=langs,
        ocrmypdf_args=" ".join(args), date=datetime.date.today().isoformat()))
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("keys", nargs="*")
    parser.add_argument("--all", action="store_true")
    opts = parser.parse_args()

    plan = [r for r in load_plan() if r["ocr_status"] == "OCR needed"]
    if not opts.all:
        wanted = set(opts.keys)
        plan = [r for r in plan if r["attachment_key"] in wanted]
        missing = wanted - {r["attachment_key"] for r in plan}
        if missing:
            sys.exit(f"Not marked 'OCR needed' in ocr_plan.csv: {', '.join(sorted(missing))}")

    OUT_DIR.mkdir(exist_ok=True)
    failed = []
    with tempfile.TemporaryDirectory() as tmp:
        for entry in plan:
            try:
                ok = ocr_one(entry, Path(tmp))
            except Exception as e:  # noqa: BLE001 -- one bad file must not stop the others
                print(f"{entry['attachment_key']}: error: {type(e).__name__}: {e}")
                ok = False
            if not ok:
                failed.append(entry["attachment_key"])
    print(f"\nDone: {len(plan) - len(failed)} ok, {len(failed)} failed {failed or ''}")
    if failed:
        sys.exit(1)  # only after every file has been attempted


if __name__ == "__main__":
    main()
