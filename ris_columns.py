r"""Split Muratori, RIS X (1727) into the Mussato works not otherwise available as text, and OCR them
one printed column at a time, so each chunk can be cited by RIS column. (De obsidione, cols. 687-714, is
left out: Gianola's edition is in the library from MQDQ.)

    .venv\Scripts\python.exe ris_columns.py split            # the image PDFs, for Zotero
    .venv\Scripts\python.exe ris_columns.py ocr [KEY ...]    # Tesseract (WSL), resumable
    .venv\Scripts\python.exe ris_columns.py build [KEY ...]  # TSV -> .txt (format A, column markers)
    .venv\Scripts\python.exe ris_columns.py ocr --scans 301,592 --test a   # trial pages, RIS_X scan numbers

RIS_X.pdf is the Bayerische Staatsbibliothek scan (urn:nbn:de:bvb:12-bsb10868999-6): one JPEG per page,
about 300 dpi, with no text layer. It declares 72 dpi, so the split PDFs get page sizes for 300 dpi
(the JPEGs are copied, not re-encoded).

The OCR model is UB Mannheim's german_print (tessdata_best), trained on early modern print: unlike
Tesseract's lat, it reads the long s as a long s rather than as f. Its path is TESSDATA (in WSL). Sauvola
(adaptive) thresholding is needed: with Tesseract's default global Otsu, the shadow toward the binding
blanks out line ends and curved, skewed columns come out as garbage.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
import pymupdf
from scipy.ndimage import grey_opening, uniform_filter1d

ROOT = Path(__file__).resolve().parent
SOURCE_PDF = ROOT / "RIS_X.pdf"
OUT = ROOT / "ris_x"
TESSDATA = "~/rist/models"            # WSL path holding german_print.traineddata
MODEL = "german_print"
JOBS = 4
DPI = 300

# Scan pages (1-based, in RIS_X.pdf) of each work, and how each page is numbered. Two-column pages
# run from `first_page` (left column `first_col`) to `last_page`; `single` pages are one wide column
# with the printed number given; `blank` pages go into the PDF but are not OCR'd.
WORKS = {
    "historia_augusta": dict(
        title="Albertino Mussato, De gestis Heinrici VII Caesaris (Historia Augusta), RIS X, cols. 1-568",
        pages=(198, 487), first_page=208, first_col=9, last_page=487,
        # Binding order of the scan: title (198, unnumbered: not OCR'd), Pignoria's dedication (col. 5),
        # Muratori's Praefatio (1-3), Pignoria's Spicilegium (7-8); 199, 201 and 205 are blank versos.
        single={200: 5, 202: 1, 203: 2, 204: 3, 206: 7, 207: 8}, blank={198, 199, 201, 205}),
    "ludovicus_bavarus": dict(
        title="Albertino Mussato, Ludovicus Bavarus ad filium, RIS X, cols. 769-784",
        pages=(590, 597), first_page=590, first_col=769, last_page=597, single={}, blank=set()),
}


def to_wsl(path: Path) -> str:
    drive, rest = path.resolve().drive, path.resolve().as_posix()[2:]
    return f"/mnt/{drive[0].lower()}{rest}"


def crops_of(work: dict, scan: int) -> list[tuple[str, int]]:
    """[(crop, printed column)] for one scan page: crop is "L", "R" or "F" (full width)."""
    if scan in work["blank"]:
        return []
    if scan in work["single"]:
        return [("F", work["single"][scan])]
    left = work["first_col"] + 2 * (scan - work["first_page"])
    return [("L", left), ("R", left + 1)]


# ------------------------------------------------------------------ split

def split() -> None:
    src = pymupdf.open(SOURCE_PDF)
    OUT.mkdir(exist_ok=True)
    for key, work in WORKS.items():
        out = pymupdf.open()
        first, last = work["pages"]
        for scan in range(first, last + 1):
            (xref, *_), = src[scan - 1].get_images(full=True)
            img = src.extract_image(xref)
            w, h = img["width"] * 72 / DPI, img["height"] * 72 / DPI
            out.new_page(width=w, height=h).insert_image(pymupdf.Rect(0, 0, w, h), stream=img["image"])
        out.set_metadata({"title": work["title"], "subject": f"{SOURCE_PDF.name}, scans {first}-{last}"})
        path = OUT / f"{key}.pdf"
        out.save(path, garbage=3, deflate=True)
        print(f"{path.name}: {last - first + 1} pages, {path.stat().st_size / 2**20:.0f} MB")


# ------------------------------------------------------------------ layout

GUTTER_SCALE = 0.25
GUTTER_BANDS = 14
GUTTER_TOLERANCE = 12       # px at GUTTER_SCALE: a band's valley this far from the median is an outlier


def gray(page: pymupdf.Page, scale: float = 1.0) -> np.ndarray:
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.stride)[:, :pix.w]


def gutter(page: pymupdf.Page) -> tuple[float, float]:
    """The column gutter as x = a + b*y in full-resolution pixels.

    Ink is measured on a small render; narrow spikes (the printed rule, stray marks) are removed with a
    1-D opening so the gutter shows as a valley between two columns. A valley is sought in each of
    GUTTER_BANDS horizontal bands; bands far from the median valley (full-width titles, ornaments, blank
    space) are dropped and a line is fitted through the rest, which absorbs the skew of the scan.
    """
    a = gray(page, GUTTER_SCALE).astype(float)
    dark = a < np.median(a) * 0.72
    h, w = dark.shape
    lo, hi = int(w * .3), int(w * .7)
    ys, xs = [], []
    for k in range(GUTTER_BANDS):
        y0, y1 = int(h * (.08 + k * .84 / GUTTER_BANDS)), int(h * (.08 + (k + 1) * .84 / GUTTER_BANDS))
        p = grey_opening(uniform_filter1d(dark[y0:y1].mean(axis=0), 3), size=7)
        if p[lo:hi].max() < 0.08:                       # nearly blank band
            continue
        g = lo + int(np.argmin(p[lo:hi]))
        thr, a0, b0 = p[g] + 0.03, g, g
        while a0 > lo and p[a0 - 1] <= thr:
            a0 -= 1
        while b0 < hi - 1 and p[b0 + 1] <= thr:
            b0 += 1
        if p[max(0, a0 - 40):a0].mean() < 0.08 or p[b0 + 1:b0 + 41].mean() < 0.08:
            continue                                    # no text on one side: not a gutter
        ys.append((y0 + y1) / 2)
        xs.append((a0 + b0) / 2)
    ys_, xs_ = np.array(ys), np.array(xs)
    keep = np.abs(xs_ - np.median(xs_)) <= GUTTER_TOLERANCE
    ys_, xs_ = ys_[keep], xs_[keep]
    slope, icpt = np.polyfit(ys_, xs_, 1) if len(xs_) >= 3 else (0.0, float(np.median(xs_)))
    return icpt / GUTTER_SCALE, slope


def crop_images(page: pymupdf.Page, crops: list[str]) -> dict[str, np.ndarray]:
    """Greyscale column images: the far side of the gutter line is painted white."""
    img = gray(page)
    if crops == ["F"]:
        return {"F": img}
    a, b = gutter(page)
    h, w = img.shape
    gx = (a + b * np.arange(h)).astype(int)[:, None]
    xs = np.arange(w)[None, :]
    right = int(gx.max())
    left = int(gx.min())
    out = {}
    out["L"] = np.where(xs[:, :right] < gx, img[:, :right], 255).astype(np.uint8)
    out["R"] = np.where(xs[:, left:] >= gx, img[:, left:], 255).astype(np.uint8)
    return out


# ------------------------------------------------------------------ OCR

def save_png(arr: np.ndarray, path: Path) -> None:
    pymupdf.Pixmap(pymupdf.csGRAY, arr.shape[1], arr.shape[0], arr.tobytes(), 0).save(path)


def tesseract(png: Path, out_base: Path) -> int:
    # Outputs are set with -c: the "tsv"/"txt" config files are looked up in --tessdata-dir, which has none.
    cmd = (f"OMP_THREAD_LIMIT=1 tesseract '{to_wsl(png)}' '{to_wsl(out_base)}' --tessdata-dir {TESSDATA} "
           f"-l {MODEL} --psm 4 --dpi {DPI} -c thresholding_method=2 "
           f"-c tessedit_create_tsv=1 -c tessedit_create_txt=1")
    return subprocess.run(["wsl", "bash", "-c", cmd], capture_output=True).returncode


def ocr(keys: list[str], scans: list[int] | None, test: str | None) -> None:
    src = pymupdf.open(SOURCE_PDF)
    render_lock = Lock()                                # PyMuPDF is not thread-safe
    jobs = []                                           # (work_dir, scan, crops still to OCR)
    for key in keys:
        work = WORKS[key]
        first, last = work["pages"]
        work_dir = OUT / (f"test_{test}" if test else "ocr") / key
        work_dir.mkdir(parents=True, exist_ok=True)
        for scan in scans or range(first, last + 1):
            if not first <= scan <= last:
                continue
            crops = [c for c, _ in crops_of(work, scan)]
            todo = [c for c in crops if not (work_dir / f"{scan:04d}{c}.tsv").exists()]
            if todo:
                jobs.append((work_dir, scan, crops, todo))
    print(f"{sum(len(j[3]) for j in jobs)} column(s) on {len(jobs)} page(s) to OCR with {MODEL}, "
          f"{JOBS} pages at a time")

    def run(job):
        work_dir, scan, crops, todo = job
        with render_lock:
            images = crop_images(src[scan - 1], crops)
        failed = []
        for c in todo:
            png, base = work_dir / f"{scan:04d}{c}.png", work_dir / f"{scan:04d}{c}"
            save_png(images[c], png)
            rc = tesseract(png, base)
            if rc == 0:
                png.unlink()
            else:
                failed.append(f"{c} (exit {rc})")
        return scan, failed

    with ThreadPoolExecutor(JOBS) as pool:
        for n, (scan, failed) in enumerate(pool.map(run, jobs), 1):
            print(f"[{n}/{len(jobs)}] scan {scan}" + (f"  FAILED: {', '.join(failed)}" if failed else ""),
                  flush=True)


# ------------------------------------------------------------------ build

# Line pitch (px between line tops at 300 dpi): the text is set at ~48, the editors' notes and the
# apparatus at ~38-41. A line is a note when the median pitch around it is below NOTE_PITCH.
NOTE_PITCH = 44
PITCH_WINDOW = 3            # lines on each side
BLOCK_GAP = 70              # px between line tops: a rule or heading space, never crossed by the window
MIN_ALPHA = 3               # lines with fewer letters are noise (stray marks, signatures, page edges)
EDGE_SLACK = 12             # px: words ending this far left of the fitted text edge are page-edge noise
GUTTER_LETTER = re.compile(r"^[A-E][.,;:]?$")
NOTES_HEADING = re.compile(r"\bNOT\w*\s+AD\s+RUB", re.I)     # "NOTÆ AD RUBRICAM III." (set in text size)
SIGNATURE = re.compile(r"^\W*(Tom\b.*|[A-Z][a-z]?([a-z])\2{1,3}\W*\d?)\W*$")
HEADER_ZONE = 0.16          # fraction of page height in which the running head is sought
NORMALIZE = str.maketrans({"ſ": "s", "æ": "ae", "Æ": "Ae", "œ": "oe", "Œ": "Oe", "ę": "ae", "ꝙ": "quod",
                           "⸗": "-", "—": "-", "„": "", "·": "", "☛": "", "⁵": ""})


@dataclass
class Line:
    top: int
    words: list[dict]        # TSV rows: left, top, width, height, conf, text

    @property
    def text(self) -> str:
        return " ".join(w["text"] for w in self.words)

    @property
    def left(self) -> int:
        return min(int(w["left"]) for w in self.words)


def read_tsv(path: Path) -> list[Line]:
    lines: dict[tuple, list[dict]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if r["level"] == "5" and r["text"].strip():
                lines.setdefault((r["block_num"], r["par_num"], r["line_num"]), []).append(r)
    out = [Line(min(int(w["top"]) for w in ws), sorted(ws, key=lambda w: int(w["left"]))) for ws in lines.values()]
    return sorted(out, key=lambda l: (l.top, l.left))


def alpha(s: str) -> int:
    return sum(ch.isalpha() for ch in s)


def good_word(w: dict) -> bool:
    return float(w["conf"]) >= 60 and alpha(w["text"]) >= 3


def text_edge(lines: list[Line]) -> tuple[float, float] | None:
    """The left edge of the text as x = c + s*y: Theil-Sen fit through each line's first good word."""
    pts = [(l.top, int(next(w for w in l.words if good_word(w))["left"])) for l in lines
           if any(good_word(w) for w in l.words)]
    if len(pts) < 5:
        return None
    ys, xs = np.array(pts, float).T
    i, j = np.triu_indices(len(ys), 1)
    ok = np.abs(ys[j] - ys[i]) > 200
    s = float(np.median((xs[j] - xs[i])[ok] / (ys[j] - ys[i])[ok])) if ok.any() else 0.0
    # The 25th percentile, not the median: indented verse and paragraph openings sit right of the edge.
    return float(np.percentile(xs - s * ys, 25)), s


def clean_column(lines: list[Line], crop: str, expected: int | None, height: int) -> tuple[list[str], list[str], dict]:
    """(text lines, note lines, report) for one column's OCR lines."""
    rep = {"header": None, "edge_noise": 0, "gutter_letters": 0, "noise_lines": 0}
    # Running head: the first line in the header zone that carries a number or (two-column pages only:
    # a full-width page's capitals are its title) is mostly capitals; it and anything above it go.
    for i, l in enumerate(lines[:4]):
        number = re.fullmatch(r"\W*\d{1,3}\W*", l.text) if crop == "F" else re.search(r"\d{2,}", l.text)
        caps = crop != "F" and sum(c.isupper() for c in l.text) >= 0.5 * max(1, alpha(l.text))
        prose = len(re.findall(r"\b[a-zſæœ]{3,}\b", l.text)) >= 2    # e.g. a note line opening "(95) In hoc"
        if l.top < height * HEADER_ZONE and (number or caps) and not prose:
            rep["header"] = l.text
            lines = lines[i + 1:]
            break
    # The printed number checks the page map; OCR often garbles it, so only a clean wrong number counts.
    read = [int(n) for n in re.findall(r"(?<!\d)\d{3}(?!\d)", rep["header"] or "")]
    if expected is not None and read:
        rep["header_ok"] = expected in read
    edge = text_edge(lines)
    kept: list[Line] = []
    for l in lines:
        words = l.words
        if edge:
            c, s = edge
            limit = c + s * l.top - EDGE_SLACK
            n0 = len(words)
            words = [w for w in words if int(w["left"]) + int(w["width"]) >= limit]
            rep["edge_noise"] += n0 - len(words)
        # Column letters A-E stand in the gutter, at the inner end of the line.
        inner = -1 if crop == "L" else 0
        if crop in "LR" and len(words) > 1 and GUTTER_LETTER.match(words[inner]["text"]):
            words = words[:-1] if inner == -1 else words[1:]
            rep["gutter_letters"] += 1
        line = Line(l.top, words)
        if not words or alpha(line.text) < MIN_ALPHA:
            rep["noise_lines"] += 1
            continue
        kept.append(line)
    # Signature ("Tom. X.", "Ffff") and, at the foot of a right-hand column, the catchword.
    while kept and (SIGNATURE.match(kept[-1].text) or (crop == "R" and len(kept[-1].words) == 1)):
        kept.pop()
        rep["noise_lines"] += 1
    note = [n or bool(NOTES_HEADING.search(l.text)) for l, n in zip(kept, classify(kept))]
    text = [l.text for l, n in zip(kept, note) if not n]
    notes = [l.text for l, n in zip(kept, note) if n]
    rep["text_lines"], rep["note_lines"] = len(text), len(notes)
    return text, notes, rep


def classify(lines: list[Line]) -> list[bool]:
    """True for note lines. A gap over BLOCK_GAP (the rule above the apparatus, the space before
    "NOTAE AD RUBRICAM") separates blocks; within a block a line takes the median pitch of the lines
    around it, so one odd pitch does not move a line across. Pitches under 25 px are OCR line splits."""
    tops = [l.top for l in lines]
    pitch = [b - a for a, b in zip(tops, tops[1:])]
    out = []
    start = 0
    for end in [i + 1 for i, p in enumerate(pitch) if p > BLOCK_GAP] + [len(lines)]:
        ps = [p for p in pitch[start:end - 1] if p >= 25]
        for i in range(start, end):
            near = [p for p in pitch[max(start, i - PITCH_WINDOW): min(end - 1, i + PITCH_WINDOW)] if p >= 25]
            out.append(bool(near or ps) and float(np.median(near or ps)) < NOTE_PITCH)
        start = end
    return out


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).translate(NORMALIZE)
    # Strip the accents of 18th-century Latin print (quòd, cùm, sinè) and stray combining marks.
    s = "".join(ch for ch in unicodedata.normalize("NFD", s) if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", s)


def join_lines(lines: list[str]) -> str:
    """One paragraph: line-end hyphens are joined, other line breaks become spaces."""
    out = ""
    for l in lines:
        l = normalize(l).strip()
        if out.endswith("-") and len(out) > 1 and out[-2].isalpha():
            out = out[:-1] + l
        else:
            out = f"{out} {l}" if out else l
    return re.sub(r"[ \t]+", " ", out).strip()


def build(keys: list[str], test: str | None) -> None:
    for key in keys:
        work = WORKS[key]
        first, last = work["pages"]
        work_dir = OUT / (f"test_{test}" if test else "ocr") / key
        out, report, missing = [], [], []
        for scan in range(first, last + 1):
            pdf_page = scan - first + 1
            for crop, col in crops_of(work, scan):
                tsv = work_dir / f"{scan:04d}{crop}.tsv"
                if not tsv.exists():
                    missing.append(tsv.name)
                    continue
                height = int(next(csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t",
                                                 quoting=csv.QUOTE_NONE))["height"])
                text, notes, rep = clean_column(read_tsv(tsv), crop, col, height)
                label = f"col. {col}"
                out.append(f"## {label} (#{pdf_page}) ####\n{join_lines(text)}\n")
                if notes:
                    out.append(f"## {label}, notes (#{pdf_page}) ####\n{join_lines(notes)}\n")
                flag = "" if rep.get("header_ok", True) else "  << HEADER MISMATCH"
                report.append(f"scan {scan} (#{pdf_page}) {crop} {label}: {rep['text_lines']} text + "
                              f"{rep['note_lines']} note lines; dropped {rep['edge_noise']} edge words, "
                              f"{rep['gutter_letters']} gutter letters, {rep['noise_lines']} noise lines; "
                              f"head: {rep['header']!r}{flag}")
        if missing and not test:
            print(f"{key}: {len(missing)} column(s) not OCR'd yet, e.g. {missing[:3]}")
        txt = OUT / (f"test_{test}" if test else "") / f"{key}.txt"
        txt.write_text("\n".join(out), encoding="utf-8")
        rpt = txt.with_name(f"{key}_build_report.txt")
        rpt.write_text(work["title"] + "\n\n" + "\n".join(report) + "\n", encoding="utf-8")
        bad = sum("HEADER MISMATCH" in r for r in report)
        print(f"{txt.relative_to(ROOT)}: {len(out)} sections; {bad} header mismatch(es); report {rpt.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["split", "ocr", "build"])
    ap.add_argument("keys", nargs="*", help=f"works (default: all): {', '.join(WORKS)}")
    ap.add_argument("--scans", help="comma-separated RIS_X.pdf scan numbers (ocr only)")
    ap.add_argument("--test", metavar="NAME", help="write to ris_x/test_NAME instead of ris_x/ocr")
    args = ap.parse_args()
    keys = args.keys or list(WORKS)
    for k in keys:
        if k not in WORKS:
            sys.exit(f"unknown work {k!r}; choose from {', '.join(WORKS)}")
    if args.command == "split":
        split()
    elif args.command == "ocr":
        ocr(keys, [int(s) for s in args.scans.split(",")] if args.scans else None, args.test)
    else:
        build(keys, args.test)


if __name__ == "__main__":
    main()
