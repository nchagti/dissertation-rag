"""MCP server for the dissertation RAG: search the index built by ingest.py.

Transport: stdio, for Claude Desktop and Claude Code.
Read-only: keyword.sqlite is opened with mode=ro, and Chroma is only queried
(Chroma has no read-only mode; this server never calls add/update/delete).

    .venv\\Scripts\\python.exe rag_server.py
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    # stdout belongs to the MCP protocol. Keep a private copy of it for the transport,
    # then point fd 1 at stderr so nothing else (Hugging Face, tqdm, C libraries,
    # stray prints) can ever write to the protocol stream.
    _MCP_STDOUT_FD = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    # Same for stdin, for a Windows reason: while the transport has a blocking read pending on
    # the stdin pipe, any other thread that queries that handle blocks too, and every native
    # library (numpy, torch) queries the std handles as it loads. The background model load
    # then hangs until the client sends another message. So the transport reads a private
    # duplicate, and the process's standard input becomes NUL.
    _MCP_STDIN_FD = os.dup(0)
    _nul = os.open(os.devnull, os.O_RDONLY)
    os.dup2(_nul, 0)
    os.close(_nul)
    sys.stdin = open(os.devnull, encoding="utf-8")
    if sys.platform == "win32":
        import ctypes
        import msvcrt
        ctypes.windll.kernel32.SetStdHandle(-10, msvcrt.get_osfhandle(0))   # STD_INPUT_HANDLE
        ctypes.windll.kernel32.SetStdHandle(-11, msvcrt.get_osfhandle(1))   # STD_OUTPUT_HANDLE

for _var, _val in {"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TQDM_DISABLE": "1", "TRANSFORMERS_VERBOSITY": "error",
                   "TOKENIZERS_PARALLELISM": "false", "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1"}.items():
    os.environ.setdefault(_var, _val)

import io
import json
import logging
import re
import sqlite3
import threading
import time
import unicodedata
from collections import Counter
from pathlib import Path

import anyio
from mcp.server.fastmcp import FastMCP

PROJECT_DIR = Path(__file__).resolve().parent
DB_DIR = PROJECT_DIR / "db"
KEYWORD_DB = DB_DIR / "keyword.sqlite"
LOG_FILE = DB_DIR / "server.log"
COLLECTION_NAME = "zotero_chunks"
MODEL_NAME = "BAAI/bge-m3"
if (Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-m3").is_dir():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

CANDIDATES = 50            # per retriever, before fusion
RRF_K = 60
MAX_CHARS = 25_000         # cap on any tool response
GPU_MIN_FREE = 2.8 * 2**30 # bge-m3 needs ~2.2 GB; otherwise use CPU

SOURCE_TYPES = ["pdf_text", "snapshot", "text_file", "zotero_highlight", "zotero_note", "adobe_annotation",
                "child_note", "metadata"]
MY_TYPES = ["zotero_highlight", "zotero_note", "adobe_annotation", "child_note"]
LANG_CODES = {"en": "English", "it": "Italian", "la": "Latin", "fr": "French", "de": "German"}
STOPWORDS = set("""
a an and are as at be by for from in into is it its of on or that the this to was were with what which who
il lo la i gli le un una di da del della dei delle nel nella e ed che per con non si al alla sul
et in est ad cum de qui quae quod non sed ut
le la les un une des du de et en que qui dans pour sur au aux
der die das und in den von zu mit sich des auf ist im dem nicht ein eine
""".split())

log = logging.getLogger("rag_server")


# ------------------------------------------------------------- formatting

def strip_header(text: str) -> str:
    """Chunks start with ingest's "[Author year · title · page]" line; results carry a citation instead."""
    first, _, rest = text.partition("\n")
    return rest if first.startswith("[") and first.endswith("]") and rest else text


def norm(s: str) -> str:
    """Lowercase and strip accents, for accent-insensitive matching."""
    return "".join(ch for ch in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(ch)).lower()


def short_title(d: dict) -> str:
    """Zotero's Short Title only if it is really a short form of the title (some chapters hold the
    book's title there); otherwise the title up to ':' or '. '. Same rule as ingest.py's headers."""
    title = d.get("title", "") or ""
    st = (d.get("shortTitle") or "").strip()
    if st and set(re.findall(r"\w+", st.lower())) <= set(re.findall(r"\w+", title.lower())):
        return st
    return re.split(r"[:.]\s", title, maxsplit=1)[0]


def split_tags(s: str) -> set[str]:
    return {t.strip().lower() for t in (s or "").split(";") if t.strip()}


def as_list(v) -> list | None:
    if v is None or v == "" or v == []:
        return None
    return [v] if isinstance(v, (str, int)) else list(v)


def page_of(meta: dict) -> str:
    """ "p. <printed>" or "PDF p. <n>"; empty for snapshots, metadata and unpaged chunks (pdf_page 0).

    Text files give their location ("vv. 1-14", "a. 1208, ...", "fol. I recto") or, for paged
    editions, "p. <printed>"."""
    if meta.get("source_type") == "text_file":
        if meta.get("location"):
            return meta["location"]
        printed = (meta.get("printed_page") or "").strip()
        return f"p. {printed}" if printed else ""
    if meta.get("source_type") in ("snapshot", "metadata") or not meta.get("pdf_page"):
        return ""
    printed = re.sub(r"^p+\.\s*", "", (meta.get("printed_page") or "").strip())
    return f"p. {printed}" if printed else f"PDF p. {meta['pdf_page']}"


def voice(meta: dict, text: str) -> str:
    st = meta["source_type"]
    if st in ("pdf_text", "snapshot"):
        return "source text (the author's words)"
    if st == "text_file":
        return "primary-source text from an edition (not the user's notes)"
    if st == "metadata":
        return "bibliographic record"
    if st == "child_note":
        return "MY note (my words, not the author's)"
    if text.startswith("Highlighted:"):
        return ("MY highlight: the 'Highlighted:' text quotes the source; 'My comment:' and 'Tags:' lines are mine"
                if "My comment:" in text else "MY highlight: the highlighted text quotes the source")
    return "MY note/comment (my words, not the author's)"


# ----------------------------------------------------------------- index

class Index:
    """Read-only view of keyword.sqlite + Chroma, with chunk metadata held in memory for filtering."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.mtime = None
        self._cand: set[str] | None = None
        self.load()

    def load(self) -> None:
        import chromadb
        from chromadb.api.client import SharedSystemClient
        from chromadb.config import Settings
        if not KEYWORD_DB.exists():
            raise RuntimeError(f"Index not found at {KEYWORD_DB}; run ingest.py first.")
        self.db = sqlite3.connect(f"{KEYWORD_DB.as_uri()}?mode=ro", uri=True, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.create_function("rag_ok", 1, lambda cid: self._cand is None or cid in self._cand,
                                deterministic=True)
        self.meta: dict[str, dict] = {}
        for cid, meta in self.db.execute("SELECT chunk_id, meta FROM chunks"):
            self.meta[cid] = json.loads(meta)
        self.items: dict[str, dict] = {r["item_key"]: dict(r) for r in self.db.execute("SELECT * FROM items")}
        for it in self.items.values():
            it["data"] = json.loads(it.pop("data_json") or "{}")
        SharedSystemClient.clear_system_cache()        # pick up a re-ingested Chroma on reload
        client = chromadb.PersistentClient(path=str(DB_DIR), settings=Settings(anonymized_telemetry=False))
        self.col = client.get_collection(COLLECTION_NAME)
        self.mtime = KEYWORD_DB.stat().st_mtime_ns
        log.info("Index loaded: %d chunks, %d items", len(self.meta), len(self.items))

    def refresh(self) -> None:
        """Reload if ingest.py has changed the index since we loaded it."""
        if KEYWORD_DB.stat().st_mtime_ns != self.mtime:
            with self.lock:
                self.db.close()
                self.load()

    # -- citations
    def citation(self, meta: dict, with_page: bool = True) -> str:
        item = self.items.get(meta["item_key"], {})
        d = item.get("data", {})
        creators = d.get("creators", [])
        first = next((c for c in creators if c.get("creatorType") == "author"), creators[0] if creators else None)
        surname = (first.get("lastName") or first.get("name", "")) if first else ""
        title = short_title(d) if d else meta.get("title", "")
        words = title.split()
        title = " ".join(words[:10]) + ("…" if len(words) > 10 else "")
        parts = [" ".join(p for p in (surname, meta.get("year") or item.get("year", "")) if p) or "[no author]",
                 title]
        if with_page and page_of(meta):
            parts.append(page_of(meta))
        return ", ".join(p for p in parts if p)

    def result(self, cid: str, text: str, extra: dict | None = None) -> dict:
        m = self.meta[cid]
        body = strip_header(text)
        r = {"citation": self.citation(m), "source_type": m["source_type"], "voice": voice(m, body), "text": body,
             "chunk_id": cid, "item_key": m["item_key"], "attachment_key": m["attachment_key"] or None,
             "language": m["language"] or None, "ocr": bool(m["ocr"])}
        if m["source_type"] == "pdf_text":
            r.update(my_annotations_on_page=m["my_annotations_on_page"],
                     page_annotation_tags=m.get("page_annotation_tags") or "",
                     has_ink_on_page=bool(m["has_ink_on_page"]))
        if m.get("annotation_tags"):
            r["annotation_tags"] = m["annotation_tags"]
        if m.get("url"):
            r["url"] = m["url"]
        if extra:
            r.update(extra)
        return r

    def texts(self, ids: list[str]) -> dict[str, str]:
        if not ids:
            return {}
        q = f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({','.join('?' * len(ids))})"
        return dict(self.db.execute(q, ids).fetchall())

    # -- filters
    def candidates(self, source_types=None, languages=None, tags=None, item_keys=None,
                   year_from=None, year_to=None) -> set[str] | None:
        """Chunk IDs passing every filter, or None when no filter is set."""
        source_types, languages, tags, item_keys = map(as_list, (source_types, languages, tags, item_keys))
        if not any((source_types, languages, tags, item_keys, year_from, year_to)):
            return None
        if source_types:
            bad = set(source_types) - set(SOURCE_TYPES)
            if bad:
                raise ValueError(f"Unknown source_types {sorted(bad)}; use {SOURCE_TYPES}")
        langs = {LANG_CODES.get(str(l).lower(), str(l).title()) for l in languages} if languages else None
        tagset = {str(t).lower() for t in tags} if tags else None
        keys = set(item_keys) if item_keys else None
        out = set()
        for cid, m in self.meta.items():
            if source_types and m["source_type"] not in source_types:
                continue
            if langs and m["language"] not in langs:
                continue
            if keys and m["item_key"] not in keys:
                continue
            if year_from or year_to:
                y = int(m["year"]) if str(m.get("year", "")).isdigit() else None
                if y is None or (year_from and y < int(year_from)) or (year_to and y > int(year_to)):
                    continue
            if tagset and not tagset & (split_tags(m["tags"]) | split_tags(m.get("annotation_tags"))
                                        | split_tags(m.get("page_annotation_tags"))):
                continue
            out.add(cid)
        return out

    # -- retrievers
    def semantic(self, qvec, cand: set[str] | None, k: int = CANDIDATES) -> list[tuple[str, float]]:
        if cand is not None and not cand:
            return []
        n = min(k, len(cand) if cand is not None else len(self.meta))
        kwargs = {"ids": sorted(cand)} if cand is not None else {}
        res = self.col.query(query_embeddings=[qvec], n_results=n, include=["distances"], **kwargs)
        return [(cid, 1.0 - d) for cid, d in zip(res["ids"][0], res["distances"][0])]

    def keyword(self, fts_query: str, cand: set[str] | None, k: int = CANDIDATES) -> list[tuple[str, float]]:
        if cand is not None and not cand:
            return []
        with self.lock:
            self._cand = cand
            try:
                rows = self.db.execute(
                    "SELECT c.chunk_id, bm25(chunks_fts) AS s FROM chunks_fts JOIN chunks c "
                    "ON c.rowid = chunks_fts.rowid WHERE chunks_fts MATCH ? AND rag_ok(c.chunk_id) "
                    "ORDER BY s LIMIT ?", (fts_query, k)).fetchall()
            finally:
                self._cand = None
        return [(r[0], -r[1]) for r in rows]


def fts_terms(query: str, joiner: str, drop_stopwords: bool) -> str:
    terms = re.findall(r"\w+", query)
    if drop_stopwords:
        kept = [t for t in terms if norm(t) not in STOPWORDS and len(t) > 1]
        terms = kept or terms
    return joiner.join('"' + t.replace('"', "") + '"' for t in terms)


# -------------------------------------------------------------- embedder

class Embedder:
    """bge-m3, loaded by the warm-up thread; the first query waits for it if needed."""

    def __init__(self) -> None:
        self.ready = threading.Event()
        self.model = None
        self.error: Exception | None = None
        self.device = None

    def _load(self) -> None:
        try:
            t0 = time.time()
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cpu"
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                if free >= GPU_MIN_FREE:
                    device = "cuda"
                else:
                    log.warning("Only %.1f GB VRAM free (another process is using the GPU); using CPU",
                                free / 2**30)
            self.model = SentenceTransformer(MODEL_NAME, device=device)
            self.model.max_seq_length = 512
            self.model.encode(["warm-up"], normalize_embeddings=True, show_progress_bar=False)
            self.device = device
            log.info("bge-m3 loaded on %s in %.0f s", device, time.time() - t0)
        except Exception as e:  # noqa: BLE001
            if self.device is None and "cuda" in str(e).lower():
                log.warning("GPU load failed (%s); retrying on CPU", e)
                try:
                    from sentence_transformers import SentenceTransformer
                    self.model = SentenceTransformer(MODEL_NAME, device="cpu")
                    self.model.max_seq_length = 512
                    self.device = "cpu"
                except Exception as e2:  # noqa: BLE001
                    self.error = e2
            else:
                self.error = e
            if self.error:
                log.exception("Could not load bge-m3")
        finally:
            self.ready.set()

    def embed(self, text: str):
        start_background()
        if not self.ready.wait(timeout=600):
            raise RuntimeError("The embedding model is still loading; try again in a minute.")
        if self.error:
            raise RuntimeError(f"The embedding model failed to load: {self.error}")
        return self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()


# ---------------------------------------------------------------- state

_index: Index | None = None
_index_error: Exception | None = None
_index_lock = threading.Lock()
EMBEDDER = Embedder()
INDEX_READY = threading.Event()
_warmup: threading.Thread | None = None
_warmup_lock = threading.Lock()


def _warm_up() -> None:
    """Import the heavy libraries, open the index, then load the model, all in ONE thread.

    Importing numpy/torch/chromadb from two threads at once deadlocks on Windows (seen here:
    one thread stuck in numpy's C-extension import while another imported torch).
    """
    global _index, _index_error
    try:
        import torch  # noqa: F401  (pulls in numpy first, alone)
        import sentence_transformers  # noqa: F401
        import chromadb  # noqa: F401
        with _index_lock:
            _index = Index()
    except Exception as e:  # noqa: BLE001
        _index_error = e
        log.exception("Could not open the index")
    finally:
        INDEX_READY.set()
    EMBEDDER._load()


def start_background() -> None:
    global _warmup
    with _warmup_lock:
        if _warmup is None:
            _warmup = threading.Thread(target=_warm_up, name="warm-up", daemon=True)
            _warmup.start()


def index() -> Index:
    start_background()
    if not INDEX_READY.wait(timeout=600):
        raise RuntimeError("The index is still loading; try again in a minute.")
    if _index is None:
        raise RuntimeError(f"The index could not be opened: {_index_error}")
    with _index_lock:
        _index.refresh()
        return _index


def capped(payload: dict, list_key: str) -> str:
    """JSON-encode payload, dropping list items from the end until it fits MAX_CHARS."""
    items = payload[list_key]
    text = json.dumps(payload, ensure_ascii=False, indent=1)
    dropped = 0
    while len(text) > MAX_CHARS and items:
        if len(items) == 1:
            items[0]["text"] = items[0].get("text", "")[: max(500, len(items[0].get("text", "")) - (len(text) - MAX_CHARS) - 200)] + " […]"
            text = json.dumps(payload, ensure_ascii=False, indent=1)
            break
        items.pop()
        dropped += 1
        payload["truncated"] = f"{dropped} more result(s) omitted to stay under {MAX_CHARS} characters"
        text = json.dumps(payload, ensure_ascii=False, indent=1)
    return text


# -------------------------------------------------------- tool functions

def do_search(query: str, n: int = 10, mode: str = "hybrid", source_types=None, languages=None, tags=None,
              item_keys=None, year_from=None, year_to=None) -> str:
    if mode not in ("hybrid", "semantic", "keyword"):
        raise ValueError('mode must be "hybrid", "semantic" or "keyword"')
    ix = index()
    n = max(1, min(int(n), 50))
    cand = ix.candidates(source_types, languages, tags, item_keys, year_from, year_to)
    notes = []
    sem, kw = [], []
    if mode in ("hybrid", "semantic"):
        sem = ix.semantic(EMBEDDER.embed(query), cand)
    if mode == "keyword":
        try:
            kw = ix.keyword(query, cand)
        except sqlite3.OperationalError as e:
            fixed = fts_terms(query, " ", drop_stopwords=False)
            notes.append(f"FTS5 syntax error ({e}); searched as plain terms instead: {fixed}")
            kw = ix.keyword(fixed, cand) if fixed else []
    elif mode == "hybrid":
        fq = fts_terms(query, " OR ", drop_stopwords=True)
        kw = ix.keyword(fq, cand) if fq else []

    fused: dict[str, dict] = {}
    for name, hits in (("semantic", sem), ("keyword", kw)):
        for rank, (cid, _score) in enumerate(hits, 1):
            f = fused.setdefault(cid, {"rrf": 0.0})
            f["rrf"] += 1.0 / (RRF_K + rank)
            f[f"{name}_rank"] = rank
    order = sorted(fused, key=lambda c: -fused[c]["rrf"])[:n]
    texts = ix.texts(order)
    results = []
    for cid in order:
        f = fused[cid]
        match = ", ".join(f"{k.split('_')[0]} #{v}" for k, v in f.items() if k.endswith("_rank"))
        results.append(ix.result(cid, texts[cid], {"match": match}))
    filters = {k: v for k, v in dict(source_types=source_types, languages=languages, tags=tags, item_keys=item_keys,
                                    year_from=year_from, year_to=year_to).items() if v}
    payload = {"query": query, "mode": mode, "filters": filters or None,
               "hits": {"semantic": len(sem), "keyword": len(kw)}, "results": results}
    if notes:
        payload["note"] = " ".join(notes)
    if not results:
        payload["note"] = (payload.get("note", "") + " No matching chunks. Say so plainly; do not fill in from "
                           "general knowledge. Try other languages, Latin key terms, or keyword mode.").strip()
    return capped(payload, "results")


def chunk_no(cid: str) -> int:
    m = re.search(r":c(\d+)$", cid)
    return int(m.group(1)) if m else 0


def merge_overlap(a: str, b: str) -> str:
    """Join consecutive chunks of one page, removing the ~60-token overlap they share."""
    for k in range(min(len(a), len(b), 2000), 15, -1):
        if a.endswith(b[:k]):
            return a + b[k:]
    return a + "\n\n" + b


def do_get_context(chunk_id: str, pages_before: int = 1, pages_after: int = 1) -> str:
    ix = index()
    if chunk_id not in ix.meta:
        raise ValueError(f"Unknown chunk_id {chunk_id!r}")
    m = ix.meta[chunk_id]
    pages_before, pages_after = max(0, min(int(pages_before), 5)), max(0, min(int(pages_after), 5))
    hit = {"chunk_id": chunk_id, "citation": ix.citation(m), "source_type": m["source_type"]}
    if m["attachment_key"] and m["pdf_page"]:
        p = m["pdf_page"]
        rows = ix.db.execute(
            "SELECT chunk_id, pdf_page, text FROM chunks WHERE attachment_key=? AND source_type='pdf_text' "
            "AND pdf_page BETWEEN ? AND ?", (m["attachment_key"], p - pages_before, p + pages_after)).fetchall()
        by_page: dict[int, list] = {}
        for r in sorted(rows, key=lambda r: (r["pdf_page"], chunk_no(r["chunk_id"]))):
            by_page.setdefault(r["pdf_page"], []).append(r)
        pages = []
        for pno, rs in by_page.items():
            text = strip_header(rs[0]["text"])
            for r in rs[1:]:
                text = merge_overlap(text, strip_header(r["text"]))
            pm = ix.meta[rs[0]["chunk_id"]]
            page = {"pdf_page": pno, "citation": ix.citation(pm), "is_hit_page": pno == p, "text": text,
                    "ocr": bool(pm["ocr"]), "my_annotations_on_page": pm["my_annotations_on_page"],
                    "page_annotation_tags": pm.get("page_annotation_tags") or "",
                    "has_ink_on_page": bool(pm["has_ink_on_page"])}
            pages.append(page)
        payload = {"hit": hit, "pages": pages,
                   "note": "Pages with no text layer (blank pages, plates) have no entry." if pages else
                   "This PDF has no indexed text around that page."}
        return capped(payload, "pages")
    if m["source_type"] == "text_file":
        seq = m.get("chunk_seq", 0)
        ids = sorted((c for c, cm in ix.meta.items() if cm["attachment_key"] == m["attachment_key"]
                      and cm["source_type"] == "text_file"
                      and seq - pages_before <= cm.get("chunk_seq", 0) <= seq + pages_after),
                     key=lambda c: ix.meta[c].get("chunk_seq", 0))
        texts = ix.texts(ids)
        chunks = [ix.result(i, texts[i], {"is_hit": i == chunk_id, "chunk_seq": ix.meta[i].get("chunk_seq", 0)})
                  for i in ids if i in texts]
        return capped({"hit": hit, "neighboring_chunks": chunks,
                       "note": "pages_before/pages_after count chunks here, in file order (a text file is chunked "
                               "by page, verse range, chapter or folio); consecutive chunks may repeat a few lines."},
                      "neighboring_chunks")
    prefix = re.sub(r":c\d+$", "", chunk_id)
    if prefix != chunk_id and m["source_type"] in ("snapshot", "child_note"):
        n = chunk_no(chunk_id)
        ids = [f"{prefix}:c{i}" for i in range(max(1, n - pages_before), n + pages_after + 1)]
        texts = ix.texts([i for i in ids if i in ix.meta])
        chunks = [ix.result(i, texts[i], {"is_hit": i == chunk_id}) for i in ids if i in texts]
        return capped({"hit": hit, "neighboring_chunks": chunks,
                       "note": "pages_before/pages_after count chunks here (this source has no pages)."},
                      "neighboring_chunks")
    texts = ix.texts([chunk_id])
    return capped({"hit": hit, "chunks": [ix.result(chunk_id, texts[chunk_id])],
                   "note": "This chunk has no pages or neighbors to show; it is complete as returned."}, "chunks")


def annotation_order(ix: Index, cid: str) -> tuple:
    m = ix.meta[cid]
    return (m["attachment_key"] or "~", m["pdf_page"] or 10**6, m["source_type"], cid)


def do_get_item(item_key: str, offset: int = 0) -> str:
    ix = index()
    if item_key not in ix.items:
        raise ValueError(f"Item {item_key!r} is not in the index. Use find_items to look it up.")
    it = ix.items[item_key]
    fields = {k: v for k, v in it.items() if k not in ("data", "attachments") and v not in ("", None)}
    fields["short_title"] = short_title(it["data"])
    chunk_ids = [c for c, m in ix.meta.items() if m["item_key"] == item_key]
    atts = []
    for a in json.loads(it["attachments"] or "[]"):
        ms = [ix.meta[c] for c in chunk_ids if ix.meta[c]["attachment_key"] == a["key"]]
        pdf = [m for m in ms if m["source_type"] == "pdf_text"]
        atts.append({**a, "indexed_chunks": dict(Counter(m["source_type"] for m in ms)),
                     "ocr": any(m["ocr"] for m in pdf),
                     "pdf_pages_with_text": len({m["pdf_page"] for m in pdf}),
                     "pages_with_my_ink": sorted({page_of(m) for m in pdf if m["has_ink_on_page"]},
                                                 key=lambda s: int(re.sub(r"\D", "", s) or 0))})
    mine = sorted((c for c in chunk_ids if ix.meta[c]["source_type"] in MY_TYPES),
                  key=lambda c: annotation_order(ix, c))
    offset = max(0, int(offset))
    page = mine[offset:offset + 400]
    texts = ix.texts(page)
    anns = [{"citation": ix.citation(ix.meta[c]), "source_type": ix.meta[c]["source_type"],
             "text": strip_header(texts[c]), "annotation_tags": ix.meta[c].get("annotation_tags") or None,
             "chunk_id": c} for c in page]
    payload = {"citation": ix.citation({"item_key": item_key, "year": it["year"]}, with_page=False),
               "item": fields, "attachments": atts, "my_annotations_and_notes_total": len(mine),
               "offset": offset, "annotations_and_notes": anns,
               "note": "All annotation/note texts are MY words, except 'Highlighted:' lines, which quote the source."}
    text = capped(payload, "annotations_and_notes")
    shown = len(payload["annotations_and_notes"])
    if offset + shown < len(mine):
        payload["next_offset"] = offset + shown
        payload["truncated"] = (f"Showing {offset + 1}-{offset + shown} of {len(mine)}; "
                                f"call get_item again with offset={offset + shown}")
        text = capped(payload, "annotations_and_notes")
    return text


def do_list_annotations(tag=None, item_key=None, has_comment=None, source_type=None, limit: int = 200,
                        offset: int = 0) -> str:
    ix = index()
    types = as_list(source_type) or MY_TYPES
    bad = set(types) - set(MY_TYPES)
    if bad:
        raise ValueError(f"source_type must be one of {MY_TYPES}")
    tagset = {t.lower() for t in as_list(tag)} if tag else None
    ids = []
    cand = [c for c, m in ix.meta.items() if m["source_type"] in types
            and (not item_key or m["item_key"] == item_key)
            and (not tagset or tagset & split_tags(m.get("annotation_tags")))]
    texts = ix.texts(cand) if has_comment is not None else {}
    for c in cand:
        if has_comment is not None:
            commented = "My comment:" in texts[c] or ix.meta[c]["source_type"] == "child_note"
            if commented != bool(has_comment):
                continue
        ids.append(c)

    def order(c):
        m = ix.meta[c]
        return (ix.citation(m, with_page=False).lower(), m["attachment_key"], m["pdf_page"] or 10**6, c)
    ids.sort(key=order)
    offset, limit = max(0, int(offset)), max(1, min(int(limit), 500))
    page = ids[offset:offset + limit]
    texts = ix.texts(page)
    anns = [{"citation": ix.citation(ix.meta[c]), "source_type": ix.meta[c]["source_type"],
             "text": strip_header(texts[c]), "annotation_tags": ix.meta[c].get("annotation_tags") or None,
             "item_key": ix.meta[c]["item_key"], "chunk_id": c} for c in page]
    payload = {"total_matching": len(ids), "offset": offset, "annotations": anns,
               "note": "These are MY annotations: comments, notes and tags are my words; "
                       "'Highlighted:' lines quote the source."}
    text = capped(payload, "annotations")
    shown = len(payload["annotations"])
    if offset + shown < len(ids):
        payload["next_offset"] = offset + shown
        text = capped(payload, "annotations")
    return text


def do_find_items(query: str, limit: int = 25) -> str:
    ix = index()
    words = [w for w in re.findall(r"\w+", norm(query))]
    if not words:
        raise ValueError("Give an author, title word, year or publication to look for.")
    counts: dict[str, Counter] = {}
    for m in ix.meta.values():
        counts.setdefault(m["item_key"], Counter())[m["source_type"]] += 1
    hits = []
    for key, it in ix.items.items():
        hay = norm(" ".join(str(it.get(f) or "") for f in ("authors", "editors", "title", "year", "publication",
                                                          "publisher", "date"))
                   + " " + it["data"].get("shortTitle", ""))
        hay_words = set(re.findall(r"\w+", hay))
        if all(w in hay_words or (len(w) >= 4 and w in hay) for w in words):
            hits.append(key)
    hits.sort(key=lambda k: (ix.items[k]["authors"].lower(), ix.items[k]["year"]))
    items = [{"item_key": k, "citation": ix.citation({"item_key": k, "year": ix.items[k]["year"]}, with_page=False),
              "authors": ix.items[k]["authors"] or None, "editors": ix.items[k]["editors"] or None,
              "year": ix.items[k]["year"], "title": ix.items[k]["title"], "item_type": ix.items[k]["item_type"],
              "publication": ix.items[k]["publication"] or None,
              "attachments": len(json.loads(ix.items[k]["attachments"] or "[]")),
              "indexed_chunks": dict(counts.get(k, {}))} for k in hits[:max(1, min(int(limit), 100))]]
    payload = {"query": query, "total_matching": len(hits), "items": items}
    if not hits:
        payload["note"] = "No item in the library matches. Say plainly that it is not in the library."
    return capped(payload, "items")


# ------------------------------------------------------------ MCP server

INSTRUCTIONS = """\
Search tools for the user's dissertation library (Zotero), in English, Italian, Latin, French and German.

Rules for using these results:
- Tool results are the ONLY basis for citations. Never cite a source, page or quotation that did not come back
  from a tool. If nothing relevant comes back, say so plainly instead of filling in from general knowledge.
- Always cite with the result's `citation` line. It says "p. N" for a printed page and "PDF p. N" when no printed
  page is known; write "PDF p." in that case, never a bare "p.". Text-file editions (text_file) are cited by
  verse range ("vv. 1-14"), year and chapter ("a. 1208, ..."), folio ("fol. I recto") or printed page ("p. N");
  keep that form and never turn a verse range into a page.
- Keep three voices apart: the source's text (pdf_text, snapshot, text_file); passages the user highlighted
  ("Highlighted:" lines quote the source); and the USER's OWN words: zotero_note, adobe_annotation comments,
  child_note, and every "My comment:" or "Tags:" line. Never attribute the user's comments to the author.
- When a pdf_text result has my_annotations_on_page > 0, page_annotation_tags, or has_ink_on_page, mention that
  the user marked that page.
- Text with ocr: true may contain OCR errors, especially names in small caps and footnote numbers. Say so when
  quoting it, and check exact quotations with get_context before relying on them.
- For substantive questions search more than once: in English, in Italian, and with the Latin key terms; use
  mode="keyword" for names, manuscript shelfmarks and exact phrases.
"""

mcp = FastMCP("dissertation-rag", instructions=INSTRUCTIONS)


@mcp.tool()
async def search(query: str, n: int = 10, mode: str = "hybrid", source_types: list[str] | None = None,
                 languages: list[str] | None = None, tags: list[str] | None = None,
                 item_keys: list[str] | None = None, year_from: int | None = None,
                 year_to: int | None = None) -> str:
    """Search the user's dissertation library (PDF pages, web snapshots, plain-text editions, the user's own
    annotations and notes, and item records). source_type "text_file" is primary-source text from an edition
    (chronicles, annals, Mussato's works, Rolandino), not the user's notes; filter on it when scholarship would
    otherwise outrank the sources themselves.

    mode: "hybrid" (default: semantic + keyword, merged by reciprocal rank fusion), "semantic", or "keyword".
    Keyword mode accepts FTS5 syntax: "exact phrase", prefix*, NEAR("a" "b", 5), AND/OR/NOT; use it for names,
    manuscript shelfmarks (e.g. "Vat Lat 2962") and exact wording. Matching ignores case and accents.
    Filters (all optional, combined with AND): source_types from pdf_text, snapshot, text_file, zotero_highlight,
    zotero_note, adobe_annotation, child_note, metadata; languages (English, Italian, Latin, French, German, or
    en/it/la/fr/de); tags (matches item tags, annotation tags and page-level annotation tags); item_keys;
    year_from/year_to.

    How to use the results:
    - Cite ONLY what this tool returns, always with the result's `citation` line. It uses "p. N" for printed
      pages and "PDF p. N" when no printed page is known: write "PDF p.", never a bare "p.". text_file results
      may instead give a verse range ("vv. 1-14"), a year and chapter ("a. 1208, ...") or a folio ("fol. I
      recto"); cite those as given. Never cite a page or source that did not come back from a tool.
    - `voice` says whose words a result is. zotero_note, adobe_annotation comments, child_note and every
      "My comment:"/"Tags:" line are the USER's words, not the author's; "Highlighted:" lines quote the source.
    - If a pdf_text result has my_annotations_on_page > 0, page_annotation_tags or has_ink_on_page, mention
      that the user marked that page.
    - ocr: true means the text was OCR'd and may contain errors (especially small-caps names and footnote
      numbers): say so when quoting, and check exact quotations with get_context.
    - For substantive questions, search several times: in English, in Italian, with the Latin key terms, and
      in keyword mode for names and exact phrases. If nothing relevant comes back, say so plainly.
    """
    return await anyio.to_thread.run_sync(lambda: do_search(query, n, mode, source_types, languages, tags,
                                                            item_keys, year_from, year_to))


@mcp.tool()
async def get_context(chunk_id: str, pages_before: int = 1, pages_after: int = 1) -> str:
    """Read around a search hit: the full text of the PDF pages before and after the hit's page (up to 5 each
    way), or the neighboring chunks of a web snapshot, note or text file (text_file). For snapshots, notes and
    text files, pages_before/pages_after count CHUNKS, not pages. Use it to check exact quotations (especially
    in OCR'd text) and to see a passage's context before citing it. Cite each part with its own `citation`."""
    return await anyio.to_thread.run_sync(lambda: do_get_context(chunk_id, pages_before, pages_after))


@mcp.tool()
async def get_item(item_key: str, offset: int = 0) -> str:
    """Full record of one library item: metadata, attachments (OCR status, pages where the user drew ink), and
    ALL of the user's annotations and notes on it in page order, each with its citation line. Long lists are
    paged: call again with next_offset. Annotation and note texts are the USER's words, except "Highlighted:"
    lines, which quote the source."""
    return await anyio.to_thread.run_sync(lambda: do_get_item(item_key, offset))


@mcp.tool()
async def list_annotations(tag: str | None = None, item_key: str | None = None, has_comment: bool | None = None,
                           source_type: str | None = None, limit: int = 200, offset: int = 0) -> str:
    """Browse the user's own annotations and notes across the whole library, e.g. everything tagged
    "intertextuality". tag matches the annotation's own tags (case-insensitive). has_comment=True keeps only
    annotations with a "My comment:" (and child notes). source_type: zotero_highlight, zotero_note,
    adobe_annotation or child_note. Results are sorted by source and page, with total_matching and next_offset
    for paging. These are the USER's annotations: comments, notes and tags are the user's words."""
    return await anyio.to_thread.run_sync(lambda: do_list_annotations(tag, item_key, has_comment, source_type,
                                                                      limit, offset))


@mcp.tool()
async def find_items(query: str, limit: int = 25) -> str:
    """Check whether a source is in the user's library at all: matches author/editor names, title words, year
    and publication (case- and accent-insensitive; every word must match). Returns item keys, citation lines
    and how much of each item is indexed. If nothing matches, say plainly that the source is not in the library."""
    return await anyio.to_thread.run_sync(lambda: do_find_items(query, limit))


async def serve() -> None:
    from mcp.server.stdio import stdio_server
    stdin = anyio.wrap_file(io.TextIOWrapper(os.fdopen(_MCP_STDIN_FD, "rb"), encoding="utf-8"))
    stdout = anyio.wrap_file(io.TextIOWrapper(os.fdopen(_MCP_STDOUT_FD, "wb"), encoding="utf-8"))
    async with stdio_server(stdin=stdin, stdout=stdout) as (read_stream, write_stream):
        await mcp._mcp_server.run(read_stream, write_stream, mcp._mcp_server.create_initialization_options())


def setup_logging() -> None:
    DB_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)
    err = logging.StreamHandler(sys.stderr)
    err.setFormatter(fmt)
    err.setLevel(logging.WARNING)
    root = logging.getLogger()
    root.handlers[:] = [file_handler, err]
    root.setLevel(logging.INFO)
    for noisy in ("httpx", "chromadb", "sentence_transformers", "urllib3", "mcp", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


if __name__ == "__main__":
    setup_logging()
    log.info("Starting server (pid %d)", os.getpid())
    start_background()                     # index + model load while the client connects
    anyio.run(serve)
