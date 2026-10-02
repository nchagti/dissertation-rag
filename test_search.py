"""Call the rag_server tools directly (no MCP) and print the top results.

    .venv\\Scripts\\python.exe test_search.py
"""
import asyncio
import json
import sys
import time

import rag_server as rs

TOP = 5


def show(title: str, raw: str, list_key: str, top: int = TOP, width: int = 300) -> dict:
    data = json.loads(raw)
    rows = data.get(list_key, [])
    head = {k: v for k, v in data.items() if k not in (list_key,)}
    print(f"\n{'=' * 100}\n{title}\n{json.dumps(head, ensure_ascii=False)}  | response {len(raw):,} chars")
    for i, r in enumerate(rows[:top], 1):
        flags = []
        if r.get("ocr"):
            flags.append("OCR")
        if r.get("my_annotations_on_page"):
            flags.append(f"my annotations on page: {r['my_annotations_on_page']}")
        if r.get("page_annotation_tags"):
            flags.append(f"page tags: {r['page_annotation_tags']}")
        if r.get("has_ink_on_page"):
            flags.append("ink")
        if r.get("annotation_tags"):
            flags.append(f"tags: {r['annotation_tags']}")
        text = " ".join((r.get("text") or r.get("title") or "").split())
        print(f"\n{i}. {r.get('citation')}  [{r.get('source_type', r.get('item_type'))}"
              f"{', ' + r['language'] if r.get('language') else ''}]  {r.get('match', '')}")
        print(f"   {r.get('chunk_id') or r.get('item_key')}  {' | '.join(flags)}")
        print(f"   {text[:width]}{'…' if len(text) > width else ''}")
    return data


async def main() -> None:
    t0 = time.time()
    rs.start_background()
    rs.index()
    print(f"index loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    t = time.time()
    a = await rs.search("Mussato's five pairs of Senecan tragedies in the argumentum", mode="semantic")
    print(f"(first semantic query incl. model load: {time.time() - t:.1f}s on {rs.EMBEDDER.device})")
    show('a) semantic: "Mussato\'s five pairs of Senecan tragedies in the argumentum"', a, "results")

    show('b) keyword: "Vat. Lat. 2962"', await rs.search("Vat. Lat. 2962", mode="keyword"), "results")

    t = time.time()
    c = await rs.search("Ezzelino and the centaurs in Dante's Inferno", mode="hybrid")
    print(f"(hybrid query: {time.time() - t:.2f}s)")
    show("c) hybrid: \"Ezzelino and the centaurs in Dante's Inferno\"", c, "results")

    show('d) list_annotations(tag="intertextuality")', await rs.list_annotations(tag="intertextuality"),
         "annotations", width=160)

    show('e) find_items("Paratore")', await rs.find_items("Paratore"), "items", top=10)

    # extra checks on the other tools and edge cases
    print(f"\n{'=' * 100}\nextra checks")
    hit = json.loads(c)["results"][0]["chunk_id"]
    ctx = json.loads(await rs.get_context(hit))
    print("get_context", hit, "->", [(p.get("pdf_page"), p.get("citation"), len(p.get("text", "")))
                                    for p in ctx.get("pages", ctx.get("neighboring_chunks", ctx.get("chunks", [])))])
    item = json.loads(await rs.get_item("X6S3KQ52"))
    print("get_item X6S3KQ52:", item["citation"], "|", item["my_annotations_and_notes_total"], "annotations/notes;",
          "first:", item["annotations_and_notes"][0]["citation"], "|", item["attachments"][0]["pages_with_my_ink"][:5])
    big = await rs.get_item("W4JNUZST")
    print("get_item W4JNUZST (1,339 annotations):", len(big), "chars,", json.loads(big).get("truncated"))
    kw = json.loads(await rs.search('"Ecerinis" NEAR("Seneca" "tragedia", 10)', mode="keyword", n=3))
    print("keyword NEAR query hits:", kw["hits"], kw.get("note"))
    filt = json.loads(await rs.search("tragedia", n=3, languages=["it"], tags=["intertextuality"], year_from=1990,
                                      year_to=1999))
    print("filtered search:", [(r["citation"], r["source_type"], r["language"]) for r in filt["results"]])
    none = json.loads(await rs.find_items("Xyzzy Nonexistent"))
    print("find_items miss:", none["total_matching"], "|", none.get("note"))


if __name__ == "__main__":
    asyncio.run(main())
