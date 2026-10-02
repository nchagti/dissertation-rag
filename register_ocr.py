r"""Register a hand-made OCR'd PDF in ocr_manifest.csv (runs no OCR).

First copy the PDF to ocr_output\<attachment key>.pdf, then:
    .venv\Scripts\python.exe register_ocr.py <attachment key>
Zotero must be open; it is only read (GET).
"""
import argparse
import datetime
import sys

import requests

from audit_zotero import ZoteroAPI, first_author, year_of
from run_ocr import OUT_DIR, ROOT, update_manifest

ap = argparse.ArgumentParser()
ap.add_argument("attachment_key")
ap.add_argument("--languages", default="lat")
ap.add_argument("--note", default="Ghostscript 300 dpi; cover page removed; --skip-text")
opts = ap.parse_args()

pdf = OUT_DIR / f"{opts.attachment_key}.pdf"
if not pdf.is_file():
    sys.exit(f"Copy the OCR'd PDF to {pdf} first")


def fetch(api: ZoteroAPI, key: str, what: str) -> dict:
    # items/<key> returns exactly that item; items?itemKey=<key> also returns its children.
    try:
        return api.get(f"items/{key}").json()
    except requests.HTTPError as e:
        sys.exit(f"{what} {key} not found in Zotero (HTTP {e.response.status_code})")


api = ZoteroAPI()
api.check_connection()
att = fetch(api, opts.attachment_key, "Attachment")
if att["data"]["itemType"] != "attachment":
    sys.exit(f"{opts.attachment_key} is a {att['data']['itemType']}, not an attachment")
parent = att["data"].get("parentItem")
if not parent:
    sys.exit(f"Attachment {opts.attachment_key} has no parent item")
item = fetch(api, parent, "Parent item")
author = first_author(item["data"])

update_manifest(dict(
    attachment_key=opts.attachment_key, item_key=item["key"], author=author,
    year=year_of(item), source_file="processed by hand",
    ocr_pdf=str(pdf.relative_to(ROOT)), sidecar_txt="", languages=opts.languages,
    ocrmypdf_args=opts.note, date=datetime.date.today().isoformat()))
print(f"Registered {pdf.name}: item {item['key']} ({item['data']['itemType']}), "
      f"{author} {year_of(item)}, \"{item['data'].get('title', '')}\"")
