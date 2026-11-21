"""Read one sheet of an .xlsx file with the standard library only (no new dependency).

Enough for MU-Glioma-Post's clinical spreadsheet: shared strings, inline strings and numbers,
cells addressed by their "A1"-style reference so empty cells stay in place.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
       "pr": "http://schemas.openxmlformats.org/package/2006/relationships"}


def _column(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _number(text: str):
    value = float(text)
    return int(value) if value.is_integer() else value


def read_sheet(path: str | Path, sheet: str) -> list[list]:
    """Rows of ``sheet`` as lists (None for an empty cell); the first row is the header."""
    with zipfile.ZipFile(path) as z:
        book = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in rels.findall("pr:Relationship", _NS)}
        rid = next((s.get(f"{{{_NS['r']}}}id") for s in book.find("m:sheets", _NS)
                    if s.get("name") == sheet), None)
        if rid is None:
            raise KeyError(f"{path}: no sheet named {sheet!r}")
        target = targets[rid].lstrip("/")
        target = target if target.startswith("xl/") else f"xl/{target}"
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
        root = ET.fromstring(z.read(target))
    rows = []
    for row in root.find("m:sheetData", _NS).findall("m:row", _NS):
        cells = {}
        for c in row.findall("m:c", _NS):
            kind, v = c.get("t"), c.find("m:v", _NS)
            if kind == "s" and v is not None:
                value = shared[int(v.text)]
            elif kind == "inlineStr":
                value = "".join(t.text or "" for t in c.iter(f"{{{_NS['m']}}}t"))
            elif kind in ("str", "e") and v is not None:
                value = v.text
            elif kind == "b" and v is not None:
                value = v.text == "1"
            elif v is not None and v.text is not None:
                value = _number(v.text)
            else:
                value = None
            cells[_column(c.get("r"))] = value
        width = max(cells) + 1 if cells else 0
        rows.append([cells.get(i) for i in range(width)])
    width = max((len(r) for r in rows), default=0)
    return [r + [None] * (width - len(r)) for r in rows]
