"""Render the CHMT-Q DOCX through Microsoft Word and rasterize every PDF page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fitz
import win32com.client


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--png-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    source = args.input.resolve(strict=True)
    pdf = args.pdf.resolve()
    png_dir = args.png_dir.resolve()
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    for old in png_dir.glob("page-*.png"):
        old.unlink()

    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    document = None
    try:
        document = word.Documents.Open(
            str(source),
            ConfirmConversions=False,
            ReadOnly=True,
            AddToRecentFiles=False,
            Visible=False,
        )
        document.Repaginate()
        document.Fields.Update()
        document.ExportAsFixedFormat(
            OutputFileName=str(pdf),
            ExportFormat=17,
            OpenAfterExport=False,
            OptimizeFor=0,
            Range=0,
            Item=0,
            IncludeDocProps=True,
            KeepIRM=True,
            CreateBookmarks=1,
            DocStructureTags=True,
            BitmapMissingFonts=True,
            UseISO19005_1=False,
        )
    finally:
        if document is not None:
            document.Close(SaveChanges=False)
        word.Quit()

    rendered = fitz.open(pdf)
    scale = args.dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    for index, page in enumerate(rendered, start=1):
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        pixmap.save(png_dir / f"page-{index:03d}.png")
    pages = len(rendered)
    rendered.close()
    print(json.dumps({"pdf": str(pdf), "pages": pages, "png_dir": str(png_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
