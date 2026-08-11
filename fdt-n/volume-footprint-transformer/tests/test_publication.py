from __future__ import annotations

import pytest

from daxalgo_footprint_transformer.publication import (
    REQUIRED_FIGURES,
    _break_hash,
    _inline,
    _validate_manuscript,
    parse_front_matter,
)


def _manuscript() -> str:
    directives = "\n".join(
        f"{{{{FIGURE:{name}|Figure. Synthetic caption.}}}}" for name in sorted(REQUIRED_FIGURES)
    )
    return f"""# Synthetic manuscript

2.68% above a train-only direct ridge model.
The model does not outperform the train-only direct ridge model.
The intervals are not as calibrated quantiles.
The output does not define a coherent joint candle distribution.
No result in this paper supports deployment.

{directives}
"""


def test_parse_and_validate_anonymous_front_matter() -> None:
    source = """---
title: Synthetic title
short_title: Synthetic
author: Anonymous author(s)
affiliation: Anonymous review manuscript
date: 29 July 2026
keywords: test; manuscript
---

""" + _manuscript()

    metadata, manuscript = parse_front_matter(source)

    assert metadata["title"] == "Synthetic title"
    _validate_manuscript(metadata, manuscript)


def test_validation_rejects_fabricated_author() -> None:
    metadata = {
        "title": "Synthetic title",
        "short_title": "Synthetic",
        "author": "Invented Author",
        "affiliation": "Invented University",
        "date": "29 July 2026",
        "keywords": "test",
    }

    with pytest.raises(ValueError, match="must not be fabricated"):
        _validate_manuscript(metadata, _manuscript())


def test_hash_breaks_remain_reportlab_markup() -> None:
    digest = "a" * 64

    assert "<br/>" in _break_hash(_inline(digest))
