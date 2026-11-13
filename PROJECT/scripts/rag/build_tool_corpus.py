#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build RAG tool corpus from PROJECT/configs/tool_catalog.jsonl."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catalog.export_tool_descriptions import main


if __name__ == "__main__":
    raise SystemExit(main())
