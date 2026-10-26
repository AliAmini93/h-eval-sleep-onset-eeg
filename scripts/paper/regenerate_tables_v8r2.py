#!/usr/bin/env python3
"""Reproduce presentation-only edits of v8r2 tables from unchanged aggregate result files."""
import importlib.util
import json
import tempfile
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "scripts/paper/make_tables_paper_v8r1.py"
CHANGES = ROOT / "scripts/paper/table_edits_v8r2.json"
TARGET = ROOT / "paper/latex/tables_generated.tex"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("v8r1_tables_for_editorial", ORIGINAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with tempfile.TemporaryDirectory() as tmp:
        mod.OUT = Path(tmp) / "original_tables.tex"
        mod.main()
        rows = mod.OUT.read_text().split("\n")
    changes = json.loads(CHANGES.read_text())
    for ch in changes:
        i = ch["line"]-1
        if rows[i] != ch["old"]:
            raise RuntimeError(f"Source result/table-generator drift at line {ch['line']}")
        rows[i] = ch["new"]
    reconstructed = "\n".join(rows)
    current = TARGET.read_text()
    if current != reconstructed and not args.write:
        raise RuntimeError("Generated table mismatch; do not publish before investigating")
    if args.write:
        TARGET.write_text(reconstructed)
    print(f"OK: {len(changes)} presentation edits reproduced from frozen aggregate results.")

if __name__ == "__main__":
    main()
