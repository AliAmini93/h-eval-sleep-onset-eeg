#!/usr/bin/env bash
# Build the manuscript PDF from paper/latex and paper/figures. No model is trained or evaluated.
#
#   PYTHON=/path/to/existing/python3 bash scripts/paper/build_paper.sh
#
# PYTHON must point to an existing interpreter that satisfies requirements-paper.txt (no environment is created and
# nothing is installed). pdflatex and bibtex (TeX Live) are required. Output: build/h_eval_paper_<mode>.pdf
#   H_EVAL_PAPER_MODE=anonymous (default)  : no repository link (double-blind safe)
#   H_EVAL_PAPER_MODE=author-review        : requires H_EVAL_CODE_URL=https://github.com/<owner>/h-eval-sleep-onset-eeg
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:?Set PYTHON to an existing Python interpreter (see requirements-paper.txt)}"
[ -x "$PYTHON" ] || { echo "STOP: PYTHON is not an executable interpreter: $PYTHON" >&2; exit 13; }
"$PYTHON" -c 'import sys, numpy, matplotlib, PIL; print("Python", sys.version.split()[0], "| numpy", numpy.__version__, "| matplotlib", matplotlib.__version__)' \
  || { echo "STOP: numpy/matplotlib/Pillow missing in $PYTHON" >&2; exit 14; }
for tool in pdflatex bibtex; do command -v "$tool" >/dev/null || { echo "STOP: $tool not found" >&2; exit 15; }; done

FROZEN_TABLE_BLOB=f4ac863b0130ef5c8c9253d14bbc2e375b550cd9
blob=$("$PYTHON" -c 'import hashlib,sys; d=open(sys.argv[1],"rb").read(); print(hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest())' "$ROOT/paper/latex/tables_generated.tex")
[ "$blob" = "$FROZEN_TABLE_BLOB" ] || { echo "STOP: paper/latex/tables_generated.tex differs from the frozen table blob" >&2; exit 10; }

# 1) Regenerate tables and figures from the aggregate snapshots and require identity with the committed files.
"$PYTHON" "$ROOT/scripts/paper/regenerate_tables_v8r2.py"
"$PYTHON" "$ROOT/scripts/paper/make_figures_v8r5.py" >/dev/null
(cd "$ROOT/paper/figures" && sha256sum --quiet -c FIGURE_SHA256.txt) || { echo "STOP: regenerated figures differ" >&2; exit 11; }
"$PYTHON" "$ROOT/scripts/paper/verify_final_claims_v8r5.py" >/dev/null

# 2) LaTeX build in a temporary directory.
MODE="${H_EVAL_PAPER_MODE:-anonymous}"
CODE_URL="${H_EVAL_CODE_URL:-}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/figures" "$ROOT/build"
cp "$ROOT"/paper/latex/{main.tex,checklist_filled.tex,tables_generated.tex,references.bib,neurips_2026.sty} "$TMP/"
cp "$ROOT"/paper/figures/v8r5_fig*.pdf "$TMP/figures/"
case "$MODE" in
  author-review)
    [[ "$CODE_URL" =~ ^https://github\.com/[A-Za-z0-9_.-]+/h-eval-sleep-onset-eeg$ ]] \
      || { echo "STOP: author-review mode needs H_EVAL_CODE_URL=https://github.com/<owner>/h-eval-sleep-onset-eeg" >&2; exit 23; }
    printf '\\def\\HEVALCodeAuthorURL{%s}\n\\def\\HEVALCodeAuthorVisible{%s}\n' "$CODE_URL" "${CODE_URL#https://}" > "$TMP/code_access_config.tex" ;;
  anonymous)
    if [ -n "$CODE_URL" ]; then
      [[ "$CODE_URL" == https://* && "$CODE_URL" != *"github.com/"* ]] \
        || { echo "STOP: anonymous mode cannot link an identifying GitHub URL" >&2; exit 24; }
      printf '\\def\\HEVALCodeBlindURL{%s}\n' "$CODE_URL" > "$TMP/code_access_config.tex"
    fi ;;
  *) echo "STOP: H_EVAL_PAPER_MODE must be anonymous or author-review" >&2; exit 25 ;;
esac
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1792886400}" FORCE_SOURCE_DATE=1
(cd "$TMP" && pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null && bibtex main >/dev/null \
  && pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null)
n=3
while grep -q "Rerun to get cross-references right" "$TMP/main.log" && [ "$n" -lt 6 ]; do
  n=$((n + 1)); (cd "$TMP" && pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null)
done
grep -q "Rerun to get cross-references right" "$TMP/main.log" && { echo "STOP: cross-references not stable" >&2; exit 12; }
OUTPDF="$ROOT/build/h_eval_paper_${MODE}.pdf"
cp "$TMP/main.pdf" "$OUTPDF"
echo "LaTeX passes: $n; $(pdfinfo "$OUTPDF" 2>/dev/null | grep Pages || true)"
(cd "$ROOT/build" && sha256sum "$(basename "$OUTPDF")")
