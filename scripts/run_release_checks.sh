#!/usr/bin/env bash
# Owner-only, fail-closed verification. Publish sanitized check evidence to GitHub.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PYTHON='/mnt/HDD/AliWorks/HCI Tagging Database/HCI/bin/python3'
BRANCH='fix/reproduction-imports-and-preflight-20261026'
if [[ ! -x "$PYTHON" ]]; then
  printf 'BLOCKED: exact HCI venv required: %s\n' "$PYTHON" >&2
  exit 2
fi
cd "$ROOT"
[[ "$(git branch --show-current)" == "$BRANCH" ]] || {
  echo 'BLOCKED: dedicated repair-branch checkout required' >&2; exit 2;
}
[[ -z "$(git status --porcelain)" ]] || {
  echo 'BLOCKED: dirty worktree; preserve changes first' >&2; exit 2;
}
mkdir -p reports/release
LOG="reports/release/source_check_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
printf 'Interpreter: %s\nBranch: %s\nNo EEG data read by this procedure.\n' "$PYTHON" "$BRANCH" | tee "$LOG"
set +e
"$PYTHON" scripts/release_preflight.py --tier source 2>&1 | tee -a "$LOG"
SOURCE_RC=$?
"$PYTHON" -m unittest discover -s tests -p test_release_preflight.py -v 2>&1 | tee -a "$LOG"
TEST_RC=$?
"$PYTHON" scripts/release_preflight.py --tier replay 2>&1 | tee -a "$LOG"
REPLAY_RC=$?
set -e
printf 'Exit codes: SOURCE=%s TESTS=%s REPLAY=%s\n' "$SOURCE_RC" "$TEST_RC" "$REPLAY_RC" | tee -a "$LOG"
git add -- "$LOG"
git commit -m 'evidence: owner source check and replay prerequisite audit'
if ! git push origin "$BRANCH"; then
  echo "NOT_SYNCED: git push failed; local evidence remains in $LOG" >&2
  exit 3
fi
LOCAL_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git ls-remote origin "refs/heads/$BRANCH" | cut -f1)"
if [[ "$LOCAL_SHA" != "$REMOTE_SHA" || -z "$REMOTE_SHA" ]]; then
  echo 'NOT_SYNCED: remote SHA differs from local HEAD' >&2
  exit 3
fi
echo "VERIFIED_SYNC: $REMOTE_SHA"
if (( SOURCE_RC != 0 || TEST_RC != 0 )); then
  echo 'SOURCE_OR_TEST_GATE_FAILED; see committed log' >&2
  exit 2
fi
if (( REPLAY_RC != 0 )); then
  echo 'REPLAY_BLOCKED_PRIVATE_ARTIFACTS: source checks passed; full reproduction not proven' >&2
  exit 2
fi
echo 'SOURCE_GATE_PASSED; input presence alone is not scientific replication'
