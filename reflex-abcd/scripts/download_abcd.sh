#!/usr/bin/env bash
# Fetch ABCD (spec 3.1/3.2) into the directory given by data.abcd_dir.
#
# The data is ALREADY DOWNLOADED for this project; see configs/default.yaml's
# data.abcd_dir. This script exists so the repo is reproducible from scratch.
#
# Layout produced (what scripts/inspect_abcd.py expects):
#   $DEST/data/abcd_v1.1.json      (~122 MB unzipped)
#   $DEST/data/utterances.json
#   $DEST/data/ontology.json
#   $DEST/data/kb.json
#   $DEST/data/guidelines.json
#   $DEST/utils/{evaluate,process,load,help,arguments}.py
#
# utils/evaluate.py holds the official AST/CDS metrics that spec 13 forbids
# reimplementing. It imports `components.systems`, which the upstream repo DOES
# ship -- clone the whole repo and you get it. If you only vendor utils/, see
# the shim documented in reflex.contracts.load_official_metrics.

set -euo pipefail

DEST="${1:-data/abcd}"
REPO="https://github.com/asappresearch/abcd"

echo "Cloning ${REPO} into ${DEST} ..."
mkdir -p "$(dirname "${DEST}")"
if [ -d "${DEST}/.git" ]; then
  echo "${DEST} already a git clone; pulling."
  git -C "${DEST}" pull --ff-only
else
  git clone --depth 1 "${REPO}" "${DEST}"
fi

if [ -f "${DEST}/data/abcd_v1.1.json.gz" ]; then
  echo "Unzipping abcd_v1.1.json.gz ..."
  gunzip -k "${DEST}/data/abcd_v1.1.json.gz"
fi

echo
echo "Checking expected files:"
for f in data/abcd_v1.1.json data/utterances.json data/ontology.json \
         data/kb.json data/guidelines.json utils/evaluate.py; do
  if [ -e "${DEST}/${f}" ]; then
    printf '  ok      %s\n' "${f}"
  else
    printf '  MISSING %s\n' "${f}"
  fi
done

echo
echo "Record the repository LICENSE in README.md before any public use (spec 3.5):"
ls -1 "${DEST}" | grep -i -E 'licen|copying' || echo "  (no LICENSE file found in the clone)"

echo
echo "Now set data.abcd_dir in configs/default.yaml to: $(cd "${DEST}" && pwd)"
echo "Then run: python scripts/inspect_abcd.py"
