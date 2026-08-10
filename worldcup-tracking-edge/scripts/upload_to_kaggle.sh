#!/usr/bin/env bash
# Upload the SoccerNet train split to Kaggle as a PRIVATE dataset.
#
# Prereq: ~/.kaggle/kaggle.json (kaggle.com -> Settings -> API -> Create New Token).
#
# Uploads the archive rather than 37k loose jpgs: Kaggle expands archives on
# ingest, and one large file uploads far faster and fails far less often than
# tens of thousands of small ones.
#
# The dataset is created PRIVATE (kaggle's default; we never pass --public).
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$PROJ/kaggle_upload"
DATA="$PROJ/data/soccernet/tracking-2023"
SLUG="soccernet-tracking-2023-train"

command -v kaggle >/dev/null || { echo "kaggle CLI not on PATH (uv tool install kaggle)"; exit 1; }
[ -f "$HOME/.kaggle/kaggle.json" ] || { echo "missing ~/.kaggle/kaggle.json"; exit 1; }
chmod 600 "$HOME/.kaggle/kaggle.json"

USER_NAME="$(python3 -c 'import json,pathlib;print(json.loads(pathlib.Path.home().joinpath(".kaggle/kaggle.json").read_text())["username"])')"
echo "kaggle user: $USER_NAME"

# Stamp the real username into the metadata.
python3 - "$STAGE/dataset-metadata.json" "$USER_NAME/$SLUG" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]); d = json.loads(p.read_text())
d["id"] = sys.argv[2]
p.write_text(json.dumps(d, indent=2))
print("dataset id:", d["id"])
PY

# Hard-link the archive in: same filesystem, so this costs no extra disk.
[ -f "$DATA/train.zip" ] || { echo "missing $DATA/train.zip"; exit 1; }
[ -e "$STAGE/train.zip" ] || ln "$DATA/train.zip" "$STAGE/train.zip"

# COCO annotations built by snt-prepare, if they exist yet.
if [ -d "$PROJ/output/coco" ]; then
  mkdir -p "$STAGE/coco"
  cp -f "$PROJ"/output/coco/*.json "$STAGE/coco/" 2>/dev/null || true
fi

echo "staging contents:"; du -sh "$STAGE"/* 2>/dev/null

# -t keeps our .txt/.csv ground-truth files intact: without it the CLI helpfully
# "converts tabular files to CSV", which would rewrite gt.txt.
if kaggle datasets status "$USER_NAME/$SLUG" >/dev/null 2>&1; then
  echo "dataset exists -> pushing a new version"
  kaggle datasets version -p "$STAGE" -m "update $(date -u +%Y-%m-%dT%H:%MZ)" -t -r skip
else
  echo "creating PRIVATE dataset"
  kaggle datasets create -p "$STAGE" -t -r skip
fi

echo
echo "Done. Verify it is PRIVATE at https://www.kaggle.com/datasets/$USER_NAME/$SLUG/settings"
