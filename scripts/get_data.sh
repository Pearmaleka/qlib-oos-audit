#!/usr/bin/env bash
# Download the exact data release used in the paper and verify it.
set -euo pipefail
TAG=2026-11-14
SHA=3c4a631f5f053abd77d29088b26b7b13108cb4782157c4310f5258d154b9ba7b
DEST=${1:-$HOME/data}
mkdir -p "$DEST"
cd "$DEST"
URL=https://github.com/chenditc/investment_data/releases/download/$TAG
curl -fsSL -o qlib_bin.tar.gz "$URL/qlib_bin.tar.gz"
curl -fsSL -o qlib_bin.manifest.json "$URL/qlib_bin.manifest.json"
echo "$SHA  qlib_bin.tar.gz" | sha256sum -c -
tar -xzf qlib_bin.tar.gz
echo "Data in $DEST/qlib_bin; set OOSAUDIT_QLIB_DIR=$DEST/qlib_bin"
