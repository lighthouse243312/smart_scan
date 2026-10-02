#!/bin/bash
# Runs the app's real erase pipeline on a folder of photos on a connected iPhone and pulls the
# results back (see tool/batch_erase_main.dart).
#
#   tool/run_batch.sh <image folder> <output folder> [--build]
#
# --build rebuilds + installs the harness first (needed after native changes).
set -euo pipefail
IN="$1"; OUT="$2"; BUILD="${3:-}"
BUNDLE=com.beacon.smartScan
# a non-login shell may lack flutter on PATH: use the project's own SDK (Generated.xcconfig)
if ! command -v flutter >/dev/null; then
  FR=$(sed -n 's/^FLUTTER_ROOT=//p' "$(dirname "$0")/../ios/Flutter/Generated.xcconfig" 2>/dev/null)
  [ -x "$FR/bin/flutter" ] && export PATH="$FR/bin:$PATH"
fi
cd "$(dirname "$0")/.."
# DEVICE=<identifier> overrides; otherwise the first connected iPhone (a wired phone can also
# show as "available (paired)" — set DEVICE then)
DEVICE=${DEVICE:-$(xcrun devicectl list devices 2>/dev/null | awk '/connected/ && /iPhone/ {for(i=1;i<=NF;i++) if ($i ~ /^[0-9A-F]{8}-/) print $i}' | head -1)}
[ -n "$DEVICE" ] || { echo "no connected iPhone"; exit 1; }
copy_to()   { xcrun devicectl device copy to   --device "$DEVICE" --domain-type appDataContainer --domain-identifier $BUNDLE --source "$1" --destination "$2" >/dev/null; }
copy_from() { xcrun devicectl device copy from --device "$DEVICE" --domain-type appDataContainer --domain-identifier $BUNDLE --source "$1" --destination "$2" >/dev/null 2>&1; }
list_out()  { xcrun devicectl device info files --device "$DEVICE" --domain-type appDataContainer --domain-identifier $BUNDLE --subdirectory Documents/batch_out 2>/dev/null | awk '{print $1}' | grep -E '\.png$|^\.done$' || true; }

if [ "$BUILD" = "--build" ]; then
  LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 flutter build ios --profile -t tool/batch_erase_main.dart
  xcrun devicectl device install app --device "$DEVICE" build/ios/iphoneos/Runner.app >/dev/null
fi
xcrun devicectl device process launch --device "$DEVICE" --terminate-existing $BUNDLE >/dev/null
sleep 3
n=0
for f in "$IN"/*; do
  case "$(echo "$f" | tr "[:upper:]" "[:lower:]")" in *.jpg|*.jpeg|*.png|*.heic|*.webp) ;; *) continue ;; esac
  n=$((n+1)); name=$(printf 'p%02d' $n)
  ext="${f##*.}"
  copy_to "$f" "Documents/batch_in/$name.$ext"
  echo "$name ← $(basename "$f")"
done
TOKEN="run-$(date +%s)-$$"
TMP=$(mktemp -d); echo "$TOKEN" > "$TMP/.go"; copy_to "$TMP/.go" Documents/batch_in/.go
echo "processing $n images…"
# wait for THIS run's .done (an older one may still be there until the app starts)
until copy_from Documents/batch_out/.done "$TMP/.done" && [ "$(cat "$TMP/.done")" = "$TOKEN" ]; do sleep 5; done
mkdir -p "$OUT"
for f in $(list_out | grep png); do copy_from "Documents/batch_out/$f" "$OUT/$f"; done
echo "done → $OUT"
