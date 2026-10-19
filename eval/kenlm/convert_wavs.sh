#!/usr/bin/env bash
# Convert the MP3s listed in the valid / test / test_spont manifests to
# 16 kHz mono WAV under data/wav/{scripted,spontaneous}/ (what the manifests expect).
# Run from project root on the host (WSL), needs ffmpeg:  sudo apt install -y ffmpeg
#   bash eval/kenlm/convert_wavs.sh [num_workers] [subset ...]
#   bash eval/kenlm/convert_wavs.sh 8 valid
set -euo pipefail

WORKERS=${1:-8}
shift || true
SUBSETS=("$@")
[ ${#SUBSETS[@]} -eq 0 ] && SUBSETS=(valid test test_spont)

MANIFEST_DIR="data/manifests/finetune/qxp_v2"
SCRIPTED_DIR=$(ls -d data/train_data/scripted/cv-corpus-*/qxp/clips | head -1)
SPONT_DIR=$(ls -d data/train_data/spontaneous/sps-corpus-*-qxp/audios | head -1)
echo "Scripted MP3s:    $SCRIPTED_DIR"
echo "Spontaneous MP3s: $SPONT_DIR"

mkdir -p data/wav/scripted data/wav/spontaneous

for subset in "${SUBSETS[@]}"; do
    echo "=== $subset ==="
    tail -n +2 "$MANIFEST_DIR/$subset.tsv" | cut -f1 | while read -r rel; do
        wav="data/wav/$rel"
        [ -f "$wav" ] && continue
        base=$(basename "$rel" .wav)
        case "$rel" in
            scripted/*)    echo "$SCRIPTED_DIR/$base.mp3 $wav" ;;
            spontaneous/*) echo "$SPONT_DIR/$base.mp3 $wav" ;;
        esac
    done | xargs -P "$WORKERS" -L1 bash -c '
        [ -f "$0" ] || { echo "MISSING mp3: $0" >&2; exit 0; }
        ffmpeg -y -i "$0" -ar 16000 -ac 1 -sample_fmt s16 "$1" -loglevel error
    '
done

echo "Done."
for subset in "${SUBSETS[@]}"; do
    missing=$(tail -n +2 "$MANIFEST_DIR/$subset.tsv" | cut -f1 | while read -r rel; do [ -f "data/wav/$rel" ] || echo x; done | wc -l)
    echo "  $subset: $missing WAVs still missing"
done
