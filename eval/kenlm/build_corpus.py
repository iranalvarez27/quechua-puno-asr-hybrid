"""
Build a plain-text training corpus for a Puno Quechua (qxp) KenLM language model.

Pulls normalized transcripts from every text source in this repo that is NOT
a test split, using the same normalize() as training/scripts/build_finetune_manifests.py
so the LM's text distribution matches what the acoustic model was fine-tuned on.

Sources (toggle with --sources):
  scripted     data/splits/validated_scripted/{train,dev}.tsv                    (col: sentence)
  spontaneous  data/splits/validated_spontaneous/{train_spontaneous,dev_spontaneous}.tsv (col: sentence)
  silver       data/splits/silver_spontaneous/{train,dev}.tsv                     (col: sentence) [auto-transcribed, noisier]
  additional   data/additional_data/additional_data_qxp.tsv                       (col: sentence) [not split, bonus text]
  curated      data/shared-task/validated_sentences_curated.tsv                  (col: sentence_curated) [best orthography]

NOTE on leakage: the scripted dataset draws from a fixed pool of ~2,051 sentences
read by many speakers, so the same sentence text can legitimately appear across
train/dev/test clips regardless of what this script does. It only avoids reading
files literally named as test splits -- it cannot remove that inherent pool overlap.
Keep that caveat in mind when interpreting any WER gain on the scripted test set.

Usage (from project root):
  python eval/kenlm/build_corpus.py
  python eval/kenlm/build_corpus.py --sources scripted spontaneous curated
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SOURCES = {
    "scripted": [
        (ROOT / "data/splits/validated_scripted/train.tsv", "sentence"),
        (ROOT / "data/splits/validated_scripted/dev.tsv", "sentence"),
    ],
    "spontaneous": [
        (ROOT / "data/splits/validated_spontaneous/train_spontaneous.tsv", "sentence"),
        (ROOT / "data/splits/validated_spontaneous/dev_spontaneous.tsv", "sentence"),
    ],
    "silver": [
        (ROOT / "data/splits/silver_spontaneous/train.tsv", "sentence"),
        (ROOT / "data/splits/silver_spontaneous/dev.tsv", "sentence"),
    ],
    "additional": [
        (ROOT / "data/additional_data/additional_data_qxp.tsv", "sentence"),
    ],
    "curated": [
        (ROOT / "data/shared-task/validated_sentences_curated.tsv", "sentence_curated"),
    ],
}


def normalize(text: str) -> str:
    """Lowercase and strip punctuation -- mirrors build_finetune_manifests.py:normalize()."""
    text = text.lower()
    return "".join(c for c in text if c not in "?!¿¡.,").strip()


def load_source(name: str, exclude_dev: bool = False) -> list[str]:
    lines = []
    for path, col in SOURCES[name]:
        if exclude_dev and path.stem.startswith("dev"):
            print(f"  skipping dev file {path.relative_to(ROOT)}")
            continue
        if not path.exists():
            print(f"  WARNING: missing {path.relative_to(ROOT)}, skipping")
            continue
        df = pd.read_csv(path, sep="\t", dtype=str, usecols=[col])
        n_before = len(lines)
        for raw in df[col].dropna():
            text = normalize(str(raw))
            if text:
                lines.append(text)
        print(f"  {path.relative_to(ROOT)}: {len(lines) - n_before} lines")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "corpus.txt")
    parser.add_argument("--exclude-dev", action="store_true",
                        help="Leave out dev files and the curated list (it holds every scripted "
                             "sentence, dev included) so the LM can be tuned on valid without leakage")
    args = parser.parse_args()

    sources = [s for s in args.sources if not (args.exclude_dev and s == "curated")]
    all_lines = []
    print("Building KenLM training corpus...\n")
    for name in sources:
        print(f"[{name}]")
        lines = load_source(name, args.exclude_dev)
        all_lines.extend(lines)
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line + "\n")

    # --- Stats, so content can be sanity-checked before training the LM ---
    word_counts = Counter(w for line in all_lines for w in line.split())
    total_words = sum(word_counts.values())
    hapax = sum(1 for c in word_counts.values() if c == 1)

    print("=" * 60)
    print(f"Total lines:      {len(all_lines)}")
    print(f"Total words:      {total_words}")
    print(f"Distinct words:   {len(word_counts)}")
    if word_counts:
        print(f"Hapax (count==1): {hapax} ({hapax / len(word_counts) * 100:.1f}% of vocab)")
    print(f"Top 15 words:     {', '.join(w for w, _ in word_counts.most_common(15))}")
    print(f"\nWrote corpus -> {args.output.resolve().relative_to(ROOT)}")


if __name__ == "__main__":
    main()
