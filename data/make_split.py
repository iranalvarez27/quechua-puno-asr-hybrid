#!/usr/bin/env python3
"""
Makes the train/dev/test split from Common Voice metadata (speaker, duration, gender)
Usage :
    python make_split.py --input metadata.tsv --output_dir ./splits \
        --train_ratio 0.8 --dev_ratio 0.1 --test_ratio 0.1
"""

import argparse
import numpy as np
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Common Voice TSV file")
    parser.add_argument("--output_dir", default=".", help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument(
        "--max_speaker_share",
        type=float,
        default=0.10,
        help="Max share of a single speaker (0-1) in dev/test",
    )
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()

    assert abs(args.train_ratio + args.dev_ratio + args.test_ratio - 1.0) < 1e-6, \
        "The ratios must add up to 1"
    
    if not os.path.isdir(args.output_dir):
       print("Creating output directory...")
       os.mkdir(args.output_dir)

    df = pd.read_csv(args.input, sep="\t")
    required_cols = {"client_id", "gender", "duration_ms"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in TSV : {missing}")

    df["gender"] = df["gender"].fillna("unknown").replace("", "unknown")

    total_h = df["duration_ms"].sum() / 3_600_000
    print(f"Corpus: {total_h:.2f} h, {len(df)} clips, "
          f"{df['client_id'].nunique()} speakers\n")

    # --- Aggregation by speaker ---
    speaker_stats = df.groupby("client_id").agg(
        duration_ms=("duration_ms", "sum"),
        n_clips=("duration_ms", "count"),
    ).reset_index()
    gender_map = df.groupby("client_id")["gender"].agg(lambda x: x.value_counts().index[0])
    speaker_stats["gender"] = speaker_stats["client_id"].map(gender_map)

    rng_seed = args.seed
    splits = ["train", "dev", "test"]
    ratios = {"train": args.train_ratio, "dev": args.dev_ratio, "test": args.test_ratio}
    speaker_split = {}

    # --- Allocation stratified by gender ---
    for gender, group in speaker_stats.groupby("gender"):
        # Shuffle to eliminate order biases, then sort in descending order by duration
        group = group.sample(frac=1.0, random_state=rng_seed).reset_index(drop=True)
        group = group.sort_values("duration_ms", ascending=False).reset_index(drop=True)

        group_total = group["duration_ms"].sum()
        targets = {s: ratios[s] * group_total for s in splits}
        assigned = {s: 0.0 for s in splits}
        max_dev_test = {
            "dev": args.max_speaker_share * targets["dev"],
            "test": args.max_speaker_share * targets["test"],
        }

        for _, row in group.iterrows():
            cid, dur = row["client_id"], row["duration_ms"]
            candidates = {}
            for s in splits:
                if s in ("dev", "test") and targets[s] > 0 and dur > max_dev_test[s]:
                    continue  # locuteur trop volumineux pour ce split
                candidates[s] = targets[s] - assigned[s]  # déficit vs cible
            if not candidates:
                candidates = {"train": targets["train"] - assigned["train"]}
            best_split = max(candidates, key=candidates.get)
            speaker_split[cid] = best_split
            assigned[best_split] += dur

    df["split"] = df["client_id"].map(speaker_split)

    # --- Check for speaker disjunction ---
    overlap = df.groupby("client_id")["split"].nunique()
    assert (overlap == 1).all(), "Error: one speaker appears in several splits!"

    # --- Report ---
    print("--- Repartition (duration / clips / speakers) ---")
    report = df.groupby("split").agg(
        duration_h=("duration_ms", lambda x: x.sum() / 3_600_000),
        n_clips=("duration_ms", "count"),
        n_speakers=("client_id", "nunique"),
    ).reindex(splits)
    report["pct_duree"] = (report["duration_h"] / report["duration_h"].sum() * 100).round(1)
    print(report.round(2), "\n")

    print("--- Duration (h) by split x gender ---")
    print((df.groupby(["split", "gender"])["duration_ms"].sum().unstack().fillna(0)
           / 3_600_000).round(2), "\n")

    # --- Writing files ---
    for s in splits:
        out_path = f"{args.output_dir}/{s}.tsv"
        df[df["split"] == s].drop(columns=["split"]).to_csv(out_path, sep="\t", index=False)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
