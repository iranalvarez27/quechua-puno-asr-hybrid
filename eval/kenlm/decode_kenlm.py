"""
Beam-search CTC decoding with a KenLM language model (pyctcdecode) for the
fine-tuned wav2vec 2.0 qxp models.

fairseq's own kenlm decoder needs flashlight, which is missing from the Docker
image (see eval/fairseq/infer_patched.py), so decoding is done with pyctcdecode.

Acoustic log-probs are computed once per (checkpoint, subset) and cached, so
an alpha/beta/beam grid only re-runs the beam search. A greedy (no-LM) score is
always reported as a sanity check against results/summary.csv.

Run inside Docker from the project root:
  # tune on valid
  python eval/kenlm/decode_kenlm.py --subset valid \
      --alpha 0.3 0.5 0.8 1.2 --beta 0.5 1.0 1.5 --beam-width 50 100 250
  # final eval with the best combination found on valid
  python eval/kenlm/decode_kenlm.py --subset test       --alpha 0.5 --beta 1.0 --beam-width 100
  python eval/kenlm/decode_kenlm.py --subset test_spont --alpha 0.5 --beta 1.0 --beam-width 100

  # blind challenge test -> submission file (no references, single best alpha/beta/beam)
  python eval/kenlm/decode_kenlm.py --submit-dir data/qxp_test_dataset \
      --alpha 0.5 --beta 1.0 --beam-width 100

Outputs in results/kenlm/<checkpoint_dir_name>/<subset>/ (--submit-dir: qxp_test/qxp.tsv):
  logprobs.pt   cached acoustic log-probs
  grid.tsv      WER/CER for greedy and every (alpha, beta, beam_width)
  hypos.tsv     per-utterance ref / greedy / best-LM hypothesis
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import unicodedata
import warnings
from math import gcd
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, "/fairseq")
warnings.filterwarnings("ignore")

import jiwer
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from pyctcdecode import build_ctcdecoder
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = ROOT / "checkpoints" / "ft_cpt_validated" / "checkpoint_best.pt"
DEFAULT_MANIFEST_DIR = ROOT / "data" / "manifests" / "finetune" / "qxp_v2"
KENLM_DIR = ROOT / "eval" / "kenlm"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def ltr_to_text(line: str) -> str:
    text = "".join(" " if tok == "|" else tok for tok in line.split())
    return " ".join(text.split())


def read_manifest(manifest_dir: Path, subset: str, audio_root: str | None):
    with open(manifest_dir / f"{subset}.tsv") as f:
        header = f.readline().strip()
        root = Path(audio_root) if audio_root else Path(header)
        paths = [root / line.split("\t")[0] for line in f if line.strip()]
    with open(manifest_dir / f"{subset}.ltr") as f:
        refs = [ltr_to_text(line) for line in f]
    if len(paths) != len(refs):
        raise ValueError(f"{subset}: {len(paths)} audio rows but {len(refs)} transcripts")
    return paths, refs


def read_submission_names(submit_dir: Path) -> list[str]:
    template = submit_dir / "qxp.tsv"
    if not template.exists():
        template = submit_dir / "qxp_test_dataset.tsv"
    with open(template, encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("\t")[0] for line in f if line.strip()]
    return [r for r in rows if r.endswith(".wav")]


def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        g = gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g)
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# alphabet
# ---------------------------------------------------------------------------

def build_alphabet(dictionary):
    """
    Map the fairseq dictionary to a duplicate-free pyctcdecode alphabet.

    bos is the CTC blank; <pad>/</s> are folded into it. Both '|' and <unk> are
    folded into the space token, because the published checkpoints were trained
    with <unk> as the word boundary while the current dict.ltr.txt uses '|'.
    Columns sharing a label are merged with logsumexp in reduce_logprobs().
    """
    groups: dict[str, list[int]] = {"": [], " ": []}
    for i in range(len(dictionary)):
        sym = dictionary[i]
        if i == dictionary.bos() or sym in ("<pad>", "</s>"):
            key = ""
        elif sym in ("|", "<unk>"):
            key = " "
        else:
            key = sym
        groups.setdefault(key, []).append(i)
    labels = list(groups)
    return labels, [groups[label] for label in labels]


def reduce_logprobs(lp: torch.Tensor, index_groups: list[list[int]]) -> torch.Tensor:
    return torch.stack([torch.logsumexp(lp[:, idx], dim=-1) for idx in index_groups], dim=-1)


# ---------------------------------------------------------------------------
# acoustic model
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_logprobs(ckpt: Path, manifest_dir: Path, paths: list[Path], device: str):
    from fairseq import checkpoint_utils

    print(f"Loading model {ckpt} ...")
    models, _, task = checkpoint_utils.load_model_ensemble_and_task(
        [str(ckpt)], arg_overrides={"data": str(manifest_dir)}
    )
    model = models[0].eval().to(device)
    labels, index_groups = build_alphabet(task.target_dictionary)
    normalize = getattr(task.cfg, "normalize", False)
    print(f"  alphabet ({len(labels)}): {labels}")
    print(f"  audio layer-norm: {normalize}\n")

    out = []
    for i, path in enumerate(paths):
        x = torch.from_numpy(load_audio(path)).to(device)
        if normalize:
            x = F.layer_norm(x, x.shape)
        x = x.unsqueeze(0)
        net_out = model(source=x, padding_mask=torch.zeros_like(x, dtype=torch.bool))
        lp = model.get_normalized_probs(net_out, log_probs=True)[:, 0].float()  # (T, V)
        if i == 0 and lp.shape[-1] != len(task.target_dictionary):
            raise RuntimeError(
                f"Model outputs {lp.shape[-1]} classes but dict.ltr.txt gives "
                f"{len(task.target_dictionary)}: the dictionary does not match this checkpoint."
            )
        out.append(reduce_logprobs(lp, index_groups).cpu())
        if (i + 1) % 200 == 0 or i + 1 == len(paths):
            print(f"  acoustic forward: {i + 1}/{len(paths)}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return labels, out


# ---------------------------------------------------------------------------
# decoding + scoring
# ---------------------------------------------------------------------------

def greedy_decode(lp: np.ndarray, labels: list[str]) -> str:
    chars, prev = [], None
    for idx in lp.argmax(axis=-1):
        if idx != prev and labels[idx] != "":
            chars.append(labels[idx])
        prev = idx
    return " ".join("".join(chars).split())


def beam_decode(decoder, logits: list[np.ndarray], beam_width: int, num_workers: int) -> list[str]:
    if num_workers > 1:
        with get_context("fork").Pool(num_workers) as pool:
            hyps = decoder.decode_batch(pool, logits, beam_width=beam_width)
    else:
        hyps = [decoder.decode(lp, beam_width=beam_width) for lp in logits]
    return [" ".join(h.split()) for h in hyps]


def score(refs: list[str], hyps: list[str]) -> tuple[float, float]:
    return jiwer.wer(refs, hyps) * 100, jiwer.cer(refs, hyps) * 100


def load_unigrams(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return sorted({w for line in f for w in line.split()})


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--subset", default="valid", help="valid | test | test_spont")
    parser.add_argument("--audio-root", default=None,
                        help="Override the manifest root (default: first line of the .tsv)")
    parser.add_argument("--lm", type=Path, default=KENLM_DIR / "qxp_4gram.bin")
    parser.add_argument("--unigrams", type=Path, default=KENLM_DIR / "corpus.txt",
                        help="Text file whose words form the LM vocabulary (needed for .bin LMs)")
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.5], help="LM weight")
    parser.add_argument("--beta", type=float, nargs="+", default=[1.0], help="Word insertion bonus")
    parser.add_argument("--beam-width", type=int, nargs="+", default=[100])
    parser.add_argument("--unk-offset", type=float, default=-10.0,
                        help="Score offset for words missing from the LM vocabulary")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--recompute", action="store_true", help="Ignore cached log-probs")
    parser.add_argument("--submit-dir", type=Path, default=None,
                        help="Blind test folder (audios/ + qxp.tsv template): decode it and "
                             "write the submission qxp.tsv instead of scoring a manifest")
    args = parser.parse_args()

    if args.submit_dir:
        args.subset = "qxp_test"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or ROOT / "results" / "kenlm" / args.ckpt.parent.name / args.subset
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "logprobs.pt"

    if args.submit_dir:
        names = read_submission_names(args.submit_dir)
        paths, refs = [args.submit_dir / "audios" / n for n in names], None
    else:
        paths, refs = read_manifest(args.manifest_dir, args.subset, args.audio_root)
    print(f"Subset {args.subset}: {len(paths)} utterances  |  device: {device}\n")

    if cache.exists() and not args.recompute:
        cached = torch.load(cache)
        if cached["paths"] != [str(p) for p in paths]:
            raise RuntimeError(f"{cache} was built from a different manifest; use --recompute")
        labels, logprobs = cached["labels"], cached["logprobs"]
        print(f"Loaded cached log-probs from {cache}\n")
    else:
        labels, logprobs = compute_logprobs(args.ckpt, args.manifest_dir, paths, device)
        torch.save({"paths": [str(p) for p in paths], "labels": labels, "logprobs": logprobs}, cache)
        print(f"\nCached log-probs -> {cache}\n")

    logits = [lp.numpy() for lp in logprobs]

    unigrams = load_unigrams(args.unigrams)
    decoder = build_ctcdecoder(
        labels,
        kenlm_model_path=str(args.lm),
        unigrams=unigrams,
        alpha=args.alpha[0],
        beta=args.beta[0],
        unk_score_offset=args.unk_offset,
    )

    if args.submit_dir:
        if len(args.alpha) * len(args.beta) * len(args.beam_width) != 1:
            raise SystemExit("--submit-dir takes a single --alpha/--beta/--beam-width "
                             "(tune them on valid first)")
        decoder.reset_params(alpha=args.alpha[0], beta=args.beta[0], unk_score_offset=args.unk_offset)
        hyps = beam_decode(decoder, logits, args.beam_width[0], args.num_workers)
        out_tsv = out_dir / "qxp.tsv"
        with open(out_tsv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            for p, h in zip(paths, hyps):
                writer.writerow([p.name, unicodedata.normalize("NFC", h).lower()])
        empty = sum(1 for h in hyps if not h.strip())
        print(f"Wrote {len(hyps)} lines ({empty} empty) -> {out_tsv.relative_to(ROOT)}")
        return

    greedy = [greedy_decode(lp, labels) for lp in logits]
    g_wer, g_cer = score(refs, greedy)
    rows = [{"decoder": "greedy", "alpha": "", "beta": "", "beam_width": "",
             "wer": f"{g_wer:.2f}", "cer": f"{g_cer:.2f}"}]
    print(f"greedy (no LM)                          WER={g_wer:6.2f}%  CER={g_cer:5.2f}%")

    best = None
    for alpha, beta, beam in itertools.product(args.alpha, args.beta, args.beam_width):
        decoder.reset_params(alpha=alpha, beta=beta, unk_score_offset=args.unk_offset)
        hyps = beam_decode(decoder, logits, beam, args.num_workers)
        wer, cer = score(refs, hyps)
        print(f"kenlm alpha={alpha:<4} beta={beta:<4} beam={beam:<4}    WER={wer:6.2f}%  CER={cer:5.2f}%")
        rows.append({"decoder": "kenlm", "alpha": alpha, "beta": beta, "beam_width": beam,
                     "wer": f"{wer:.2f}", "cer": f"{cer:.2f}"})
        if best is None or wer < best[0]:
            best = (wer, cer, alpha, beta, beam, hyps)

    with open(out_dir / "grid.tsv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    wer, cer, alpha, beta, beam, best_hyps = best
    with open(out_dir / "hypos.tsv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["path", "ref", "greedy", "kenlm"])
        writer.writerows(zip(map(str, paths), refs, greedy, best_hyps))

    print(f"\nBest: alpha={alpha} beta={beta} beam={beam}  WER={wer:.2f}%  CER={cer:.2f}%  "
          f"(greedy {g_wer:.2f}% / {g_cer:.2f}%)")
    print(f"Results -> {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
