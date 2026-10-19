#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SIMBig 2026 - Task 1: ASR para quechua de Puno (qxp). Pipeline completo en un solo script.

Hace, en orden: datos (splits oficiales del repo + texto curado + silver SIN contaminación) ->
entrenamiento de recetas (validated, silver_conf, silver) -> evaluación honesta en dev/test propios ->
enrutado por tipo de clip -> predicciones del test oficial -> qxp.tsv + zip + run_config.json.

Uso (desde la raíz del repo asr-puno-quechua):
  python run_qxp.py fetch --download-data     # NODO DE ACCESO (con internet): modelo base y datos
  python run_qxp.py run --smoke               # prueba rápida (minutos) para validar el entorno
  python run_qxp.py run                       # corrida completa (nodo con GPU, vía sbatch)
  python run_qxp.py run --no-train --lm eval/kenlm/qxp_4gram.bin --lm-tune eval/kenlm/qxp_tune.bin
                                              # re-evalúa/predice con beam search + KenLM (sin reentrenar)

Recetas (se entrenan en el orden dado por --recipes):
  validated    : leído (texto curado) + espontáneo validado x3
  silver_conf  : validated + silver filtrado por confianza (teacher = modelo 'validated';
                 puntaje = acuerdo teacher-vs-etiqueta silver, desempate por confianza CTC)
  silver       : validated + todo el silver limpio

Notas de diseño:
  * El audio del test oficial nunca entra en train/dev. El silver se limpia de clips y hablantes de dev/test.
  * La confianza por secuencia es la probabilidad máxima media de los frames no-blank del camino greedy.
    Para FILTRAR por cuantil no hace falta calibrarla; solo importa el orden.
  * Los nodos de cómputo de Khipu no tienen internet: usa `fetch` antes y se activa el modo offline.
  * --lm (apagado por defecto) decodifica evaluate/predict con pyctcdecode + KenLM. alpha/beta se eligen
    en dev por receta y tipo de clip; choose() decide entre greedy y LM según dev. Usa --lm-tune con un
    LM entrenado SIN texto de dev para que esa elección no tenga fuga. El filtro silver_conf sigue en greedy.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import inspect
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import unicodedata
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SR = 16000
AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a"}
_APOS = {ord(c): "'" for c in "’‘´`ʼʻ′"}
# IDs tomados de data/download_data.py del HEAD del repo (git); los antiguos (cmu62..., cmu60...) ya no coinciden
MDC_DATASETS = {"scripted": "cmn2cxfs801dvo107r5rkzmvx", "spontaneous": "cmn1pujk200uno107g6el5r9y"}
MDC_API_BASE = "https://datacollective.mozillafoundation.org/api/datasets"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------------------
# Texto
# ----------------------------------------------------------------------------------------------
def normalize(text):
    """Minúsculas, NFC, sin puntuación; el apóstrofe se conserva (parte de las eyectivas)."""
    t = unicodedata.normalize("NFC", str(text)).lower().translate(_APOS)
    out = []
    for ch in t:
        if ch == "'":
            out.append(ch)
        elif unicodedata.category(ch)[0] in ("P", "S"):
            out.append(" ")
        else:
            out.append(ch)
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", "".join(out)).strip())


def rd(p):
    return pd.read_csv(p, sep="\t", dtype=str, quoting=csv.QUOTE_NONE).fillna("")


def index_audio(root):
    root = Path(root)
    if not root.exists():
        return {}
    return {p.name: str(p) for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXT}


def get_dur(p):
    import soundfile as sf
    try:
        return float(sf.info(p).duration)
    except Exception:
        import librosa
        return float(librosa.get_duration(path=p))


# ----------------------------------------------------------------------------------------------
# Descarga (solo nodo con internet)
# ----------------------------------------------------------------------------------------------
def mdc_download(key, dataset_id, dest_dir):
    import requests
    import tarfile
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = os.getenv("MDC_API_KEY")
    if not api_key:
        sys.exit("Define MDC_API_KEY (perfil de Mozilla Data Collective > API) en el entorno o en .env")
    r = requests.post(f"{MDC_API_BASE}/{dataset_id}/download",
                      headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    r.raise_for_status()
    url = r.json()["downloadUrl"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    arch = dest_dir.parent / f"{key}.tar.gz"
    with requests.get(url, stream=True) as resp:
        resp.raise_for_status()
        with open(arch, "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
    with tarfile.open(arch) as tar:
        try:
            tar.extractall(dest_dir, filter="data")
        except TypeError:  # Python < 3.12
            tar.extractall(dest_dir)
    arch.unlink()


def cmd_fetch(a):
    """Correr en el nodo de acceso (con internet)."""
    from huggingface_hub import snapshot_download
    p = snapshot_download(a.base_model)
    log("modelo base en caché:", p)
    if a.download_data:
        root = Path(a.root).resolve()
        for key, d in (("scripted", root / "data" / "scripted"), ("spontaneous", root / "data" / "spontaneous")):
            if not any(d.rglob("*.mp3")) if d.exists() else True:
                log("descargando", key)
                mdc_download(key, MDC_DATASETS[key], d)
            else:
                log(key, "ya existe, se omite")


# ----------------------------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------------------------
class Pipeline:
    def __init__(self, a):
        import torch
        self.torch = torch
        self.a = a
        self.root = Path(a.root).resolve()
        # --smoke usa su propia carpeta para no mezclar modelos de juguete con los de la corrida real
        default_work = self.root / "runs" / ("qxp_smoke" if a.smoke else "qxp_full")
        self.work = Path(a.work).resolve() if a.work else default_work
        self.work.mkdir(parents=True, exist_ok=True)
        self.scripted_dir = Path(a.scripted_dir) if a.scripted_dir else self.root / "data" / "scripted"
        self.spont_dir = Path(a.spont_dir) if a.spont_dir else self.root / "data" / "spontaneous"
        self.test_dir = Path(a.test_dir) if a.test_dir else self.root / "data" / "test"

        self.cuda = torch.cuda.is_available()
        if not self.cuda and not a.allow_cpu:
            sys.exit("No se detecta GPU CUDA. Revisa --gres en sbatch o tu instalación de PyTorch (usa --allow-cpu solo para pruebas).")
        self.device = torch.device("cuda" if self.cuda else "cpu")
        if self.cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
        else:
            self.gpu_name, vram, self.use_bf16 = "cpu", 0.0, False
        if a.batch:
            self.bs = a.batch
        elif vram >= 40:
            self.bs = 16
        elif vram >= 22:
            self.bs = 8
        elif vram >= 14:
            self.bs = 4
        else:
            self.bs = 2
        self.acc = max(1, a.eff_batch // self.bs)
        log(f"dispositivo: {self.gpu_name} | {vram:.1f} GB | bf16={self.use_bf16} | batch {self.bs} x acc {self.acc}")

        # caché de duraciones compartida entre corridas/trabajos (se escribe de forma atómica)
        self.dur_file = self.root / "runs" / "durations.json"
        self.dur_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.dur = json.loads(self.dur_file.read_text()) if self.dur_file.exists() else {}
        except ValueError:
            self.dur = {}
        self.S = {}
        self.silver_conf = None
        self.best = {}
        self.lm_params, self._decoders, self._lm_groups = {}, {}, None
        if a.lm:
            for p in filter(None, (a.lm, a.lm_tune, a.lm_unigrams)):
                if not Path(p).exists():
                    sys.exit(f"No existe {p}")
            try:
                import pyctcdecode, kenlm  # noqa: F401  (fallar antes de horas de entrenamiento)
            except ImportError as e:
                sys.exit(f"--lm requiere pyctcdecode y kenlm: {e}")

    # ------------------------------------------------------------------ datos
    def fill_durations(self, paths):
        todo = [p for p in set(paths) if p not in self.dur]
        if todo:
            log(f"calculando duración de {len(todo)} archivos...")
            with ThreadPoolExecutor(16) as ex:
                for p, d in zip(todo, ex.map(get_dur, todo)):
                    self.dur[p] = d
            tmp = self.dur_file.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self.dur))
            os.replace(tmp, self.dur_file)

    def load_data(self):
        a = self.a
        spl = next((p for p in [self.root / "data" / "splits", self.root / "data" / "splits_joh"] if p.exists()), None)
        if spl is None:
            sys.exit(f"No encuentro data/splits en {self.root}. Ejecuta desde la raíz del repo (git pull si falta).")
        cur_f = self.root / "data" / "shared-task" / "validated_sentences_curated.tsv"
        cur_map, self.use_curated = {}, not a.no_curated
        if self.use_curated:
            if cur_f.exists():
                cur = rd(cur_f)
                cur_map = dict(zip(cur.sentence_id, cur.sentence_curated))
            else:
                log("AVISO: no hay validated_sentences_curated.tsv; se usa el texto sin curar")
                self.use_curated = False
        idx = {"scripted": index_audio(self.scripted_dir), "spontaneous": index_audio(self.spont_dir)}
        log({k: len(v) for k, v in idx.items()}, "archivos de audio encontrados")
        if not idx["scripted"] or not idx["spontaneous"]:
            sys.exit(f"Faltan audios en {self.scripted_dir} o {self.spont_dir}. Corre `fetch --download-data` o copia los datos.")

        def prep(df, source, key, curated=False):
            name = df["path"].map(lambda x: Path(x).name)
            out = pd.DataFrame({"name": name, "speaker": df["client_id"] if "client_id" in df else "", "source": source})
            raw = df["sentence_id"].map(cur_map).fillna(df["sentence"]) if (curated and "sentence_id" in df) else df["sentence"]
            out["text"] = raw.map(normalize)
            ix = idx[key]
            out["audio_path"] = name.map(lambda n: ix.get(n) or ix.get(Path(n).with_suffix(".mp3").name) or "")
            miss = int((out.audio_path == "").sum())
            out = out[(out.audio_path != "") & (out.text != "")].reset_index(drop=True)
            log(f"  {source:<12} {len(out):>6} clips ({miss} sin audio)")
            return out

        c = self.use_curated
        S = {
            "sc_tr": prep(rd(spl / "validated_scripted" / "train.tsv"), "scripted", "scripted", c),
            "sc_dev": prep(rd(spl / "validated_scripted" / "dev.tsv"), "scripted", "scripted", c),
            "sc_te": prep(rd(spl / "validated_scripted" / "test.tsv"), "scripted", "scripted", c),
            "sp_tr": prep(rd(spl / "validated_spontaneous" / "train_spontaneous.tsv"), "spontaneous", "spontaneous"),
            "sp_dev": prep(rd(spl / "validated_spontaneous" / "dev_spontaneous.tsv"), "spontaneous", "spontaneous"),
            "sp_te": prep(rd(spl / "validated_spontaneous" / "test_spontaneous.tsv"), "spontaneous", "spontaneous"),
        }
        silver_all = prep(rd(spl / "silver_spontaneous" / "train.tsv"), "silver", "spontaneous")

        # silver SIN contaminación: fuera clips de dev/test, sus hablantes y los que ya tienen etiqueta validada
        eval_names = set()
        for k in ("sc_dev", "sc_te", "sp_dev", "sp_te"):
            eval_names |= set(S[k].name)
        eval_spk = {s for s in (set(S["sp_dev"].speaker) | set(S["sp_te"].speaker)) if s}
        if (silver_all.speaker == "").all():
            log("AVISO: el silver no trae client_id; solo se pueden quitar clips (no hablantes) de dev/test")
        is_eval = silver_all.name.isin(eval_names) | silver_all.speaker.isin(eval_spk)
        is_gold = silver_all.name.isin(set(S["sp_tr"].name))
        S["silver"] = silver_all[~is_eval & ~is_gold].drop_duplicates("name").reset_index(drop=True)
        self.silver_removed = int(is_eval.sum())
        log(f"silver: {len(silver_all)} -> {len(S['silver'])} (quitados {self.silver_removed} de dev/test u hablantes de dev/test, {int(is_gold.sum())} con etiqueta validada)")

        # duraciones y filtros
        self.fill_durations([p for d in S.values() for p in d.audio_path])
        for d in S.values():
            d["dur"] = d.audio_path.map(self.dur)
        for k in ("sc_tr", "sp_tr", "silver"):
            S[k] = S[k][(S[k].dur >= a.min_sec) & (S[k].dur <= a.max_sec)].reset_index(drop=True)
        for k in ("sc_dev", "sc_te", "sp_dev", "sp_te"):
            S[k] = S[k][S[k].dur >= a.min_sec].reset_index(drop=True)
        self.S = S

        # test oficial: sin fuga por nombre
        self.test_files = sorted(p for p in self.test_dir.rglob("*") if p.suffix.lower() in AUDIO_EXT) if self.test_dir.exists() else []
        self.test_names = {p.name for p in self.test_files}
        assert len(self.test_names) == len(self.test_files), "hay nombres duplicados en el audio del test"
        used = set()
        for d in S.values():
            used |= set(d.name)
        leak = used & self.test_names
        assert not leak, f"FUGA: {len(leak)} archivos del test oficial aparecen en tus datos: {sorted(leak)[:5]}"
        log(f"test oficial: {len(self.test_files)} archivos | sin coincidencia de nombre con train/dev")
        for k, d in S.items():
            log(f"  {k:<7} {len(d):>6} clips {d.dur.sum() / 3600:6.2f} h")

    def build_train(self, recipe):
        a, S = self.a, self.S
        parts = [S["sc_tr"]] + [S["sp_tr"]] * a.gold_upsample
        if recipe == "silver":
            parts.append(S["silver"])
        elif recipe == "silver_conf":
            assert self.silver_conf is not None, "silver_conf requiere calcular antes la confianza"
            parts.append(self.silver_conf)
        elif recipe != "validated":
            raise ValueError(f"receta desconocida: {recipe}")
        df = pd.concat(parts, ignore_index=True)
        assert not (set(df.name) & self.test_names), "fuga con el test oficial"
        held = set()
        for k in ("sc_dev", "sc_te", "sp_dev", "sp_te"):
            held |= set(S[k].name)
        assert not (set(df.name) & held), "fuga train/eval"
        return df.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)

    # ------------------------------------------------------------------ vocabulario / procesador
    def make_processor(self):
        from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
        proc_dir = self.work / "processor"
        proc_dir.mkdir(exist_ok=True)
        if (proc_dir / "vocab.json").exists():
            log("reutilizando vocabulario existente (NO cambies las recetas entre corridas):", proc_dir)
        else:
            # vocabulario con el texto de la receta más grande (silver) => válido para todas
            cnt = {}
            for ch in "".join(self.build_train("silver").text):
                cnt[ch] = cnt.get(ch, 0) + 1
            chars = sorted(c for c, n in cnt.items() if n >= self.a.vocab_min_count)
            dropped = {c: n for c, n in cnt.items() if n < self.a.vocab_min_count}
            log(f"{len(chars)} símbolos:", "".join(chars), "| descartados (raros):", dropped)
            vocab = {("|" if c == " " else c): i for i, c in enumerate(chars)}
            vocab["[UNK]"] = len(vocab)
            vocab["[PAD]"] = len(vocab)
            (proc_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False))
            self.vocab_dropped = dropped
        tok = Wav2Vec2CTCTokenizer(str(proc_dir / "vocab.json"), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
        fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SR, padding_value=0.0, do_normalize=True, return_attention_mask=True)
        self.proc = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)
        self.proc.save_pretrained(proc_dir)
        self.tok, self.fe, self.pad_id = tok, fe, tok.pad_token_id
        vocab = json.loads((proc_dir / "vocab.json").read_text())
        self.vocab_set = {(" " if k == "|" else k) for k in vocab if k not in ("[UNK]", "[PAD]")}
        self.chars = "".join(sorted(self.vocab_set))

    def to_label(self, text):
        return re.sub(r"\s+", " ", "".join(c for c in text if c in self.vocab_set)).strip()

    def shrink_for_smoke(self):
        n = self.a.smoke_clips
        for k in list(self.S):
            if len(self.S[k]) > n:
                self.S[k] = self.S[k].sample(n, random_state=self.a.seed).reset_index(drop=True)
        log(f"SMOKE: cada conjunto reducido a <= {n} clips")

    # ------------------------------------------------------------------ inferencia
    def load_model(self, path):
        from transformers import Wav2Vec2ForCTC
        return Wav2Vec2ForCTC.from_pretrained(path).to(self.device)

    def infer(self, model, paths, max_batch_samples=SR * 120, keep_logprobs=False):
        """Devuelve [(texto normalizado, confianza)] por clip. Confianza = prob. máx. media en frames no-blank.
        Con keep_logprobs devuelve además las log-probs (T, V) de cada clip, para decodificar con LM."""
        import librosa
        torch = self.torch
        model.eval()
        order = sorted(range(len(paths)), key=lambda i: self.dur.get(paths[i], 10))
        res, batch, longest = [None] * len(paths), [], 0
        lps = [None] * len(paths) if keep_logprobs else None

        def flush():
            nonlocal batch, longest
            if not batch:
                return
            wavs = []
            for i in batch:
                w, _ = librosa.load(paths[i], sr=SR, mono=True)
                if len(w) < 4000:
                    w = np.pad(w, (0, 4000 - len(w)))
                wavs.append(w)
            inp = self.fe(wavs, sampling_rate=SR, padding=True, return_tensors="pt", return_attention_mask=True)
            inp = {k: v.to(self.device) for k, v in inp.items()}
            if self.cuda:
                ctx = torch.autocast("cuda", dtype=torch.bfloat16 if self.use_bf16 else torch.float16)
            else:
                ctx = contextlib.nullcontext()
            with ctx:
                logits = model(**inp).logits
            mx, am = logits.float().softmax(-1).max(-1)
            lens = model._get_feat_extract_output_lengths(inp["attention_mask"].sum(-1)).to(am.device)
            pad_mask = torch.arange(am.shape[1], device=am.device)[None, :] >= lens[:, None]
            am = am.masked_fill(pad_mask, self.pad_id)
            texts = self.tok.batch_decode(am.cpu().numpy(), skip_special_tokens=True)  # sin "[UNK]" literal
            logp = logits.float().log_softmax(-1).cpu() if keep_logprobs else None
            for j, i in enumerate(batch):
                L = int(lens[j])
                m, nb = mx[j, :L], am[j, :L] != self.pad_id
                conf = float(m[nb].mean()) if bool(nb.any()) else float(m.mean())
                res[i] = (normalize(texts[j]), conf)
                if keep_logprobs:
                    lps[i] = logp[j, :L].numpy()
            batch, longest = [], 0

        with torch.no_grad():
            for i in order:
                n = int(self.dur.get(paths[i], 10) * SR)
                if batch and max(longest, n) * (len(batch) + 1) > max_batch_samples:
                    flush()
                batch.append(i)
                longest = max(longest, n)
            flush()
        return (res, lps) if keep_logprobs else res

    @staticmethod
    def score(refs, hyps):
        import jiwer
        pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
        r, h = zip(*pairs)
        return 100 * jiwer.wer(list(r), list(h)), 100 * jiwer.cer(list(r), list(h))

    # ------------------------------------------------------------------ decodificación con KenLM (--lm)
    def _lm_decoder(self, lm_path, n_vocab):
        from pyctcdecode import build_ctcdecoder
        key = str(lm_path)
        if key in self._decoders:
            return self._decoders[key]
        if self._lm_groups is None:
            # pyctcdecode no admite etiquetas repetidas: [PAD] (blank CTC), [UNK], <s>, </s> se funden en
            # el blank y '|' es el espacio. Las columnas que comparten etiqueta se suman en _lm_reduce().
            tk = self.tok
            special = {tk.pad_token, tk.unk_token, tk.bos_token, tk.eos_token, None}
            groups = {"": [], " ": []}
            for i, t in enumerate(tk.convert_ids_to_tokens(list(range(n_vocab)))):
                k = "" if t in special else (" " if t == tk.word_delimiter_token else t)
                groups.setdefault(k, []).append(i)
            self._lm_labels = list(groups)
            self._lm_groups = [groups[k] for k in self._lm_labels]
        unigrams = None
        if not key.endswith(".arpa"):  # un .bin no trae su vocabulario: pyctcdecode lo necesita aparte
            if self.a.lm_unigrams and key == str(self.a.lm):
                text = Path(self.a.lm_unigrams).read_text(encoding="utf-8").split()
            else:
                text = [w for k in ("sc_tr", "sp_tr", "silver") for t in self.S[k].text for w in t.split()]
            unigrams = sorted(set(text))
        self._decoders[key] = build_ctcdecoder(self._lm_labels, kenlm_model_path=key, unigrams=unigrams,
                                               unk_score_offset=self.a.lm_unk_offset)
        return self._decoders[key]

    def _lm_reduce(self, lp):
        return np.stack([np.logaddexp.reduce(lp[:, g], axis=1) for g in self._lm_groups], axis=1).astype(np.float32)

    def lm_decode(self, lm_path, lps, alpha, beta):
        from multiprocessing import get_context
        if not lps:
            return []
        dec = self._lm_decoder(lm_path, lps[0].shape[-1])
        dec.reset_params(alpha=alpha, beta=beta, unk_score_offset=self.a.lm_unk_offset)
        reduced = [self._lm_reduce(lp) for lp in lps]
        with get_context("fork").Pool(max(1, self.a.workers)) as pool:
            hyps = dec.decode_batch(pool, reduced, beam_width=self.a.lm_beam)
        return [normalize(h) for h in hyps]

    def tune_lm(self, lps, refs):
        """Barrido alpha x beta sobre dev; devuelve (alpha, beta, WER, CER) del mejor por (WER+CER)/2."""
        lm = self.a.lm_tune or self.a.lm
        best = None
        for al in self.a.lm_alpha:
            for be in self.a.lm_beta:
                w, c = self.score(refs, self.lm_decode(lm, lps, al, be))
                log(f"    alpha={al:<4} beta={be:<4} WER={w:6.2f} CER={c:5.2f}")
                if best is None or (w + c) / 2 < best[0]:
                    best = ((w + c) / 2, al, be, w, c)
        return best[1:]

    # ------------------------------------------------------------------ silver por confianza
    def make_silver_conf(self):
        import jiwer
        a = self.a
        out_dir = self.work / "silver_conf"
        out_dir.mkdir(exist_ok=True)
        cache = out_dir / "scores.csv"
        silver = self.S["silver"]
        if cache.exists():
            sc = pd.read_csv(cache, sep="\t", dtype={"name": str})
            if set(sc.name) == set(silver.name):
                log("silver_conf: reutilizando puntajes de", cache)
                return self._select_conf(silver, sc)
        tpath = self.best.get("validated") or (self.work / "validated" / "best_model")
        log("silver_conf: teacher =", tpath)
        model = self.load_model(tpath)

        # diagnóstico: ¿la confianza informa sobre el error real? (gold espontáneo dev)
        dv = self.S["sp_dev"]
        out = self.infer(model, dv.audio_path.tolist())
        cers = [jiwer.cer(r, h) if r.strip() else np.nan for r, (h, _) in zip(dv.text, out)]
        rho = pd.Series([c for _, c in out]).corr(-pd.Series(cers), method="spearman")
        log(f"diagnóstico dev espontáneo: Spearman(confianza, -CER) = {rho:.3f} (más alto = confianza más informativa)")

        out = self.infer(model, silver.audio_path.tolist())
        del model
        if self.cuda:
            self.torch.cuda.empty_cache()
        agree = []
        for r, (h, _) in zip(silver.text, out):
            try:
                agree.append(1.0 - min(1.0, jiwer.cer(r, h)) if r.strip() else 0.0)
            except Exception:
                agree.append(0.0)
        sc = pd.DataFrame({"name": silver.name, "hyp": [h for h, _ in out], "conf": [c for _, c in out], "agree": agree})
        sc.to_csv(cache, sep="\t", index=False)
        self.conf_rho = float(rho)
        return self._select_conf(silver, sc)

    def _select_conf(self, silver, sc):
        keep_n = max(1, int(round(len(silver) * self.a.conf_keep)))
        sc = sc.assign(agree_r=sc.agree.round(2)).sort_values(["agree_r", "conf"], ascending=False)
        keep = set(sc.name.head(keep_n))
        log(f"silver_conf: se conservan {keep_n} de {len(silver)} clips (conf_keep={self.a.conf_keep}); "
            f"acuerdo medio conservados {sc.head(keep_n).agree.mean():.3f} vs descartados {sc.tail(len(sc) - keep_n).agree.mean() if len(sc) > keep_n else float('nan'):.3f}")
        self.silver_conf = silver[silver.name.isin(keep)].reset_index(drop=True)
        return self.silver_conf

    # ------------------------------------------------------------------ entrenamiento
    def train_recipe(self, recipe):
        import librosa
        import jiwer
        import transformers
        from transformers import (Wav2Vec2ForCTC, Trainer, TrainingArguments, TrainerCallback)
        from transformers.trainer_pt_utils import LengthGroupedSampler
        from transformers.trainer_utils import get_last_checkpoint
        torch, a = self.torch, self.a

        out = self.work / recipe
        best = out / "best_model"
        if (best / "config.json").exists():
            log(f"[{recipe}] ya entrenada -> {best}")
            return best
        if recipe == "silver_conf":
            if "validated" not in self.best:
                self.best["validated"] = self.train_recipe("validated")
            self.make_silver_conf()

        train_df = self.build_train(recipe)
        dev_small = pd.concat([d.sample(min(a.dev_eval_subset, len(d)), random_state=a.seed)
                               for d in (self.S["sc_dev"], self.S["sp_dev"])]).reset_index(drop=True)
        self_ = self

        def load_wav(p):
            w, _ = librosa.load(p, sr=SR, mono=True)
            return w

        def augment(w):
            if random.random() < 0.5:
                f = random.choice([0.9, 1.1])
                w = librosa.resample(w, orig_sr=int(SR * f), target_sr=SR)
            w = w * (10 ** (random.uniform(-6, 6) / 20))
            if random.random() < 0.3:
                snr = random.uniform(15, 35)
                p = np.mean(w ** 2) + 1e-9
                w = w + np.random.randn(len(w)) * np.sqrt(p / (10 ** (snr / 10)))
            return np.clip(w, -1.0, 1.0).astype(np.float32)

        class ASRDataset(torch.utils.data.Dataset):
            def __init__(self, df, train=False):
                self.paths, self.train = df.audio_path.tolist(), train
                self.texts = [self_.to_label(t) for t in df.text]

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, i):
                w = load_wav(self.paths[i])
                if self.train and not a.no_augment:
                    w = augment(w)
                return {"input_values": self_.fe(w, sampling_rate=SR).input_values[0],
                        "labels": self_.tok(self.texts[i]).input_ids}

        @dataclass
        class Collator:
            def __call__(self, feats):
                batch = self_.fe.pad([{"input_values": f["input_values"]} for f in feats], padding=True, return_tensors="pt")
                lab = self_.tok.pad([{"input_ids": f["labels"]} for f in feats], padding=True, return_tensors="pt")
                batch["labels"] = lab["input_ids"].masked_fill(lab["attention_mask"].ne(1), -100)
                return batch

        pad_id = self.pad_id

        def compute_metrics(pred):
            ids = np.where(pred.predictions == -100, pad_id, pred.predictions)
            labels = np.where(pred.label_ids == -100, pad_id, pred.label_ids)
            hyp = self_.tok.batch_decode(ids, skip_special_tokens=True)
            ref = self_.tok.batch_decode(labels, group_tokens=False, skip_special_tokens=True)
            pairs = [(r, h) for r, h in zip(ref, hyp) if r.strip()]
            r, h = zip(*pairs)
            return {"wer": jiwer.wer(list(r), list(h)), "cer": jiwer.cer(list(r), list(h))}

        class LengthTrainer(Trainer):
            def _get_train_sampler(self, *args, **kwargs):
                return LengthGroupedSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps,
                                            lengths=self.train_lengths)

        class TimeBudget(TrainerCallback):
            """Corta el entrenamiento al llegar al presupuesto de tiempo, evaluando y guardando antes."""
            def __init__(self, hours):
                self.limit, self.t0 = hours * 3600, None

            def on_train_begin(self, args, state, control, **kw):
                self.t0 = time.time()

            def on_step_end(self, args, state, control, **kw):
                if self.limit and time.time() - self.t0 > self.limit:
                    log(f"[{recipe}] presupuesto de tiempo agotado: se detiene en el paso {state.global_step}")
                    control.should_training_stop = True
                    control.should_evaluate = True
                    control.should_save = True
                return control

        class HeadWarmup(TrainerCallback):
            """Primeros N pasos: solo se actualiza la cabeza CTC (se anulan los gradientes del encoder)."""
            def __init__(self, steps):
                self.steps = steps

            def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
                if state.global_step < self.steps and model is not None:
                    for n, p in model.named_parameters():
                        if "lm_head" not in n and p.grad is not None:
                            p.grad = None

        model = Wav2Vec2ForCTC.from_pretrained(
            a.base_model, attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0, layerdrop=0.0,
            mask_time_prob=0.05, ctc_loss_reduction="mean", ctc_zero_infinity=True,
            pad_token_id=self.pad_id, vocab_size=len(self.tok), ignore_mismatched_sizes=True)
        model.freeze_feature_encoder()
        if a.lora:
            # Se envuelve ANTES de activar gradient checkpointing: PEFT intenta habilitar gradientes en los
            # embeddings de entrada y wav2vec2 no los tiene.
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(
                r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "intermediate_dense", "output_dense"],
                modules_to_save=["lm_head"]))
            model.print_trainable_parameters()
            # wav2vec2 no tiene embeddings de entrada: sin esto PEFT falla al guardar checkpoints
            _orig_save = model.save_pretrained
            model.save_pretrained = lambda *aa, **kw: _orig_save(*aa, save_embedding_layers=False, **kw)
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

        train_ds, dev_ds = ASRDataset(train_df, True), ASRDataset(dev_small)
        steps_per_epoch = math.ceil(len(train_ds) / (self.bs * self.acc))
        ev = max(100, steps_per_epoch // 2)
        extra = {}
        if a.smoke:
            ev, extra["max_steps"] = a.smoke_steps, a.smoke_steps * 2
        ek = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
        args = TrainingArguments(
            output_dir=str(out / "checkpoints"), per_device_train_batch_size=self.bs, per_device_eval_batch_size=self.bs,
            gradient_accumulation_steps=self.acc, learning_rate=a.lr, warmup_steps=min(a.warmup, 5) if a.smoke else a.warmup,
            num_train_epochs=a.epochs, lr_scheduler_type="linear", max_grad_norm=1.0,
            bf16=self.use_bf16, fp16=self.cuda and not self.use_bf16, use_cpu=not self.cuda,
            save_steps=ev, eval_steps=ev, logging_steps=5 if a.smoke else 50, save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
            dataloader_num_workers=a.workers, remove_unused_columns=False, report_to="none", seed=a.seed,
            label_names=["labels"], **{ek: "steps"}, **extra)
        callbacks = []
        if a.time_budget_h:
            callbacks.append(TimeBudget(a.time_budget_h))
        if a.freeze_head_steps and not a.lora:
            if hasattr(TrainerCallback, "on_pre_optimizer_step"):
                callbacks.append(HeadWarmup(a.freeze_head_steps))
            else:
                log("AVISO: esta versión de transformers no tiene on_pre_optimizer_step; --freeze-head-steps se ignora")
        trainer = LengthTrainer(model=model, args=args, data_collator=Collator(), train_dataset=train_ds, eval_dataset=dev_ds,
                                compute_metrics=compute_metrics, callbacks=callbacks,
                                preprocess_logits_for_metrics=lambda logits, labels: logits.argmax(-1))
        trainer.train_lengths = [int(d * SR) for d in train_df.dur]
        last = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
        log(f"[{recipe}] {len(train_ds)} clips ({train_df.dur.sum() / 3600:.1f} h con repeticiones), ~{steps_per_epoch} pasos/época, "
            f"eval cada {ev}; fuentes: {train_df.source.value_counts().to_dict()}; reanuda desde: {last}")
        t0 = time.time()
        trainer.train(resume_from_checkpoint=last)
        log(f"[{recipe}] listo en {(time.time() - t0) / 3600:.2f} h")
        final = trainer.model
        if a.lora:
            final = final.merge_and_unload()
        final.save_pretrained(str(best))
        self.proc.save_pretrained(best)
        pd.DataFrame(trainer.state.log_history).to_csv(out / "log_history.csv", index=False)
        del trainer, model, final
        if self.cuda:
            torch.cuda.empty_cache()
        return best

    # ------------------------------------------------------------------ evaluación y predicción
    def evaluate(self):
        S, a = self.S, self.a
        sets = {"leído/dev": S["sc_dev"], "leído/test": S["sc_te"], "espontáneo/dev": S["sp_dev"], "espontáneo/test": S["sp_te"]}
        out_dir = self.work / "eval"
        out_dir.mkdir(exist_ok=True)
        if a.lm and not a.lm_tune:
            log("AVISO: sin --lm-tune, alpha/beta se eligen con --lm; si ese LM vio texto de dev, el dev con LM sale optimista")
        rows = []
        for recipe, path in self.best.items():
            model = self.load_model(path)
            lps = {}
            for name, d in sets.items():
                out = self.infer(model, d.audio_path.tolist(), keep_logprobs=bool(a.lm))
                if a.lm:
                    out, lps[name] = out
                hyp = [t for t, _ in out]
                w, c = self.score(d.text, hyp)
                rows.append({"receta": recipe, "decoder": "greedy", "conjunto": name, "n": len(d), "WER": w, "CER": c})
                pd.DataFrame({"name": d.name, "ref": d.text, "hyp": hyp}).to_csv(
                    out_dir / f"{recipe}__{name.replace('/', '_')}.tsv", sep="\t", index=False)
            del model
            if self.cuda:
                self.torch.cuda.empty_cache()
            if not a.lm:
                continue
            for tipo in ("leído", "espontáneo"):
                log(f"[{recipe}] LM: barrido alpha/beta en {tipo}/dev")
                al, be, w, c = self.tune_lm(lps[f"{tipo}/dev"], sets[f"{tipo}/dev"].text)
                self.lm_params[f"{recipe}/{tipo}"] = (al, be)
                rows.append({"receta": recipe, "decoder": "lm", "conjunto": f"{tipo}/dev",
                             "n": len(sets[f"{tipo}/dev"]), "WER": w, "CER": c})
                d = sets[f"{tipo}/test"]
                hyp = self.lm_decode(a.lm, lps[f"{tipo}/test"], al, be)
                w, c = self.score(d.text, hyp)
                rows.append({"receta": recipe, "decoder": "lm", "conjunto": f"{tipo}/test", "n": len(d), "WER": w, "CER": c})
                pd.DataFrame({"name": d.name, "ref": d.text, "hyp": hyp}).to_csv(
                    out_dir / f"{recipe}__lm__{tipo}_test.tsv", sep="\t", index=False)
        self.res = pd.DataFrame(rows)
        self.res.to_csv(self.work / "eval_results.csv", index=False)
        log("\n" + self.res.round(2).to_string(index=False))

    def choose(self):
        """(receta, decoder) por tipo de clip según DEV. --route fija la receta; el decoder se elige igual."""
        def pick(tipo):
            d = self.res[self.res.conjunto == f"{tipo}/dev"].copy()
            if self.a.route:
                d = d[d.receta == self.a.route]
            d["m"] = (d.WER + d.CER) / 2
            r = d.sort_values("m").iloc[0]
            return (r.receta, r.decoder)
        self.choice = {"scripted": pick("leído"), "spontaneous": pick("espontáneo")}
        log("(receta, decoder) elegidos por tipo de clip (según DEV):", self.choice)

    def predict(self):
        files = self.test_files
        if not files:
            log(f"AVISO: no hay audio de test en {self.test_dir}; se omite la predicción del test oficial")
            return
        self.fill_durations([str(p) for p in files])
        by_name = {p.name: p for p in files}
        meta_f = next(iter(self.test_dir.rglob("qxp_test_dataset.tsv")), None)
        tmpl_f = next(iter(self.test_dir.rglob("qxp.tsv")), None)
        if meta_f:
            meta = pd.read_csv(meta_f, sep="\t", dtype=str)
            assert set(meta.file_name) == set(by_name), (
                f"metadatos vs audio: faltan en audio {len(set(meta.file_name) - set(by_name))}, sobran {len(set(by_name) - set(meta.file_name))}")
            tmap = dict(zip(meta.file_name, meta["type"]))
            log("enrutado por columna 'type':", meta["type"].value_counts().to_dict())
        else:
            dd = pd.concat([self.S["sc_dev"].assign(t=0), self.S["sp_dev"].assign(t=1)])
            acc, thr = max((((dd.dur >= c).astype(int) == dd.t).mean(), c) for c in np.unique(dd.dur.round(1)))
            log(f"SIN metadatos: umbral por duración {thr:.1f} s (exactitud en dev {acc:.1%})")
            tmap = {p.name: ("spontaneous" if self.dur[str(p)] >= thr else "scripted") for p in files}
        if tmpl_f:
            names = [l.split("\t")[0] for l in tmpl_f.read_text(encoding="utf-8").splitlines() if l.strip()]
            assert len(names) == len(set(names)), "la plantilla tiene nombres repetidos"
            assert set(names) == set(by_name), f"plantilla vs audio: faltan {len(set(names) - set(by_name))}, sobran {len(set(by_name) - set(names))}"
        else:
            names = [p.name for p in files]
        types = [tmap[n] for n in names]
        log(f"{len(names)} clips | leídos {types.count('scripted')}, espontáneos {types.count('spontaneous')}")

        tipo_es = {"scripted": "leído", "spontaneous": "espontáneo"}
        preds = [None] * len(names)
        for recipe in sorted({r for r, _ in self.choice.values()}):
            idx = [i for i, t in enumerate(types) if self.choice[t][0] == recipe]
            if not idx:
                continue
            use_lm = {t for t, (r, dec) in self.choice.items() if r == recipe and dec == "lm"}
            model = self.load_model(self.best[recipe])
            out = self.infer(model, [str(by_name[names[i]]) for i in idx], keep_logprobs=bool(use_lm))
            if use_lm:
                out, lps = out
            del model
            if self.cuda:
                self.torch.cuda.empty_cache()
            for i, (t, _) in zip(idx, out):
                preds[i] = t
            for t in use_lm:
                sel = [k for k, i in enumerate(idx) if types[i] == t]
                al, be = self.lm_params[f"{recipe}/{tipo_es[t]}"]
                log(f"LM en {len(sel)} clips {t} ({recipe}, alpha={al}, beta={be})")
                for k, h in zip(sel, self.lm_decode(self.a.lm, [lps[k] for k in sel], al, be)):
                    preds[idx[k]] = h
        assert all(p is not None for p in preds)
        clean = lambda t: " ".join(str(t).replace("\t", " ").replace("\n", " ").split())
        preds = [clean(t) for t in preds]
        empty = sum(1 for t in preds if not t)
        if empty:
            log(f"AVISO: {empty} predicciones vacías")
        sub = self.work / "submission"
        sub.mkdir(exist_ok=True)
        tsv = sub / "qxp.tsv"
        with open(tsv, "w", encoding="utf-8", newline="") as f:  # sin cabecera: archivo<TAB>transcripción
            for n, t in zip(names, preds):
                f.write(f"{n}\t{t}\n")
        with zipfile.ZipFile(sub / "submission_qxp.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write(tsv, "qxp.tsv")
        log(f"{len(preds)} predicciones -> {tsv} y {sub / 'submission_qxp.zip'}")

    def write_config(self):
        a = self.a
        import transformers
        try:
            commit = subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        except Exception:
            commit = ""
        cfg = {
            "args": vars(a), "choice_by_type": getattr(self, "choice", None), "use_curated_text": self.use_curated,
            "vocab": self.chars, "vocab_dropped_rare": getattr(self, "vocab_dropped", None),
            "effective_batch": self.bs * self.acc, "precision": "bf16" if self.use_bf16 else ("fp16" if self.cuda else "fp32"),
            "data": {k: {"clips": len(d), "hours": float(d.dur.sum() / 3600)} for k, d in self.S.items()},
            "silver_removed_as_eval_clips_or_speakers": self.silver_removed,
            "silver_conf_clips": None if self.silver_conf is None else len(self.silver_conf),
            "silver_conf_spearman_dev": getattr(self, "conf_rho", None),
            "lm_alpha_beta_by_recipe_type": self.lm_params or None,
            "repo_commit": commit, "gpu": self.gpu_name, "torch": self.torch.__version__,
            "transformers": transformers.__version__, "python": platform.python_version(),
            "eval": None if not hasattr(self, "res") else self.res.round(3).to_dict("records"),
        }
        (self.work / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False, default=str))

    # ------------------------------------------------------------------ orquestación
    def run(self):
        a = self.a
        self.load_data()
        self.make_processor()
        if a.smoke:
            self.shrink_for_smoke()
        for r in a.recipes:
            best = self.work / r / "best_model"
            if (best / "config.json").exists():
                self.best[r] = best
                log(f"[{r}] ya entrenada -> {best}")
            elif a.no_train:
                log(f"[{r}] no existe y --no-train está activo: se omite")
            else:
                self.best[r] = self.train_recipe(r)
        if not self.best:
            sys.exit("No hay ningún modelo entrenado.")
        self.evaluate()
        self.choose()
        self.predict()
        self.write_config()
        log("LISTO. Entrega:", self.work / "submission" / "qxp.tsv")


def parse():
    p = argparse.ArgumentParser(description="SIMBig 2026 Task 1 - ASR quechua de Puno", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("cmd", choices=["fetch", "run"])
    p.add_argument("--root", default=".")
    p.add_argument("--work", default="")
    p.add_argument("--scripted-dir", default="")
    p.add_argument("--spont-dir", default="")
    p.add_argument("--test-dir", default="")
    p.add_argument("--base-model", default="facebook/wav2vec2-xls-r-300m")
    p.add_argument("--download-data", action="store_true")
    p.add_argument("--recipes", type=lambda s: [x for x in s.split(",") if x], default="validated,silver_conf,silver".split(","))
    p.add_argument("--no-train", action="store_true", help="solo evaluar/predecir con modelos ya entrenados")
    p.add_argument("--route", default="", help="usar una sola receta para todo el test")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--eff-batch", type=int, default=32)
    p.add_argument("--batch", type=int, default=0, help="batch por dispositivo (0 = según VRAM)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--gold-upsample", type=int, default=3)
    p.add_argument("--no-curated", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--max-sec", type=float, default=30.0)
    p.add_argument("--min-sec", type=float, default=0.5)
    p.add_argument("--vocab-min-count", type=int, default=20)
    p.add_argument("--dev-eval-subset", type=int, default=300)
    p.add_argument("--conf-keep", type=float, default=0.7, help="fracción del silver que se conserva en silver_conf")
    p.add_argument("--freeze-head-steps", type=int, default=0, help="pasos iniciales solo con la cabeza CTC (0 = desactivado)")
    p.add_argument("--time-budget-h", type=float, default=0.0, help="horas máximas por receta (0 = sin límite)")
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--smoke", action="store_true", help="prueba rápida con muy pocos datos y pasos")
    p.add_argument("--smoke-clips", type=int, default=48)
    p.add_argument("--smoke-steps", type=int, default=6)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--online", action="store_true", help="no forzar modo offline de HuggingFace")
    floats = lambda s: [float(x) for x in s.split(",") if x]
    p.add_argument("--lm", default="", help="KenLM (.bin/.arpa) para beam search en evaluate/predict (vacío = solo greedy)")
    p.add_argument("--lm-tune", default="", help="KenLM sin texto de dev, solo para elegir alpha/beta en dev (vacío = --lm)")
    p.add_argument("--lm-unigrams", default="", help="texto cuyas palabras forman el vocabulario de --lm .bin (vacío = texto de train)")
    p.add_argument("--lm-alpha", type=floats, default=[0.5, 1.0, 1.5], help="pesos del LM a probar, separados por comas")
    p.add_argument("--lm-beta", type=floats, default=[1.0, 3.0], help="bonus por palabra a probar, separados por comas")
    p.add_argument("--lm-beam", type=int, default=100)
    p.add_argument("--lm-unk-offset", type=float, default=-10.0, help="penalización a palabras fuera del vocabulario del LM")
    return p.parse_args()


def main():
    import multiprocessing as mp
    # Python 3.14 usa 'forkserver' por defecto y no puede serializar las clases locales del Dataset
    mp.set_start_method("fork", force=True)
    a = parse()
    random.seed(a.seed)
    np.random.seed(a.seed)
    if a.cmd == "fetch":
        cmd_fetch(a)
        return
    if not a.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    Pipeline(a).run()


if __name__ == "__main__":
    main()
