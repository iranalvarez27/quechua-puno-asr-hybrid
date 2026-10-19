"""
Entrega híbrida por tipo de clip (columna `type` de qxp_test_dataset.tsv):
  scripted    -> HF wav2vec2 receta `validated` + KenLM (entrenado con texto curado)
  spontaneous -> baseline fairseq ft_cpt_validated + KenLM
Escribe qxp.tsv (sin cabecera, orden de la plantilla oficial) y qxp.zip con qxp.tsv en la raíz.

Uso (desde Simbig_Task1_ASR/):  py -3 entrega_hibrida/merge.py
"""
import csv
import unicodedata
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
TEST_DIR = BASE / "asr-puno-quechua" / "data" / "qxp_test_dataset"
SOURCES = {
    "scripted": BASE / "entrega_teacher_student" / "qxp.tsv",
    "spontaneous": BASE / "asr-puno-quechua" / "results" / "kenlm" / "ft_cpt_validated" / "qxp_test" / "qxp.tsv",
}
OUT = Path(__file__).resolve().parent


def read_preds(path):
    preds = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            name, _, text = line.rstrip("\n").partition("\t")
            preds[name] = text
    return preds


def clean(text):
    return " ".join(unicodedata.normalize("NFC", text).lower().replace("\t", " ").split())


def main():
    with open(TEST_DIR / "qxp_test_dataset.tsv", encoding="utf-8") as f:
        types = {r["file_name"]: r["type"] for r in csv.DictReader(f, delimiter="\t")}
    with open(TEST_DIR / "qxp.tsv", encoding="utf-8") as f:
        order = [line.split("\t")[0] for line in f if line.strip()]

    preds = {t: read_preds(p) for t, p in SOURCES.items()}
    for t, p in preds.items():
        assert set(p) == set(order), f"{SOURCES[t]} no cubre exactamente los archivos de la plantilla"
    assert set(types) == set(order) and len(order) == len(set(order)) == 659

    rows = [(n, clean(preds[types[n]][n])) for n in order]
    empty = [n for n, t in rows if not t]
    with open(OUT / "qxp.tsv", "w", encoding="utf-8", newline="") as f:
        for n, t in rows:
            f.write(f"{n}\t{t}\n")
    with zipfile.ZipFile(OUT / "qxp.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(OUT / "qxp.tsv", "qxp.tsv")

    counts = {t: sum(1 for n in order if types[n] == t) for t in SOURCES}
    print(f"{len(rows)} líneas | por tipo: {counts} | vacías: {len(empty)} {empty[:5]}")
    print(f"-> {OUT / 'qxp.tsv'}\n-> {OUT / 'qxp.zip'}")


if __name__ == "__main__":
    main()
