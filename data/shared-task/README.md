# validated_sentences_curated.tsv

Curated reference transcriptions for the **Puno Quechua (`qxp`)** validated sentences of Mozilla Common Voice (Scripted Speech, v27).
It holds one normalized text per sentence, for use as the reference in ASR training and evaluation.

## Format

| Column | Description |
|---|---|
| `sentence_id` | Common Voice sentence ID. Joins to `sentence_id` in `validated.tsv`. |
| `sentence_curated` | Curated sentence: lowercase, no punctuation, words separated by single spaces. |

- Tab-separated, UTF-8, header row, sorted by `sentence_id`.

## Contents

| | |
|---|---|
| Sentences (rows) | 2,067 (every `sentence_id` in `validated.tsv`) |
| Distinct sentence texts | 2,051 |
| Recordings covered | 22,727, from 77 speakers |
| Words (total / distinct) | 8,272 / 4,595 |
| Words per sentence | 1–10 (mean 4.0) |
| Sentences with spelling changes vs. the original | 627 (all rows were also lowercased and had punctuation removed) |

## Curation conventions

Spelling follows **Puno Quechua variety orthography**:

- **Characters:** `a–z`, `ñ` and the apostrophe. The apostrophe is part of the letter for ejectives (`k' p' q' t' ch'`). There are no accents and no punctuation.
- **Three-vowel spelling** for Quechua words, with the semivowels written as `w`/`y` (`ñawpaq`, `chawpi`, `kawsay`, `wathiya`).
- **Puno Quechua forms:** `huq`, `alqu`, `ashkha`, `ishkay`, `chhaqay`, `chaqra`, `qulqi`, `qhilqa`, `llank'ay`, `mishk'i`, `nuqa`, `ashwan`, `chhiqaq`.
- **`n` before `p`** at morpheme boundaries (`wasinpi`, `allinpuni`, `ashwanpis`). The evidential ending is written `-n`, not `-m`.
- **`-hina`** (comparative case) is joined to the word before it when that word is a noun, a pronoun or a `-spa` form (`khuchihina`, `qanhinachu`, `parlaspahina`). It is written separately after finite verbs and `-chu` forms (`rikuykun hina`, `manachu hina`). The root and connector *hina* also stays separate (`hinaqa`, `hinaspa`, `hinan`).

## Known open points

- **16 duplicate pairs:** 16 texts are each shared by two `sentence_id`s (362 recordings). (try to) Keep each pair on the same side of any train/dev/test split.
- **Near-duplicates:** about 30 sentence pairs differ by a single word, because they were written from shared templates. Treat them the same way when splitting.
- **Mixed spelling policy:**
  - **Personal names** use Spanish spelling (`pablo`, `rosas`, `domitila`), however some names are spelled the Quechua way (`lurdes`, `dius`, `juliuq`).
  - **Borrowed words** are mixed (`plasa`, `lisinsia` vs. `alcalde`, `tecnologia`).
  - **Place names** use Spanish spelling (`juliaca`, `sicuani`, `amantani`), while `punu` and `qusqu` use Quechua spelling.

## Usage

Join to the clips with `sentence_id`, for example with pandas:

```python
import pandas as pd
clips = pd.read_csv("validated.tsv", sep="\t", quoting=3)
ref = pd.read_csv("validated_sentences_curated.tsv", sep="\t", quoting=3)
data = clips.merge(ref, on="sentence_id")   # use data["sentence_curated"] as the transcript
```

Most distinct words (about 77%) occur only once, because Quechua words carry many suffixes. Subword units (BPE, unigram or Morfessor) are recommended over whole-word vocabularies.