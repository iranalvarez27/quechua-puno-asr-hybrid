# QXP Test Dataset — Puno Quechua ASR

Test set for evaluating automatic speech recognition on **Puno Quechua** (ISO 639-3 `qxp`).

**659 clips · 59 min · 7 speakers**

| Type | Clips | Duration | Description |
|---|---|---|---|
| `scripted` | 438 | 25.7 min | Read-aloud sentences |
| `spontaneous` | 221 | 33.5 min | Elder speakers answering open questions |

## Contents

```
qxp_test_dataset/
├── audios/                 659 WAV files (16 kHz, mono, 16-bit PCM)
├── qxp_test_dataset.tsv    metadata
└── qxp.tsv                 submission template
```

### `qxp_test_dataset.tsv`

| Column | Values |
|---|---|
| `file_name` | e.g. `qxp_008067c9b0cb.wav` |
| `gender` | `female_feminine`, `male_masculine` |
| `age` | `twenties`, `thirties`, `sixties` (`sixties` = 60 or older) |
| `type` | `scripted`, `spontaneous` |

## Submission format

A tab-separated file named `qxp.tsv`, with no header and one line per audio file:

```
qxp_008067c9b0cb.wav<TAB>your predicted transcription here
```

`qxp.tsv` in this folder already lists every file. Replace the placeholder text with your system's output.

## Scoring

- Character Error Rate (CER) and Word Error Rate (WER) are the main metrics.
- Before scoring, both the reference and your output are normalised:
  - Unicode NFC
  - lowercase
  - punctuation removed
- **The apostrophe `'` and `ñ` are kept.** The apostrophe marks ejective consonants (`k'anka` ≠ `kanka`).

## Recording conditions

Speakers recorded the clips themselves, in a web browser on their own devices. Rooms, microphones and background noise were not controlled.
