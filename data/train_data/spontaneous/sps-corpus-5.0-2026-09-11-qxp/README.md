# *Punu qhichwa* &mdash; Puno Quechua (`qxp`)

This datasheet is for sps-corpus-5.0-2026-09-11 of the Mozilla Common Voice *Spontaneous Speech* dataset for Puno Quechua [Punu qhichwa - `qxp`]. The dataset contains 7305 clips representing 35.41 hours of recorded speech (5.18 hours validated) from 110 speakers.

## Language

Puno Quechua, identified by the ISO 639-3: [qxp](https://iso639-3.sil.org/code/qxp), belongs to the Quechua II group. The [2017 National
Census in Peru](https://www.gob.pe/en/535-check-results-of-the-2017-national-census) estimates approximately 474203 speaker of Puno Quechua, predominantly in the Puno region of Peru.

## Data splits for modelling

The dataset clips are categorised by transcription status and training-set assignment. The following tables summarise the distribution.

### Audio clips

| Bucket | Clips | % |
| --- | --- | --- |
| Transcribed & Validated | 1,074 | 14.7% |
| Transcribed & Pending | 148 | 2.0% |
| Not transcribed | 6,083 | 83.3% |

### Training splits

| Bucket | Clips | % |
| --- | --- | --- |
| Train | 478 | 6.5% |
| Dev | 300 | 4.1% |
| Test | 296 | 4.1% |
| Unassigned | 6,231 | 85.3% |

Training split coverage: 1,074 of 1,074 transcribed & validated clips (100.0%)

## Transcriptions

### Transcription status

| Bucket | Clips | % |
| --- | --- | --- |
| Validated | 1,074 | 87.9% |
| Pending | 148 | 12.1% |
| Edited | 20 | 1.6% |

### Writing system

The writting system for the Puno Quechua language is not formaly described, however there are some resources that can be mentioned. Linguistic resources for Puno Quechua, as cataloged by Glottolog [puno1238](https://glottolog.org/resource/languoid/id/puno1238), are predominantly from the 1963–1993 period, with limited publications post-2005. These materials offer critical insights into the Puno Quechua language's grammatical framework. Like all Quechua languages, Puno Quechua forms words by sequentially adding suffixes to a root and follows a Subject-Object-Verb (SOV) word order.

### Samples

#### Questions

There follows a randomly selected sample of questions used in the corpus.

1. *Imatataq tarpurqanki kay watapi?*
2. *Wawa kaspayki, imaynatan taytamamaykita yanaparqanki?*
3. *Imatan sapa p'unchay puñunallaykipaqña ruwanki?*
4. *Imaynatan paqariy p'unchayniykita rayminki?*
5. *Wasiykimanta willariway.*

#### Responses

There follows a randomly selected sample of transcribed responses from the corpus.

1. *Manan llakisqa kashanichu.*
2. *Wawa kaspayqa, uywakunatan michiq rirqani, unuta p'uñuypi apamurqani, hinallataq chaqrapipas yanapakurqani, papata hasp'ispa, hawasta t'iraspa.*
3. *Apamun, kunititan kunititan qampis wahamushawanki, maypi imaymana kasqaykita, familiyaykunata imaynachá kashan, chay phasil yachanaypaq. Chaykuna allin kayta apamun. Chay!*
4. *Nuqa sapa p'unchaymi llank'akuni chayta ruwaspa, llapa runakunapaq, allinchispaq.*
5. *Ukllay runa sapallan tiyasqa, chay millay runaqa sapallan tiyaspa manas piwanpas rimanakunchu ichaqa uq p'unchay uq warmis chayasqa chay warmiqa napayukunsi IImaynallan turay¡ nisqa,  chay millay runaqa manas imaninpaschu chayrayku warmiqa kutipun, waqaspa kutipun.*

### Fields

Each row of a `tsv` file represents a single audio clip, and contains the following information:

- `client_id` - hashed UUID of a given user
- `audio_id` - numeric id for audio file
- `audio_file` - audio file name
- `duration_ms` - duration of audio in milliseconds
- `prompt_id` - numeric id for prompt
- `prompt` - question for user
- `transcription` - transcription of the audio response
- `votes` - number of people that who approved a given transcript
- `age` - age of the speaker[^1]
- `gender` - gender of the speaker[^1]
- `language` - language name
- `split` - for data modelling, which subset of the data does this clip pertain to
- `char_per_sec` - how many characters of transcription per second of audio
- `quality_tags` - some automated assessment of the transcription--audio pair, separated by `|`
  - `transcription-length` - character per second under 3 characters per second
  - `speech-rate` - characters per second over 30 characters per second
  - `short-audio` - audio length under 2 seconds
  - `long-audio` - audio length over 5 minutes
  - `non-allowed-script` - transcription contains characters from a writing system not associated with the language
  - `mixed-script-words` - a single word contains characters from multiple writing systems
  - `mixed-script-transcription` - transcription spans multiple writing systems, but each word consistently uses only one

---

[^1]: For a full list of age, gender, and accent options, see the [demographics spec](https://github.com/common-voice/common-voice/blob/main/web/src/stores/demographics.ts). These will only be reported if the speaker opted in to provide that information.

## Get involved

### Community links

* [QuechuaBase telegram group](https://t.me/QuechuaBase)

- [Common Voice translators on Pontoon](https://pontoon.mozilla.org/qxp/common-voice/contributors/)
- [Common Voice Communities](https://github.com/common-voice/common-voice/blob/main/docs/COMMUNITIES.md)

### Discussions

- [Common Voice on Matrix](https://chat.mozilla.org/#/room/#common-voice:mozilla.org)
- [Common Voice on Discourse](https://discourse.mozilla.org/t/about-common-voice-readme-first/17218)
- [Common Voice on Discord](https://discord.gg/9QTj9zwn)
- [Common Voice on Telegram](https://t.me/mozilla_common_voice)

### Contribute

- [Contribute questions](https://commonvoice.mozilla.org/spontaneous-speech/beta/question)
- [Validate questions](https://commonvoice.mozilla.org/spontaneous-speech/beta/validate)
- [Answer questions](https://commonvoice.mozilla.org/spontaneous-speech/beta/prompts)
- [Transcribe recordings](https://commonvoice.mozilla.org/spontaneous-speech/beta/transcribe)
- [Validate transcriptions](https://commonvoice.mozilla.org/spontaneous-speech/beta/check-transcript)

## Acknowledgements

### Datasheet authors

* Elwin Huaman &lt;elwin.huaman@hotmail.com&gt;

### Citation guidelines

If you use this dataset in your research, please cite the following publication:

```bibtex
    @inproceedings{HuamanHHQ25,
    author       = {Elwin Huaman and
                    Jorge Luis Huaman and
                    Wendi Huaman and
                    Ninfa Quispe},
    title        = {Quechua Speech Datasets in Common Voice: The Case of Puno Quechua},
    booktitle    = {Information Management and Big Data - 12th Annual International Conference, SIMBig 2025, Lima, Peru, October 29-31, 2025, Proceedings},
    series       = {Communications in Computer and Information Science},  
    publisher    = {Springer},
    year         = {2025},
    }
```

### Funding

This dataset was partially funded by the *Open Multilingual Speech Fund* managed by Mozilla Common Voice.

## Licence

This dataset is released under the [Creative Commons Zero (CC-0)](https://creativecommons.org/public-domain/cc0/) licence. By downloading this data you agree to not determine the identity of speakers in the dataset.
