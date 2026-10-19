# *Punu qhichwa* &mdash; Puno Quechua (`qxp`)

This datasheet is for cv-corpus-27.0-2026-09-11 of the Mozilla Common Voice *Scripted Speech* dataset for Puno Quechua [Punu qhichwa - `qxp`]. The dataset contains 25423 clips representing 34.88 hours of recorded speech (31.18 hours validated) from 81 speakers, recorded from a text corpus of 2,071 sentences.

## Language

Puno Quechua, identified by the ISO 639-3: [qxp](https://iso639-3.sil.org/code/qxp), belongs to the Quechua II group. The [2017 National
Census in Peru](https://www.gob.pe/en/535-check-results-of-the-2017-national-census) estimates approximately 474203 speaker of Puno Quechua, predominantly in the Puno region of Peru.

### Accents

| Code | Accent | Clips | Speakers |
|---|---|---|---|
| - |  | 8,919 (35.1%) | 20 (24.7%) |

## Demographic information

The dataset includes the following self-declared age and gender distributions. A coverage summary is shown below each table.

### Gender

Self-declared gender information. The table shows clip and speaker counts with percentages. Speakers who did not declare a gender are listed as Unspecified. A dash (-) indicates zero.

| Code | Gender | Clips | Speakers |
|---|---|---|---|
| male_masculine | Male, masculine | 384 (1.5%) | 1 (1.2%) |
| female_feminine | Female, feminine | 17,871 (70.3%) | 57 (70.4%) |
| transgender | Transgender | - | - |
| non-binary | Non-binary | - | - |
| do_not_wish_to_say | Prefer not to say | - | - |
| - | Unspecified | 7,168 (28.2%) | 27 (33.3%) |

*Gender declared: 18,255 of 25,423 clips (71.8%), 54 of 81 speakers (66.7%)*

### Age

Self-declared age information. The table shows clip and speaker counts with percentages. Speakers who did not declare an age are listed as Unspecified. A dash (-) indicates zero.

| Code | Age | Clips | Speakers |
|---|---|---|---|
| teens | Teens | 251 (1.0%) | 1 (1.2%) |
| twenties | Twenties | 13,592 (53.5%) | 35 (43.2%) |
| thirties | Thirties | 6,797 (26.7%) | 23 (28.4%) |
| fourties | Fourties | 2,368 (9.3%) | 7 (8.6%) |
| fifties | Fifties | 1,943 (7.6%) | 3 (3.7%) |
| sixties | Sixties | 180 (0.7%) | 3 (3.7%) |
| seventies | Seventies | - | - |
| eighties | Eighties | - | - |
| nineties | Nineties | - | - |
| - | Unspecified | 292 (1.1%) | 14 (17.3%) |

*Age declared: 25,131 of 25,423 clips (98.9%), 67 of 81 speakers (82.7%)*

## Data splits for modelling

**Clip buckets**

| Bucket | Clips |
|---|---|
| Validated | 22,727 (89.4%) |
| Invalidated | 2,572 (10.1%) |
| Other | 124 (0.5%) |

**Training splits**

| Split | Clips |
|---|---|
| Train | 709 (3.1%) |
| Dev | 679 (3.0%) |
| Test | 679 (3.0%) |

*Training split coverage: 2,067 of 22,727 validated clips (9.1%)*

The dataset contains 22727 validated, 2572 invalidated, and 124 unresolved clips. The average clip duration is 4.939 seconds.

## Text corpus

**Validated sentences:** 2,067

| Category | Count |
|---|---|
| Unvalidated sentences | 4 |
| Pending sentences | 4 |
| Rejected sentences | - |
| Reported sentences | 197 |

The corpus contains 2,071 sentences: 2,067 validated and 4 unvalidated (4 pending review, 0 rejected), with 197 reported for review.

### Writing system

The writting system for the Puno Quechua language is not formaly described, however there are some resources that can be mentioned. Linguistic resources for Puno Quechua, as cataloged by Glottolog [puno1238](https://glottolog.org/resource/languoid/id/puno1238), are predominantly from the 1963–1993 period, with limited publications post-2005. These materials offer critical insights into the Puno Quechua language's grammatical framework. Like all Quechua languages, Puno Quechua forms words by sequentially adding suffixes to a root and follows a Subject-Object-Verb (SOV) word order.

### Sample

There follows a randomly selected sample of five sentences from the corpus.

1. *Kuraq runataqa ñapaykunallaykipuni.*
2. *Nuqaqa t'anta masaytan munani*
3. *Chay quwiyki kirunwan kanirquwan.*
4. *Mana runaq uywasqan quwiqa sallqa kapun.*
5. *Ñañay unuman rirqapuway*

### Sources

The sentences were created by native speakers, whose uploaded the sentences into Common Voice.

| Source | Sentences |
|---|---|
| Elwin Huaman | 1,671 (80.8%) |
| Jorge Huaman Quispe | 394 (19.1%) |
| Other | 2 (0.1%) |

### Text domains

The domain of the sentences are under: 
- General
- Agriculture and Food
- Nature and Environment

| Code | Domain | Clips | Speakers |
|---|---|---|---|
| general | General | 19,828 (78.0%) | 80 (98.8%) |
| agriculture_food | Agriculture and Food | 25,298 (99.5%) | 81 (100.0%) |
| automotive_transport | Automotive and Transport | - | - |
| finance | Finance | 112 (0.4%) | 32 (39.5%) |
| service_retail | Service and Retail | - | - |
| healthcare | Healthcare | 19,867 (78.1%) | 81 (100.0%) |
| history_law_government | History, Law and Government | 1,343 (5.3%) | 61 (75.3%) |
| media_entertainment | Media and Entertainment | - | - |
| nature_environment | Nature and Environment | 9,770 (38.4%) | 77 (95.1%) |
| news_current_affairs | News and Current Affairs | 161 (0.6%) | 42 (51.9%) |
| technology_robotics | Technology and Robotics | - | - |
| language_fundamentals | Language Fundamentals | - | - |

### Fields

#### Clips

Each row of a `tsv` file represents a single audio clip, and contains the following information:

- `client_id` - hashed UUID of a given user
- `path` - relative path of the audio file
- `sentence` - the sentence to be read aloud
- `sentence_id` - unique identifier for the sentence
- `sentence_domain` - domain classification(s) of the sentence
- `up_votes` - number of people who said audio matches the text
- `down_votes` - number of people who said audio does not match text
- `age` - age of the speaker[^1]
- `gender` - gender of the speaker[^1]
- `accents` - accents of the speaker[^1]
- `variant` - variant of the language[^1]
- `locale` - locale code of the language
- `segment` - if sentence belongs to a custom dataset segment, it will be listed here

[^1]: For a full list of age, gender, and accent options, see the [demographics spec](https://github.com/common-voice/common-voice/blob/main/web/src/stores/demographics.ts). These will only be reported if the speaker opted in to provide that information.

#### `validated_sentences.tsv`

The `validated_sentences.tsv` file contains one row per validated sentence in the text corpus:

- `sentence_id` - unique identifier for the sentence
- `sentence` - the sentence text
- `variant` - the variant of the language
- `sentence_domain` - the domain(s) the sentence belongs to
- `source` - the source the sentence was collected from
- `is_used` - whether the sentence is still in circulation for recording
- `clips_count` - number of clips recorded for this sentence

#### `unvalidated_sentences.tsv`

The `unvalidated_sentences.tsv` file contains one row per unvalidated sentence in the text corpus:

- `sentence_id` - unique identifier for the sentence
- `sentence` - the sentence text
- `variant` - the variant of the language
- `sentence_domain` - the domain(s) the sentence belongs to
- `source` - the source the sentence was collected from
- `up_votes` - number of upvotes the sentence received
- `down_votes` - number of downvotes the sentence received
- `status` - current status of the sentence (`pending` or `rejected`)

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

- [Speak](https://commonvoice.mozilla.org/qxp/speak)
- [Write](https://commonvoice.mozilla.org/qxp/write)
- [Listen](https://commonvoice.mozilla.org/qxp/listen)
- [Review](https://commonvoice.mozilla.org/qxp/review)

## Acknowledgements

### Datasheet authors

* Elwin Huaman <elwin.huaman@hotmail.com>

### Citation guidelines

If you use this data in your research, please cite the following paper:

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
