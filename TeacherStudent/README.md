# Teacher-student + KenLM para quechua puneño (SIMBig 2026, Task 1)

Pipeline de ASR basado en **wav2vec 2.0 (XLS-R 300M) con CTC**, entrenado con transcripciones curadas y datos *silver* (pseudo-etiquetas) filtrados, y decodificado con **beam search + modelo de lenguaje KenLM**. Es una alternativa al baseline de fairseq del resto del repositorio.

## Contenido de la carpeta

| Archivo | Para qué sirve |
|---|---|
| `confidence_aware_pseudo_labeling_simbig2026.py` | Simulación didáctica (datos **sintéticos**) del protocolo teacher-student con pseudo-etiquetas ponderadas por confianza. Proviene del asesor. Sus cifras no son resultados del reto. |
| `run_qxp.py` | Implementación real: datos, entrenamiento de tres recetas, evaluación, decodificación con LM y generación de `qxp.tsv`. |
| `run_qxp.sbatch` | Plantilla SLURM para correrlo en un cluster (partición y límites por verificar). |
| `setup_gpus.sh` | Copia del script del baseline para otra máquina (UUID de GPU de `mkt1`). **No se usa** aquí. |

## Qué hace `run_qxp.py`

1. **Datos.** Usa los splits del repo (`data/splits`) y el texto curado de `data/shared-task/validated_sentences_curated.tsv` para el habla leída. Normaliza a NFC, minúsculas, sin puntuación y **conserva `'` y `ñ`**, como exige el reto.
2. **Silver sin contaminación.** El `silver_spontaneous/train.tsv` original contiene 370 de los 371 clips del test espontáneo y un 40% de clips de hablantes de dev/test. El script los elimina: `5025 → 2570` clips (quitados 2013 de dev/test y 443 con etiqueta validada), `2469` tras el filtro de duración (8.68 h).
3. **Tres recetas**, todas desde `facebook/wav2vec2-xls-r-300m` (sin CPT):
   - `validated`: leído (texto curado) + espontáneo validado ×3.
   - `silver_conf`: `validated` + silver filtrado. El teacher es el modelo `validated`; cada clip silver se puntúa por el acuerdo entre la transcripción del teacher y la etiqueta silver (desempate por confianza CTC) y se conserva el 70% mejor (`--conf-keep`).
   - `silver`: `validated` + todo el silver limpio.
4. **Entrenamiento.** 12 épocas, `lr=1e-4`, batch efectivo 32, bf16, gradient checkpointing, aumento de datos (velocidad, ganancia, ruido). Reanuda desde el último checkpoint.
5. **Evaluación honesta** en dev y test propios; la receta y el decoder se eligen **por tipo de clip según dev**, nunca según test.
6. **Predicción del test oficial** enrutada por la columna `type` de `qxp_test_dataset.tsv`, y escritura de `qxp.tsv` + `submission_qxp.zip` (renómbralo a `qxp.zip` para el formulario).

## Decodificación con KenLM (`--lm`, apagada por defecto)

Sin `--lm` el script decodifica en *greedy* (argmax CTC). Con `--lm` hace beam search con `pyctcdecode` + KenLM solo en `evaluate()` y `predict()`; el filtro de `silver_conf` sigue en greedy.

- Para cada receta y tipo de clip se barren `alpha` y `beta` en **dev** (`--lm-alpha`, `--lm-beta`; por defecto `0.5,1.0,1.5` y `1.0,3.0`), se aplica la mejor combinación al test y `choose()` decide entre greedy y LM según dev.
- `--lm-tune` permite usar un LM entrenado **sin texto de dev** para elegir `alpha`/`beta` sin fuga; `--lm` es el LM final.
- Los LM de 4-gramas están en `../eval/kenlm/` (`qxp_4gram.bin` con todo el texto, `qxp_tune.bin` sin `dev` ni la lista curada) y se generan con `../eval/kenlm/build_corpus.py` (`--exclude-dev` para el de tuning) más `lmplz`/`build_binary`.
- Las etiquetas de `pyctcdecode` se construyen fundiendo `[PAD]`, `[UNK]`, `<s>` y `</s>` en el blank CTC y `|` en el espacio.

## Uso

```bash
# 1. Una sola vez, con internet: modelo base en caché
python run_qxp.py fetch

# 2. Prueba rápida (minutos): valida entorno, MP3 y GPU
python run_qxp.py run --smoke

# 3. Corrida completa con LM (en segundo plano)
nohup python run_qxp.py run \
    --lm eval/kenlm/qxp_4gram.bin --lm-tune eval/kenlm/qxp_tune.bin \
    --lm-unigrams eval/kenlm/corpus.txt > logs/full_lm.log 2>&1 &

# 4. Re-evaluar/predecir con modelos ya entrenados (sin reentrenar)
python run_qxp.py run --no-train --lm eval/kenlm/qxp_4gram.bin \
    --lm-tune eval/kenlm/qxp_tune.bin --lm-unigrams eval/kenlm/corpus.txt
```

Se ejecuta desde la raíz del repo (`asr-puno-quechua/`), con `data/scripted`, `data/spontaneous`, `data/test` (o `--scripted-dir`, `--spont-dir`, `--test-dir`).

**Entorno usado:** WSL2, Python 3.14, PyTorch (CUDA 12.8), `transformers`, `jiwer`, `librosa`, `pyctcdecode` y `kenlm`.
- `kenlm` hay que recompilarlo con Cython reciente (`cython --cplus python/kenlm.pyx`) porque el `kenlm.cpp` publicado no compila con Python 3.14.
- `pyctcdecode` se instala con `--no-deps` (declara `numpy<2`, pero funciona con numpy 2).
- Con Python 3.14 `main()` fuerza el método `fork` de multiprocessing; si no, falla al serializar el Dataset.

## Resultados (dev, 2831 clips leídos y 542 espontáneos)

| Receta | Leído greedy | Leído + LM | Espontáneo greedy | Espontáneo + LM |
|---|---|---|---|---|
| `validated` | 25.1 / 4.40 | **10.9 / 2.39** | 34.8 / 5.42 | 20.4 / 3.52 |
| `silver_conf` | 25.5 / 4.37 | 11.5 / 2.41 | 33.3 / 5.05 | 18.4 / 3.15 |
| `silver` | 24.8 / 4.30 | 11.8 / 2.45 | 31.7 / 4.81 | **17.2 / 2.95** |

*(WER % / CER %)*. Decoder y receta elegidos por dev: leído → `validated` + LM; espontáneo → `silver` + LM (`alpha=0.5`, `beta=1.0`).

**Conclusiones**
- El **LM reduce el WER a menos de la mitad** respecto al greedy en ambos tipos de habla.
- El **silver limpio** mejora el espontáneo ~3 puntos de WER y no cambia el leído.
- **Filtrar por confianza no superó a usar todo el silver**, aunque el filtro separa bien (acuerdo teacher-silver de 94.7% en los conservados frente a 74.7% en los descartados; Spearman confianza vs −CER = 0.454 en dev espontáneo).

**Frente al baseline de fairseq (con CPT) + KenLM**, en los mismos clips de dev:

| | Baseline + LM | Este pipeline + LM |
|---|---|---|
| Leído, referencias curadas | 13.74 / 2.40 | **10.88 / 2.39** |
| Espontáneo (WER) | **13.82** | 17.25 |

El baseline se entrenó con texto crudo y sube de 5.27% a 13.74% de WER al puntuarse con referencias curadas (el 29% de las oraciones leídas cambian al curarse). Este pipeline gana en habla leída por usar texto curado; el baseline gana en espontánea. Por eso la entrega final es **híbrida** (ver `../entrega_hibrida/`): leídos de este pipeline, espontáneos del baseline.

## Advertencias sobre las cifras

- Las métricas **de test propio con LM** (1.5% leído, 7.0% espontáneo) **no deben citarse**: el LM final incluye la lista curada (todas las oraciones leídas, test incluido) y el silver original (370 de 371 clips de test espontáneo con pseudo-etiquetas).
- El espontáneo/dev con LM es algo optimista para todos los modelos por igual: el LM de tuning contiene el silver de 383 de los 542 clips de dev espontáneo. La comparación entre modelos sigue siendo válida; los valores absolutos no.
- El test ciego del reto son hablantes nuevos, por lo que se esperan números peores que los de dev.
- Una sola semilla (`--seed 42`), sin intervalos de confianza.

## Costos de referencia (RTX 5070 12 GB)

`validated` 3.1 h, `silver_conf` 3.5 h, corrida completa con evaluación y barrido del LM ≈ 10.4 h.

## Trabajo futuro

- Que el teacher regenere las pseudo-etiquetas y use la confianza como **peso** y regularización de consistencia; aquí el silver llega etiquetado de fuera y la confianza solo filtra.
- Partir de un checkpoint con CPT en lugar de XLS-R plano.
- LM entrenado con la normalización exacta de `run_qxp.py` y vocabulario de morfemas o caracteres, dada la alta proporción de palabras que aparecen una sola vez (~52% del vocabulario).
- Varias semillas e intervalos de confianza.
