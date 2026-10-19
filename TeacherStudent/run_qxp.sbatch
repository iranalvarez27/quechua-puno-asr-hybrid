#!/bin/bash
# Plantilla para Khipu (SLURM). Copiar a ~/asr-puno-quechua/ junto con run_qxp.py y enviar con sbatch.
#
# Antes de enviarlo, verifica con (comandos SLURM estándar):
#   sinfo -o "%P %G %l %D"     -> particiones, tipo de GPU y límite de tiempo
#   quota -s ; df -h ~         -> espacio en tu home
# La documentación pública de Khipu solo confirma las particiones "debug" y "debug-gpu" (30 min);
# el nombre "gpu" sale de un ejemplo de Apptainer. Cámbialo si sinfo muestra otro.
#
# Ejemplos:
#   Prueba rápida con datos reales (valida entorno, mp3, GPU):
#     sbatch --partition=debug-gpu --time=00:30:00 --export=ALL,RUN_ARGS="--smoke" run_qxp.sbatch
#   Corrida completa:
#     sbatch run_qxp.sbatch
#   Dos trabajos en paralelo (si te dan 2 GPUs) y luego evaluar/predecir todo junto:
#     sbatch --export=ALL,RUN_ARGS="--recipes validated,silver_conf --epochs 8 --time-budget-h 10" run_qxp.sbatch
#     sbatch --export=ALL,RUN_ARGS="--recipes silver --epochs 8 --time-budget-h 10" run_qxp.sbatch
#     sbatch --export=ALL,RUN_ARGS="--no-train" run_qxp.sbatch      # cuando terminen los dos
#
#SBATCH --job-name=qxp
#SBATCH --partition=gpu            # CAMBIAR si sinfo muestra otro nombre
#SBATCH --gres=gpu:1               # si sinfo muestra tipos (a6000, a100), puedes pedir gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00            # AJUSTAR al máximo que permita tu cuenta
#SBATCH --requeue                  # si lo reinician, el entrenamiento retoma desde el último checkpoint
#SBATCH --output=logs/qxp_%j.out

set -euo pipefail
cd "$HOME/asr-puno-quechua"
mkdir -p logs

source "$HOME/venvs/qxp/bin/activate"

# Los nodos de cómputo NO tienen internet: el modelo base debe estar en caché (python run_qxp.py fetch).
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Si hace falta ffmpeg para leer mp3, busca el nombre exacto con:  module avail ffmpeg
# module load ffmpeg

nvidia-smi
python run_qxp.py run ${RUN_ARGS:-}
