#!/bin/bash
#BSUB -J deep_ensemble
#BSUB -o deep_ensemble_%J.out
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB] span[hosts=1]"

module load python3/3.9.6
module load cuda/11.3
source venv/bin/activate

python src/models/train_deep_ensemble.py \
    --multirun \
    ++hardware.gpus=1 \
    ++training.epochs=100 \
    ++num_ensembles=5,10
