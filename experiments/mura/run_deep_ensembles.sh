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

python src/train.py \
    --multirun \
    data=mura \
    training.epochs=100 \
    inference=deep_ensemble \
    inference.device=cuda \
    inference.model.num_classes=7 \
    inference.num_ensembles=5,10
