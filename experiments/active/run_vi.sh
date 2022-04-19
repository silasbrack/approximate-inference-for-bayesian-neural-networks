#!/bin/bash
#BSUB -J active
#BSUB -o active_%J.out
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB] span[hosts=1]"

python src/train_active.py \
    --multirun \
    data=mnist \
    training.epochs=100 \
    data.batch_size=8192 \
    inference=vi \
    inference/variational_family=mean_field,radial \
    inference.device=cuda
