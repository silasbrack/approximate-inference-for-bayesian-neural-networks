#!/bin/bash
#BSUB -J tyxe
#BSUB -o tyxe_%J.out
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB] span[hosts=1]"

module load python3/3.9.6
module load cuda/11.3
source venv/bin/activate

python src/models/train_tyxe.py \
    --multirun \
    training.cache_data=true \
    hardware.gpus=1 \
    training.dataset=mura \
    eval.datasets=[mura] \
    training.epochs=500 \
    training.guide=radial,meanfield,laplace,map,lowrank
