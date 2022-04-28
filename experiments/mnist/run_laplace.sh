#!/bin/bash
#BSUB -J laplace
#BSUB -o laplace_%J.out
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB] span[hosts=1]"

module load python3/3.9.6
module load cuda/11.3
source venv/bin/activate

python src/train.py \
    --multirun \
    data=mnist \
    training.epochs=100 \
    inference=nn,laplace \
    inference/model=convnet \
    inference.posterior_samples=32 \
    inference.device=cuda
