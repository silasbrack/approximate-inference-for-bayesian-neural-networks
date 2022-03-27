#!/bin/bash
#BSUB -J multiswag
#BSUB -o multiswag_%J.out
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
    training.epochs=20 \
    inference=multi_swag \
    inference.device=cuda \
    inference.num_ensembles=5,10 \
    inference.swa_start_thresh=0.8
