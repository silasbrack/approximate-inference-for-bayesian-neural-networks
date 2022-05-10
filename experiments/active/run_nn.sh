#!/bin/bash
#BSUB -J active
#BSUB -o active_%J.out
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB] span[hosts=1]"

module load python3/3.9.6
module load cuda/11.3
source venv/bin/activate

python src/train_active.py \
    --multirun \
    data=mura \
    training.lr=3e-4 \
    training.epochs=50 \
    training.active_queries=100 \
    training.initial_pool=50 \
    training.query_size=10 \
    acquisition=random,max_entropy \
    inference=nn \
    inference/model=convnet \
    inference.model.num_classes=7 \
    inference.device=cuda
