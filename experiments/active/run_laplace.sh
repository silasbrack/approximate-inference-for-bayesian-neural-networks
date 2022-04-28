#!/bin/bash
#BSUB -J active_l
#BSUB -o active_%J.out
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB] span[hosts=1]"

module load python3/3.9.6
module load cuda/11.3
source venv/bin/activate

python src/train_active.py \
    data=mnist \
    training.epochs=3 \
    training.active_queries=5 \
    training.initial_pool=50 \
    training.query_size=10 \
    inference=laplace \
    inference/model=convnet \
    inference.device=cuda
