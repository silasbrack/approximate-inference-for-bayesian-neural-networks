#!/bin/bash
#BSUB -J active_vi
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
    training.epochs=30 \
    training.active_queries=100 \
    training.initial_pool=50 \
    training.query_size=10 \
    acquisition=random,max_entropy,bald \
    inference=vi \
    data.batch_size=8192 \
    inference/model=convnet \
    inference.model.num_classes=7 \
    inference/variational_family=radial \
    inference.device=cuda
