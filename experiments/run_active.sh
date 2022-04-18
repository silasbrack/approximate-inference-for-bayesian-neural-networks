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
    data=mnist \
    training.epochs=100 \
    inference=nn \
    inference.device=cuda

#python src/train_active.py \
#    --multirun \
#    data=mnist \
#    training.epochs=100 \
#    data.batch_size=8192 \
#    inference=vi \
#    inference/variational_family=mean_field,radial \
#    inference.device=cuda
