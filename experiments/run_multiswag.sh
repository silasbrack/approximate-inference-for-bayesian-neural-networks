#!/bin/bash
#BSUB -J multiswag
#BSUB -o multiswag_%J.out
#BSUB -q hpc
#BSUB -W 12:00
#BSUB -R "rusage[mem=16GB] span[hosts=1]"

module load python3/3.9.6
module load cuda/11.3
source venv/bin/activate

python src/models/train_multiswag.py \
    --multirun \
    ++training.epochs=200 \
    ++num_ensembles=5,10,20
