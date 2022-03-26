#!/bin/bash
#BSUB -J test
#BSUB -o test_%J.out
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB] span[hosts=1]"

#module load python3/3.9.6
#module load cuda/11.3
#source venv/bin/activate

python src/train.py \
    --multirun \
    inference=vi
    inference/variational_family=radial,meanfield,lowrank

python src/train.py \
    --multirun \
    inference=swa,swag,multi_swag

python src/train.py \
    --multirun \
    inference=nn,deep_ensemble
