#!/bin/bash
#BSUB -J MNIST
#BSUB -o MNIST_%J.out
#BSUB -q hpc
#BSUB -W 04:00
#BSUB -R "rusage[mem=16GB] span[hosts=1]"

module load python3/3.9.6
module load cuda/11.3
source venv/bin/activate

python src/models/train_tyxe.py --multirun ++params.epochs=200 ++params.guide=radial,meanfield,laplace,map,ml,lowrank
