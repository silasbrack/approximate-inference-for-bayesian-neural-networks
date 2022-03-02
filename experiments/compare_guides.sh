#!/bin/bash
#BSUB -J MNIST
#BSUB -o MNIST_%J.out
#BSUB -q hpc
#BSUB -W 04:00
#BSUB -R "rusage[mem=16GB] span[hosts=1]"

#module load python3/3.9.6
#source venv/bin/activate

EPOCHS=200

GUIDES="radial meanfield laplace map ml lowrank"
for GUIDE in $GUIDES
do
  python src/models/train_tyxe.py ++params.epochs=$EPOCHS ++params.guide="$GUIDE"
done
