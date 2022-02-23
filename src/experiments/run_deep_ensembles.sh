#!/bin/bash
#BSUB -J MNIST
#BSUB -o MNIST_%J.out
#BSUB -q hpc
#BSUB -W 04:00
#BSUB -R "rusage[mem=16GB] span[hosts=1]"

#module load python3/3.9.6
#source venv/bin/activate

EPOCHS=10

ENSEMBLES="5 10 20"
for NUM_ENSEMBLES in $ENSEMBLES
do
  python src/models/train_mnist_tyxe.py ++params.epochs=$EPOCHS ++params.num_ensembles="$NUM_ENSEMBLES"
done
