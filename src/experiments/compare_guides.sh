#!/bin/bash
#BSUB -J MNIST
#BSUB -o MNIST_%J.out
#BSUB -q hpcintro
#BSUB -W 00:15
#BSUB -R "rusage[mem=1024MB] span[hosts=1]"

#module load python3/3.9.6
#source venv/bin/activate

EPOCHS=50

GUIDES="ml map laplace meanfield lowrank radial"
for GUIDE in $GUIDES
do
  python src/models/train_mnist_tyxe.py ++params.epochs=$EPOCHS ++params.guide="$GUIDE" >> results.txt
done
