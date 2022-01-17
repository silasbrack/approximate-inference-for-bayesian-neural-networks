#!/bin/bash
#BSUB -J MNIST
#BSUB -o MNIST_%J.out
#BSUB -q hpcintro
#BSUB -W 01:00
#BSUB -R "rusage[mem=2048MB] span[hosts=1]"

module load python3/3.9.6
source venv/bin/activate

make data

BATCH_SIZES="16 64 128 256 512 1024"

for BS in $BATCH_SIZES
do
  python src/models/train_model.py ++params.batch_size="$BS" ++params.epochs=200;
done
