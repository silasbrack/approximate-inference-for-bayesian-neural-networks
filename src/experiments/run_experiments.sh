#!/bin/bash
#BSUB -J MNIST
#BSUB -o MNIST_%J.out
#BSUB -q hpcintro
#BSUB -W 00:15
#BSUB -R "rusage[mem=1024MB] span[hosts=1]"

module load python3/3.9.6
source venv/bin/activate

BATCH_SIZES="16 64 128 256 512"

for BS in $BATCH_SIZES
do
  python src/models/train_model.py ++params.batch_size="$BS" ++params.epochs=5;
done
