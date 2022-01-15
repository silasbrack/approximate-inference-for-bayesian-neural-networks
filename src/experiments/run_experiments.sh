#!/bin/bash
#BSUB -q hpcintro

source venv/bin/activate

BATCH_SIZES="16 64 128 256 512"

for BS in $BATCH_SIZES
do
  python src/models/train_model.py ++params.batch_size="$BS" ++params.epochs=5;
done
