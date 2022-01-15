#!/bin/bash
#BSUB -q hpcintro

./venv/bin/activate

BATCH_SIZES="1 16 64 128 256 512"

for BS in $BATCH_SIZES
do
  python src/models/train_model.py ++params.batch_size="$BS" ++params.epochs=1;
done
