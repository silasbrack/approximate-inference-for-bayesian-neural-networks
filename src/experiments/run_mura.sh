#!/bin/bash
#BSUB -J MURA
#BSUB -o MURA_%J.out
#BSUB -q hpcintrogpu
#BSUB -W 00:45
#BSUB -R "rusage[mem=16384MB] span[hosts=1]"

module load python3/3.9.6
source venv/bin/activate

python src/models/train_mnist_tyxe.py ++params.epochs=20 >> results.txt
