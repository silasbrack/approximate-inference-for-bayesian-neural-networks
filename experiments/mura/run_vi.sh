#!/bin/bash
#BSUB -J vi
#BSUB -o vi_%J.out
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB] span[hosts=1]"

module load python3/3.9.6
module load cuda/11.3
source venv/bin/activate

python src/train.py \
    --multirun \
    data=mura \
    data.batch_size=8192 \
    training.epochs=500 \
    inference=vi \
    inference/model=convnet \
    inference/variational_family=radial,mean_field,low_rank \
    inference.model.num_classes=7 \
    inference.device=cuda
