#!/bin/bash

#SBATCH -p gpu_a100_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem 80000M
#SBATCH -t 0-23:59 # time (D-HH:MM)
#SBATCH -o ./slurm/out%j.txt
#SBATCH -e ./slurm/err%j.txt
#SBATCH --gres=gpu:1
#SBATCH --signal=SIGUSR1@180
#SBATCH --job-name="s-clip-0"

spack load cuda/s5o57xp
spack load cudnn
source jobs/conf.sh

CUDA_CACHE_DISABLE=1 $PYTHONPATH pretrain.py --architecture "scenario_clip_0" \
    --metadata_dir '/scratch/saali/Datasets/action-genome-labels' \
    --batch_size 8 --max_epochs 10 --lr 1e-5
