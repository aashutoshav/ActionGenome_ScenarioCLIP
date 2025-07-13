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
#SBATCH --job-name="s-clip"

spack load cuda/s5o57xp
spack load cudnn
source jobs/conf.sh

CHECKPOINT=""

CUDA_CACHE_DISABLE=1 $PYTHONPATH finetune.py --architecture "scenario_clip" \
    --checkpoint_file $CHECKPOINT \
    --img_dir '/scratch/saali/Datasets/coco/train2017' \
    --data_dir '/scratch/saali/Datasets/action-genome/results_9_10/' \
    --exps_dir '/scratch/'$USER'/exps/scenario-clip' \
    --metadata_json '/scratch/saali/Datasets/action-genome/results_9_10/metadata_63k.json' \
    --classes_json '/scratch/saali/Datasets/action-genome/results_9_10/classes_63k.json' \
    --embedding_store '/scratch/'$USER'/checkpoints/' \
    --batch_size 8
