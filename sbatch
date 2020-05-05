#!/bin/bash
#SBATCH -n 8
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=0-01:00:00

module add cuda/10.1 cudnn/7.6-cuda-10.0

python main.py --train --epochs 40