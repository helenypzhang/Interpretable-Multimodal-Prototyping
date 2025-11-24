#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -t 4-0         # Job "wallclock" is required. Max permitted is 4 days (4-0)
#SBATCH -n 8           # Number of CPU cores

echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

# Activate conda environment
source activate mmdp

# Run your GAN training script
bash scripts/Survival/umeml_gan.sh

# Deactivate conda
conda deactivate
