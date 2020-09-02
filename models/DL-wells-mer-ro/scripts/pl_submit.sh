#!/bin/bash -l

# queue
#SBATCH --partition=gpu2

# jobname
#SBATCH --job-name=num4_tasks4


#SBATCH --nodes=15
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1


# job time for backfill scheduling
#SBATCH --time=0-00:20:00 #d-hrs:min:sec

# User notify events
#SBATCH --mail-type=all
#SBATCH --mail-user=Rodrigo.Lopez@mpikg.mpg.de

module purge
module load python/3.8.2

# might need  the latest cuda
module load nvidia/cuda/9.1

set
# run script from above
srun ipython pl_train.py
