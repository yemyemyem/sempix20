#!/bin/bash
#
# Auto-generated by test-tube (https://github.com/williamFalcon/test-tube)
#################

# set a job name
#SBATCH --job-name=grid_testv11
#################

# a file for job output, you can check job progress
#SBATCH --output=/usr/data/bgfs1/rlopez/DL_project/DL-Project/pl_version_cluster/grid_test/slurm_out_logs/trial_11_2020-08-26__14-44-33_slurm_output_%j.out
#################

# a file for errors
#SBATCH --error=/usr/data/bgfs1/rlopez/DL_project/DL-Project/pl_version_cluster/grid_test/slurm_err_logs/trial_11_2020-08-26__14-44-33_slurm_output_%j.err
#################

# time needed for job
#SBATCH --time=0-12:00:00
#################

# gpus per node
#SBATCH --gres=gpu:2
#################

# cpus per job
#SBATCH --cpus-per-task=1
#################

# number of requested nodes
#SBATCH --nodes=11
#################

# memory per node
#SBATCH --mem=0
#################

# slurm will send a signal this far out before it kills the job
#SBATCH --signal=USR1@300
#################


# queue
#SBATCH --partition=gpu2
#################

# Tasks per node
#SBATCH --ntasks-per-node=2
#################

# Mail type
#SBATCH --mail-type=all
#################

# Mail account
#SBATCH --mail-user=Rodrigo.Lopez@mpikg.mpg.de
#################


module purge


module load python/3.8.2


module load nvidia/cuda/9.1


set


srun python3 /usr/data/bgfs1/rlopez/DL_project/DL-Project/pl_version_cluster/pl_train_cluster.py --embed_size 150 --num_layers 2 --batch_size 32 --slurm_log_path /usr/data/bgfs1/rlopez/DL_project/DL-Project/pl_version_cluster --vocab_size 10209 --hidden_size 200 --num_workers 4 --test_tube_from_cluster_hopt --test_tube_slurm_cmd_path /usr/data/bgfs1/rlopez/DL_project/DL-Project/pl_version_cluster/grid_test/slurm_scripts/trial_11_2020-08-26__14-44-33_slurm_cmd.sh --hpc_exp_number 11