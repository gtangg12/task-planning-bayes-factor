#!/bin/bash
#SBATCH -J fine_tune_lm
#SBATCH -o fine_tune_lm.out
#SBATCH -e fine_tune_lm.err
#SBATCH --mail-user=gtangg12@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=128GB
#SBATCH --time=8:00:00
#SBATCH --exclusive

## User python environment
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=task_planning_babyai
CONDA_ROOT=$HOME2/anaconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE


####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date

## Horovod execution
horovodrun -np $SLURM_NTASKS -H `cat $NODELIST` python /nobackup/users/gtangg12/task_planning_bayes_factor/finetune_task_sequence_lm.py

echo "Run completed at:- "
date