#!/bin/bash
#
# CompecTA (c) 2018
#
# NAMD job submission script
#
# TODO:
#   - Set name of the job below changing "NAMD" value.
#   - Set the requested number of nodes (servers) with --nodes parameter.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter. (Total accross all nodes)
#   - Select the partition (queue) you want to run the job in:
#     - short : For jobs that have maximum run time of 120 mins. Has higher priority.
#     - mid   : For jobs that have maximum run time of 1 days. Lower priority than short.
#     - long  : For jobs that have maximum run time of 7 days. Lower priority than long.
#     - longer: For testing purposes, queue has 15 days limit but only 2 nodes.
#     - cuda  : For CUDA jobs. Solver that can utilize CUDA acceleration can use this queue. 7 days limit.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input/output file names below.
#   - If you do not want mail please remove the line that has --mail-type and --mail-user. If you do want to get notification emails, set your email address.
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch slurm_submit.sh
#
# -= Resources =-
#
#SBATCH --job-name=DKZ
#SBATCH --account=cuda
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=50G
### SBATCH --mem-per-cpu=10000
### SBATCH --cpus-per-task=4
### SBATCH --ntasks-per-node=1
#SBATCH --qos=cuda
#SBATCH --partition=cuda
#SBATCH --time=7-0
#SBATCH --output=slurm_job_outputs/%j-job.out
#SBATCH --mail-type=ALL

# # GPU request
#SBATCH --gres=gpu:1
#SBATCH -w cn07

#INPUT_FILE="5_BERT_Siamese_NeuralNet.py"
INPUT_FILE="temp.py"
USER="easunar"

################################################################################

#Module File
echo "Loading Anaconda..."
##module load cuda/10.2
##module load anaconda052021
# Activate Anaconda work environment for OpenDrift
##module load cuda/9.2
source /cta/users/${USER}/.bashrc
source activate DKZ_pytorch

echo ""
echo "======================================================================================"
env
echo "======================================================================================"
echo ""

echo "======================================================================================"
# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

echo "Running Python script..."
echo "==============================================================================="

# Put Python script command below
# srun python -u Siamese4Word2Vec.py --a 3 --b 7 &>job_output
#VARIABLE=`srun python3 -u Sick_Siamese_butnotsickdata.py --a 3 --b 7 `
COMMAND="python $INPUT_FILE"
COMMAND="python3 main.py --MODE train --DEVICE cuda"

echo ${COMMAND}
echo "-------------------------------------------"
$COMMAND

RET=$?
echo
echo "Solver exited with return code: $RET"
exit $RET
