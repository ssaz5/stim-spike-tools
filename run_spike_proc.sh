#!/bin/bash

#SBATCH --job-name=spk_proc
#SBATCH --array=0-191
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH -c4
#SBATCH --output="/om/user/ssazaidi/spike_proc/slurm-%A_%a.out"
#SBATCH --mem=10000
#SBATCH --exclude=node052,node098,node080,node021,node006

hostname

source /om/user/ssazaidi/miniconda3/etc/profile.d/conda.sh
conda activate dicarlo_lab

cd /braintree/data2/active/users/ssazaidi/spike-tools
python spike_tools/spike_proc.py $SLURM_ARRAY_TASK_ID ${*:1}