#!/bin/bash

#SBATCH --job-name=save_rect
#SBATCH --array=0-191
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH -c8
#SBATCH --output="/om/user/ssazaidi/spike_proc/save_rect-%A_%a.out"
#SBATCH --mem=10000
#SBATCH -pdicarlo
#SBATCH --exclude=node052,node098,node080,node074,node037

hostname

source /om/user/ssazaidi/miniconda3/etc/profile.d/conda.sh
conda activate dicarlo_lab

cd /braintree/data2/active/users/ssazaidi/spike-tools/spike_tools


python save_rectified_trace.py -ch $SLURM_ARRAY_TASK_ID ${*:1}