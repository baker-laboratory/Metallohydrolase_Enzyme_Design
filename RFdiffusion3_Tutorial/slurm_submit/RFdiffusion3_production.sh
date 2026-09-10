#!/bin/bash
#SBATCH -J RFdiffusion3_production
#SBATCH -p gpu
#SBATCH -c 1
#SBATCH --gres=gpu:a4000:1
#SBATCH --mem=16g
#SBATCH -t 06:30:00
#SBATCH -o /path/to/Metallohydrolase_Enzyme_Design/RFdiffusion3_Tutorial/logs/RFdiffusion3_production_%a.stdout
#SBATCH -e /path/to/Metallohydrolase_Enzyme_Design/RFdiffusion3_Tutorial/logs/RFdiffusion3_production_%a.stderr
#SBATCH -a 1-5

PER_TASK=15
START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))
END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))
echo This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $END_NUM
for (( run=$START_NUM; run<=END_NUM; run++ )); do
  echo This is SLURM task $SLURM_ARRAY_TASK_ID, run number $run
  CMD=$(sed -n "${run}p" /path/to/Metallohydrolase_Enzyme_Design/RFdiffusion3_Tutorial/cmds/RFdiffusion3_production)
  echo "${CMD}" | bash
done
