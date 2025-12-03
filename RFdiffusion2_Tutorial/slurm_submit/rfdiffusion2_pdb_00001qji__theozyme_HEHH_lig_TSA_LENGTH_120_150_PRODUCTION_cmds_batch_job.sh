        #!/bin/bash
        #SBATCH -J rfdiffusion2_pdb_00001qji__theozyme_HEHH_lig_TSA_LENGTH_120_150_PRODUCTION_cmds_batch_job
        #SBATCH -p gpu
        #SBATCH -c 1
#SBATCH --gres=gpu:a4000:1
        #SBATCH --mem=8g
        #SBATCH -t 06:00:00
        #SBATCH -o /home/woodbuse/publication/metallohydrolase/github/Computational_Design_of_Metallohydrolases_PrivateGitHub/RFdiffusion2_Tutorial/logs/rfdiffusion2_pdb_00001qji__theozyme_HEHH_lig_TSA_LENGTH_120_150_PRODUCTION_cmds_batch_job_%a.stdout
        #SBATCH -e /home/woodbuse/publication/metallohydrolase/github/Computational_Design_of_Metallohydrolases_PrivateGitHub/RFdiffusion2_Tutorial/logs/rfdiffusion2_pdb_00001qji__theozyme_HEHH_lig_TSA_LENGTH_120_150_PRODUCTION_cmds_batch_job_%a.stderr
        #SBATCH -a 1-2

        PER_TASK=18
        START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))
        END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))
        echo This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $END_NUM
        for (( run=$START_NUM; run<=END_NUM; run++ )); do
          echo This is SLURM task $SLURM_ARRAY_TASK_ID, run number $run
          CMD=$(sed -n "${run}p" /home/woodbuse/publication/metallohydrolase/github/Computational_Design_of_Metallohydrolases_PrivateGitHub/RFdiffusion2_Tutorial/cmds/rfdiffusion2_pdb_00001qji__theozyme_HEHH_lig_TSA_LENGTH_120_150_PRODUCTION_cmds
        )
          echo "${CMD}" | bash
        done
