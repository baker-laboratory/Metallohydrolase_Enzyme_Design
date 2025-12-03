#!/bin/bash
#SBATCH --job-name=ZETA_2__DFTqc_theozyme_TSopt_Gaussian
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=/home/woodbuse/publication/metallohydrolase/github/Computational_Design_of_Metallohydrolases_PrivateGitHub/RFdiffusion2_Tutorial/logs/slurm_%j.out
#SBATCH --error=/home/woodbuse/publication/metallohydrolase/github/Computational_Design_of_Metallohydrolases_PrivateGitHub/RFdiffusion2_Tutorial/logs/slurm_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=EXAMPLE_EMAIL@uw.edu
#SBATCH --chdir=/home/woodbuse/publication/metallohydrolase/github/Computational_Design_of_Metallohydrolases_PrivateGitHub/RFdiffusion2_Tutorial/theozymes/from_quantum_chemistry/Gaussian_TSopt

### EXPORT GAUSSIAN ###
export GAUSS_EXEDIR='/net/software/gaussian/g16/'

### EXPORT TEMPORARY SCRATCH DIRECTORY ###
SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/gauss_scratch"
mkdir -p "$SCRATCH" || { echo "Failed to make $SCRATCH"; exit 1; }
export GAUSS_SCRDIR="$SCRATCH"

### GATHER INFO ###
echo "Running on node: $HOSTNAME"
echo "Scratch dir: $GAUSS_SCRDIR"
echo "This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $END_NUM"
lscpu | grep "Model name"
free -h

### EXECUTE GAUSSIAN TASK ###
filename="ZETA_2__DFTqc_theozyme_TSopt_Gaussian"
/net/software/gaussian/g16/g16 < "$filename.com" > "$filename.log"
