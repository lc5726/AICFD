#!/bin/bash
#SBATCH --job-name=my_fluent_job
#SBATCH --output=fluent_output.txt
#SBATCH --error=fluent_error.txt
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --cpus-per-task=1
#SBATCH --mem=30Gb
#SBATCH --time=30:00:00  # Set the maximum runtime for your job

#SBATCH --mail-type=BEGIN,END   # Send an email when the job finishes
#SBATCH --mail-user=lovish.chopra@mail.concordia.ca 

module load ansys/2022R2/default
rm -rf *.trn*
rm -rf *cleanup*
pwd
fluent 2ddp -t30 -g -i fluent_run.jou -scheduler_pe=smp
rm -rf *.trn*
rm -rf *cleanup*
rm -rf *.log*




