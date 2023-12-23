#!/bin/bash
#SBATCH --job-name=my_fluent_job
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=30Gb
#SBATCH --time=30:00:00  # Set the maximum runtime for your job

#SBATCH --mail-type=BEGIN,END   # Send an email when the job finishes
#SBATCH --mail-user=lovish.chopra@mail.concordia.ca 

module load ansys/2022R2/default
angles=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)
machs=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)

# Loop through the array

for mach in "${machs[@]}"; do
    for ang in "${angles[@]}"; do
        echo "Processing AoA :$ang, Mach:$mach "
        rm -rf *.trn*
        rm -rf *cleanup*
        python BC_Setup.py $mach $ang
        fluent 2ddp -t36 -g -i fluent_run.jou -scheduler_pe=smp
        rm -rf *.trn*
        rm -rf *cleanup*
        #rm -rf *.log*
    done
    echo "Breaking the loop"
    break
done





