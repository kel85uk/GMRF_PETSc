SIGMA1=(0 1 2 3 4 5 6 7 8 9)
SIGMA2=(1 2 4 6 8) #Decimals
LAMBDA=( 0.1 0.15 0.2 0.25 0.3 0.35 0.4 ) #Lambda
PCG=( hypre icc ilu )
SOLVER=( fgmres bcgs )

# Of the form -sigma ${SIGMA1}.${SIGMA2}

# Loop through the different seed values and submit a run for each

#for SIG1 in ${SIGMA1[@]}
#do
#for SIG2 in ${SIGMA2[@]}
#do
#for LAM in "${LAMBDA[@]}"
#do
#        SIG="${SIG1}.${SIG2}"
for SOLVER1 in ${SOLVER[@]}
do
for PCG1 in ${PCG[@]}
do
        # set the job name
        NAME="Test5_${PCG1}_${SOLVER1}_32SLMC1_500x500_2000s"
        echo "Submitting: ${NAME}"

        # Build a string called PBS which contains the instructions for your run
        # This requests 1 node for 1 hour. Runs a program called "my_program" with an argument.

        PBS="#!/bin/bash\n\
        #PBS -l nodes=1:ppn=32\n\
        #PBS -l walltime=08:00:00\n\
        #PBS -q normal\n\
        #PBS -M <login>@i10.informatik.uni-erlangen.de -m abe\n\
        #PBS -N ${NAME}\n\
        . /etc/profile.d/modules.sh\n\
        module load gcc openmpi cuda boost cmake\n\
        cd /home/stud/ug30owag/Documents/GMRF_PETSc\n\
        mpirun -np 32 --report-bindings --bind-to-core ./Test5 -ppc 1 -mat_type seqaij -vec_type seq -Nsamples 2000 -m 500 -n 500 -pc_type ${PCG1} -ksp_type ${SOLVER1} -dim 2 -alpha 2 -lamb 0.1 -sigma 0.3 -TOL 1e-10"
        # Note that $PBS_O_WORKDIR is escaped ("\"). We don't want bash to evaluate this variable right now. Instead it will be evaluated when the command runs on the node.
#mpirun -np 32 --report-bindings --bind-to-core ./Test5 -ppc 1 -mat_type seqaij -vec_type seq -Nsamples 2000 -m 500 -n 500 -pc_type hypre -pc_hypre_boomeramg_max_iter 2 -ksp_type richardson -ksp_max_it 500 -dim 2 -alpha 2 -lamb ${LAM} -sigma ${SIG} -TOL 1e-10
        # Echo the string PBS to the function qsub, which submits it as a cluster job for you
        # A small delay is included to avoid overloading the submission process

#        echo -e ${PBS} | qsub
        sleep 1.0
        echo "done."

done
done
#done
