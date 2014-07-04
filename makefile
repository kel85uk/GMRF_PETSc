
CCFLAGS	         = -std=c++11 -O -I./src/ -D VEC_OUTPUT
FFLAGS	         = -Wall -Wconversion -Wshadow
CPPFLAGS         = -std=c++11 -O -I./src/ -D VEC_OUTPUT
FPPFLAGS         =
#LOCDIR           = ./
#EXAMPLESC        = ex29.c Test.cc test_serial.cc test_serial2.cc Functions.cc
#MANSEC           = KSP
#CLEANFILES       = rhs.vtk solution.vtk
NP               = 1
ALL: test_MC test_PDE Test TestV2
include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${SLEPC_DIR}/conf/slepc_common

ex29: ex29.o  chkopts
	-${CLINKER} -o ex29 ex29.o  ${PETSC_SNES_LIB}
	${RM} ex29.o
	
test_MC: ./src/Functions.o test_serial.o  chkopts
	-${CLINKER} -o test_MC ./src/Functions.o test_serial.o  ${PETSC_SNES_LIB} ${SLEPC_LIB}
	${RM} test_serial.o ./src/Functions.o
	
test_PDE: ./src/Functions.o test_serial2.o  chkopts
	-${CLINKER} -o test_PDE ./src/Functions.o test_serial2.o  ${PETSC_SNES_LIB} ${SLEPC_LIB}
	${RM} test_serial2.o ./src/Functions.o
	
test_Chol: ./src/Functions.o ./src/Solver.o test_Chol.o  chkopts
	-${CLINKER} -o test_Chol ./src/Functions.o ./src/Solver.o test_Chol.o  ${PETSC_SNES_LIB} ${SLEPC_LIB}
	${RM} test_Chol.o	 ./src/Functions.o	 ./src/Solver.o	
	
Test: ./src/Functions.o Test.o  chkopts
	-${CLINKER} -o Test ./src/Functions.o Test.o  ${PETSC_SNES_LIB} ${SLEPC_LIB}
	${RM} Test.o	 ./src/Functions.o
	
TestV2: ./src/Functions.o ./src/Solver.o Test2.o  chkopts
	-${CLINKER} -o Test2 ./src/Functions.o ./src/Solver.o Test2.o  ${PETSC_SNES_LIB} ${SLEPC_LIB}
	${RM} Test2.o	 ./src/Functions.o	 ./src/Solver.o
	
TestV3: ./src/Functions.o ./src/Solver.o Test3.o  chkopts
	-${CLINKER} -o Test3 ./src/Functions.o ./src/Solver.o Test3.o  ${PETSC_SNES_LIB} ${SLEPC_LIB}
	${RM} Test3.o	 ./src/Functions.o	 ./src/Solver.o	
	
TestV4: ./src/Functions.o ./src/Solver.o Test4.o  chkopts
	-${CLINKER} -o Test4 ./src/Functions.o ./src/Solver.o Test4.o  ${PETSC_SNES_LIB} ${SLEPC_LIB}
	${RM} Test4.o	 ./src/Functions.o	 ./src/Solver.o	
	
TestV5: ./src/Functions.o ./src/Solver.o Test5.o  chkopts
	-${CLINKER} -o Test5 ./src/Functions.o ./src/Solver.o Test5.o  ${PETSC_SNES_LIB} ${SLEPC_LIB}
	${RM} Test5.o	 ./src/Functions.o	 ./src/Solver.o
	
TestV6: ./src/Functions.o ./src/Solver.o Test6.o  chkopts
	-${CLINKER} -o Test6 ./src/Functions.o ./src/Solver.o Test6.o  ${PETSC_SNES_LIB} ${SLEPC_LIB}
	${RM} Test6.o	 ./src/Functions.o	 ./src/Solver.o	
	
#mpiexec -np 4 ./test_PDE -mat_type mpiaij -vec_type mpi -ksp_monitor_short -pc_type gamg -ksp_type fgmres -ksp_gmres_modifiedgramschmidt -m 100 -n 100

#mpiexec -np 3 ./test_MC -mat_type mpiaij -vec_type mpi -Nsamples 100000 -m 300 -n 300 -pc_type ksp -ksp_type fgmres -dim 2 -alpha 2 -lamb 0.1 -sigma 0.3 -TOL 1e-3

#mpiexec -np 3 ./Test -mat_type mpiaij -vec_type mpi -Nsamples 100000 -m 50 -n 50 -pc_type ksp -ksp_type fgmres -dim 2 -alpha 2 -lamb 0.1 -sigma 0.3 -TOL 1e-2 -print_gmrf_mean -print_gmrf_var -print_sol_mean -print_sol_var -print_rho_mean -print_rho_var

#mpiexec -np 3 ./Test2 -mat_type mpiaij -vec_type mpi -Nsamples 100000 -m 50 -n 50 -pc_type ksp -ksp_type fgmres -dim 2 -alpha 2 -lamb 0.1 -sigma 0.3 -TOL 1e-2 -log_summary  

#mpiexec -np 3 ./Test5 -mat_type mpiaij -vec_type mpi -Nsamples 100000 -m 50 -n 50 -pc_type hypre -ksp_type fgmres -dim 2 -alpha 2 -lamb 0.1 -sigma 0.3 -TOL 1e-6

#mpirun -np 16 ./test_Chol -mat_type seqaij -vec_type seq -Nsamples 100000 -m 50 -n 50 -pc_type hypre -ksp_type fgmres -dim 2 -alpha 2 -lamb 0.1 -sigma 0.3 -TOL 1e-6
