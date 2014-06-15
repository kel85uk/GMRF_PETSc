
CFLAGS	         = -std=c++11 -O -Wall -Wconversion -Wshadow -I./src/ -D VEC_OUTPUT
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR           = ./
EXAMPLESC        = ex29.c Test.cc test_serial.cc test_serial2.cc Functions.cc
MANSEC           = KSP
CLEANFILES       = rhs.vtk solution.vtk
NP               = 1
ALL: test_MC test_PDE Test TestV2
include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex29: ex29.o  chkopts
	-${CLINKER} -o ex29 ex29.o  ${PETSC_SNES_LIB}
	${RM} ex29.o
	
test_MC: ./src/Functions.o test_serial.o  chkopts
	-${CLINKER} -o test_MC ./src/Functions.o test_serial.o  ${PETSC_SNES_LIB}
	${RM} test_serial.o ./src/Functions.o
	
test_PDE: ./src/Functions.o test_serial2.o  chkopts
	-${CLINKER} -o test_PDE ./src/Functions.o test_serial2.o  ${PETSC_SNES_LIB}
	${RM} test_serial2.o ./src/Functions.o	
	
Test: ./src/Functions.o Test.o  chkopts
	-${CLINKER} -o Test ./src/Functions.o Test.o  ${PETSC_SNES_LIB}
	${RM} Test.o	 ./src/Functions.o
	
TestV2: ./src/Functions.o ./src/Solver.o Test2.o  chkopts
	-${CLINKER} -o Test2 ./src/Functions.o ./src/Solver.o Test2.o  ${PETSC_SNES_LIB}
	${RM} Test2.o	 ./src/Functions.o	 ./src/Solver.o
	
#mpiexec -np 4 ./test_PDE -mat_type mpiaij -vec_type mpi -ksp_monitor_short -pc_type gamg -ksp_type fgmres -ksp_gmres_modifiedgramschmidt -m 100 -n 100

#mpiexec -np 3 ./test_MC -mat_type mpiaij -vec_type mpi -Nsamples 100000 -m 300 -n 300 -pc_type ksp -ksp_type fgmres -dim 2 -alpha 2 -lamb 0.1 -sigma 0.3 -TOL 1e-3

#mpiexec -np 3 ./Test -mat_type mpiaij -vec_type mpi -Nsamples 100000 -m 50 -n 50 -pc_type ksp -ksp_type fgmres -dim 2 -alpha 2 -lamb 0.1 -sigma 0.3 -TOL 1e-2 -print_gmrf_mean -print_gmrf_var -print_sol_mean -print_sol_var -print_rho_mean -print_rho_var

#mpiexec -np 3 ./Test2 -mat_type mpiaij -vec_type mpi -Nsamples 100000 -m 50 -n 50 -pc_type ksp -ksp_type fgmres -dim 2 -alpha 2 -lamb 0.1 -sigma 0.3 -TOL 1e-2 -log_summary   
