
CFLAGS	         = -std=c++11 -O3 -I./src
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR           = ./ #src/ksp/ksp/examples/tutorials/
EXAMPLESC        = ex29.c Test.c
MANSEC           = KSP
CLEANFILES       = rhs.vtk solution.vtk
NP               = 1

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex29: ex29.o  chkopts
	-${CLINKER} -o ex29 ex29.o  ${PETSC_SNES_LIB}
	${RM} ex29.o
	
Test: Test.o  chkopts
	-${CLINKER} -o Test Test.o  ${PETSC_SNES_LIB}
	${RM} Test.o	
	
#mpiexec -np 4 ./Test -mat_type mpiaij -vec_type mpi -ksp_monitor_short -pc_type gamg -ksp_type fgmres -ksp_gmres_modifiedgramschmidt -m 100 -n 100 -random_exact_sol

#mpiexec -np 4 ./Test -mat_type mpiaij -vec_type mpi -pc_type gamg -ksp_type fgmres -m 100 -n 200 -print_exact_sol
