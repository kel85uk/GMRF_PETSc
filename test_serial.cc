/* Test Program for Petsc */

static char help[] = "Solves a linear system in parallel with KSP.\n\
Input parameters include:\n\
  -random_exact_sol : use a random exact solution vector\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_n>       : number of mesh points in y-direction\n\n";

#include <petsc.h>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <random>
#include <cmath>
#include <Functions.hh>

#define nint(a) ((a) >= 0.0 ? (PetscInt)((a)+0.5) : (PetscInt)((a)-0.5))


int main(int argc,char **argv)
{
	Vec U, EUN, EUNm1, VUN, M2N, Z;
	Mat L;
	PetscInt Ns;
	UserCTX users;
	KSP kspGMRF;
	PetscScalar	result;
	PetscErrorCode ierr;
	PetscBool      flg = PETSC_FALSE;
	PetscMPIInt rank;	
	PetscInitialize(&argc,&argv,(char*)0,help);
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	srand(rank);
	boost::variate_generator<boost::mt19937, boost::normal_distribution<> >	generator(boost::mt19937(rand()),boost::normal_distribution<>(0.,1.));
	ierr = GetOptions(users);CHKERRQ(ierr);
	ierr = VecCreate(PETSC_COMM_WORLD,&U);CHKERRQ(ierr);
	ierr = VecSetSizes(U,PETSC_DECIDE,users.NT);CHKERRQ(ierr);
	ierr = VecSetFromOptions(U);CHKERRQ(ierr);
	ierr = VecSet(U,0.);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&EUN);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&VUN);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&EUNm1);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&M2N);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&Z);CHKERRQ(ierr);
	ierr = VecCopy(U,EUN);CHKERRQ(ierr);
	ierr = VecCopy(U,VUN);CHKERRQ(ierr);
	ierr = VecCopy(U,EUNm1);CHKERRQ(ierr);
	ierr = VecCopy(U,M2N);CHKERRQ(ierr);
	ierr = VecCopy(U,Z);CHKERRQ(ierr);
	
	ierr = MatCreate(PETSC_COMM_WORLD,&L);CHKERRQ(ierr); // Create matrix A residing in PETSC_COMM_WORLD
	ierr = MatSetSizes(L,PETSC_DECIDE,PETSC_DECIDE,users.NT,users.NT);CHKERRQ(ierr); // Set the size of the matrix A, and let PETSC decide the decomposition
	ierr = MatSetFromOptions(L);CHKERRQ(ierr);
	ierr = MatMPIAIJSetPreallocation(L,5,NULL,5,NULL);CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(L,5,NULL);CHKERRQ(ierr);
	ierr = MatSetUp(L);CHKERRQ(ierr);	
	ierr = KSPCreate(PETSC_COMM_WORLD,&kspGMRF);CHKERRQ(ierr);
	ierr = KSPSetTolerances(kspGMRF,1.e-5/(users.NT),1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
	ierr = KSPSetFromOptions(kspGMRF);CHKERRQ(ierr);	

	PetscPrintf(PETSC_COMM_WORLD,"NGhost = %d and I am Processor[%d] \n",users.NGhost,rank);
	PetscPrintf(PETSC_COMM_WORLD,"tau2 = %f \n",users.tau2);
	PetscPrintf(PETSC_COMM_WORLD,"kappa = %f \n",users.kappa);
	PetscPrintf(PETSC_COMM_WORLD,"nu = %f \n",users.nu);
	PetscPrintf(PETSC_COMM_WORLD,"dx = %f \n",users.dx);
	PetscPrintf(PETSC_COMM_WORLD,"dy = %f \n",users.dy);

	for (Ns = 1; (Ns <= users.Nsamples) && (users.tol > users.TOL); ++Ns){
		ierr = SetRandSource(Z,users.NT,users.dx,users.dy,rank,generator);CHKERRQ(ierr);
		ierr = SetGMRFOperator(L,users.m,users.n,users.NGhost,users.dx,users.dy,users.kappa);CHKERRQ(ierr);
		ierr = KSPSetOperators(kspGMRF,L,L,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);			
		ierr = KSPSolve(kspGMRF,Z,U);CHKERRQ(ierr);
		ierr = KSPGetIterationNumber(kspGMRF,&users.its);CHKERRQ(ierr);
		ierr = VecScale(U,1.0/sqrt(users.tau2));CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD,"Sample[%d]: GMRF solved in %d iterations \n",Ns,users.its);CHKERRQ(ierr);
		ierr = update_stats(EUN,VUN,EUNm1,M2N,users.tol,U,Ns);CHKERRQ(ierr);
	}
	
	ierr = VecPostProcs(VUN,"rho_mean.dat");CHKERRQ(ierr);

	ierr = VecDestroy(&U);CHKERRQ(ierr);
	ierr = VecDestroy(&M2N);CHKERRQ(ierr);
	ierr = VecDestroy(&EUN);CHKERRQ(ierr);
	ierr = VecDestroy(&EUNm1);CHKERRQ(ierr);
	ierr = VecDestroy(&VUN);CHKERRQ(ierr);
	ierr = VecDestroy(&Z);CHKERRQ(ierr);
	
	ierr = KSPDestroy(&kspGMRF);CHKERRQ(ierr);
	ierr = MatDestroy(&L);CHKERRQ(ierr);

	/*
	  Always call PetscFinalize() before exiting a program.  This routine
		 - finalizes the PETSc libraries as well as MPI
		 - provides summary and diagnostic information if certain runtime
		   options are chosen (e.g., -log_summary).
	*/
	ierr = PetscFinalize();	
	return 0;
}
