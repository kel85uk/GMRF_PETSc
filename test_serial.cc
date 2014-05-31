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
	Vec U, EUN, EUNm1, VUN, M2N, testleak;

	PetscReal		x0 = 0, x1 = 1, y0 = 0, y1 = 1, dx, dy;     /* norm of solution error */
	PetscReal		UN = 1, US = 10, UE = 5, UW = 3;
	PetscReal		dim = 2, alpha = 2, sigma = 0.3, lamb = 0.1, nu, kappa, tol = 1, TOL = 1e-10, tolE, tolV;
	PetscInt		m = 200, n = 200, its, NGhost = 2, NI = 0, NT = 0, Nsamples = 20, Ns = 1;
	PetscScalar	result;
	nu = alpha - dim/2;
	kappa = sqrt(8*nu)/(lamb);	
	dx = (x1 - x0)/m;
	dy = (y1 - y0)/n;
	NI = m*n;
	PetscErrorCode ierr;
	PetscBool      flg = PETSC_FALSE;
	boost::variate_generator<boost::mt19937, boost::normal_distribution<> >	generator(boost::mt19937(time(0)),boost::normal_distribution<>(0.,1.));
	PetscMPIInt rank;	
	PetscInitialize(&argc,&argv,(char*)0,help);
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

	ierr = VecCreate(PETSC_COMM_WORLD,&U);CHKERRQ(ierr);
	ierr = VecSetSizes(U,PETSC_DECIDE,NI);CHKERRQ(ierr);
	ierr = VecSetFromOptions(U);CHKERRQ(ierr);
	ierr = VecSet(U,0.);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&EUN);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&VUN);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&EUNm1);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&M2N);CHKERRQ(ierr);
	ierr = VecCopy(U,EUN);CHKERRQ(ierr);
	ierr = VecCopy(U,VUN);CHKERRQ(ierr);
	ierr = VecCopy(U,EUNm1);CHKERRQ(ierr);
	ierr = VecCopy(U,M2N);CHKERRQ(ierr);


	for (Ns = 1; (Ns <= Nsamples) && (tol > TOL); ++Ns){
		ierr = SetRandSource(U,NI,dx,dy,rank,generator);CHKERRQ(ierr);
//		ierr = VecSetRandom(U,randomvec1);CHKERRQ(ierr);
		ierr = update_stats(EUN,VUN,EUNm1,M2N,tol,U,Ns);CHKERRQ(ierr);
		ierr = VecCreate(PETSC_COMM_WORLD,&testleak);CHKERRQ(ierr);
		ierr = VecSetSizes(testleak,PETSC_DECIDE,NI);CHKERRQ(ierr);
		ierr = VecSetFromOptions(testleak);CHKERRQ(ierr);
		ierr = VecSet(testleak,0.);CHKERRQ(ierr);
		
	}

	ierr = VecDestroy(&U);CHKERRQ(ierr);
	ierr = VecDestroy(&M2N);CHKERRQ(ierr);
	ierr = VecDestroy(&EUN);CHKERRQ(ierr);
	ierr = VecDestroy(&EUNm1);CHKERRQ(ierr);
	ierr = VecDestroy(&VUN);CHKERRQ(ierr);
	ierr = VecDestroy(&testleak);CHKERRQ(ierr);

	/*
	  Always call PetscFinalize() before exiting a program.  This routine
		 - finalizes the PETSc libraries as well as MPI
		 - provides summary and diagnostic information if certain runtime
		   options are chosen (e.g., -log_summary).
	*/
	ierr = PetscFinalize();	
	return 0;
}
