/* Test Program for Petsc */

static char help[] = "Solves a linear system in parallel with KSP.\n\
Input parameters include:\n\
  -random_exact_sol : use a random exact solution vector\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_n>       : number of mesh points in y-direction\n\n";

#include <petsc.h>
/*#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp> */
#include <random>
#include <cmath>
#include <Functions.hh>

#define nint(a) ((a) >= 0.0 ? (PetscInt)((a)+0.5) : (PetscInt)((a)-0.5))

int main(int argc,char **argv)
{
	Vec U, Um1, EU, VU, b, res, rho, rhom1, Erho, Vrho, N01;
	Mat A, L;
	KSP kspSPDE, kspGMRF;
	PetscReal		norm, x0 = 0, x1 = 1, y0 = 0, y1 = 1, dx, dy;     /* norm of solution error */
	PetscReal		UN = 1, US = 10, UE = 5, UW = 3;
	PetscReal		dim = 2, alpha = 2, sigma = 0.3, lamb = 0.1, nu, kappa, tol = 1, TOL = 1e-5;
	PetscInt		m = 8, n = 7, its, NGhost = 2, NI = 0, NT = 0, Nsamples = 10000, Ns = 1;
	nu = alpha - dim/2;
	kappa = sqrt(8*nu)/(lamb);	
	PetscMPIInt    rank;
	dx = (x1 - x0)/m;
	dy = (y1 - y0)/n;
	PetscErrorCode ierr;
	PetscBool      flg = PETSC_FALSE;
	
	PetscInitialize(&argc,&argv,(char*)0,help);
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	ierr = GetOptions(m,n,x0,x1,y0,y1,UN,US,UE,UW,lamb);
	NGhost += PetscMax(PetscMax(nint(0.5*lamb/dx),nint(0.5*lamb/dy)),nint(0.5*lamb/(sqrt(dx*dx + dy*dy))));
	NI = m*n;
	NT = (m+2*NGhost)*(n+2*NGhost);
	PetscPrintf(PETSC_COMM_SELF,"NGhost = %d and I am Processor[%d] \n",NGhost,rank);

	ierr = VecCreate(PETSC_COMM_WORLD,&rho);CHKERRQ(ierr);
	ierr = VecSetSizes(rho,PETSC_DECIDE,NT);CHKERRQ(ierr);
	ierr = VecSetFromOptions(rho);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&rhom1);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&Erho);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&Vrho);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&N01);CHKERRQ(ierr);
	ierr = VecCreate(PETSC_COMM_WORLD,&U);CHKERRQ(ierr);
	ierr = VecSetSizes(U,PETSC_DECIDE,NI);CHKERRQ(ierr);
	ierr = VecSetFromOptions(U);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&Um1);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&EU);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&VU);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&b);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&res);CHKERRQ(ierr);		


	while (Ns < Nsamples){
		ierr = SetRandSource(N01,NT,dx,dy);CHKERRQ(ierr);
		ierr = SetGMRFOperator(L,m,n,NGhost,dx,dy,kappa);CHKERRQ(ierr);
		ierr = KSPCreate(PETSC_COMM_WORLD,&kspGMRF);CHKERRQ(ierr);
		ierr = KSPSetOperators(kspGMRF,L,L,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
		ierr = KSPSetTolerances(kspGMRF,1.e-3/(NT),1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
		ierr = KSPSetFromOptions(kspGMRF);CHKERRQ(ierr);
		ierr = KSPSolve(kspGMRF,N01,rho);CHKERRQ(ierr);
		ierr = KSPGetIterationNumber(kspGMRF,&its);CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD,"Sample[%d]: GMRF solved in %d iterations \n",Ns,its);CHKERRQ(ierr);

		ierr = SetOperator(A,rho,m,n,NGhost,dx,dy);CHKERRQ(ierr);
		ierr = SetSource(b,rho,m,n,NGhost,dx,dy,UN,US,UE,UW,lamb);CHKERRQ(ierr);
		ierr = KSPCreate(PETSC_COMM_WORLD,&kspSPDE);CHKERRQ(ierr);
		ierr = KSPSetOperators(kspSPDE,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
		ierr = KSPSetTolerances(kspSPDE,1.e-3/(NI),1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
		ierr = KSPSetFromOptions(kspSPDE);CHKERRQ(ierr);	
		ierr = KSPSolve(kspSPDE,b,U);CHKERRQ(ierr);
		ierr = KSPGetIterationNumber(kspSPDE,&its);CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD,"Sample[%d]: SPDE solved in %d iterations \n",Ns,its);CHKERRQ(ierr);

		++Ns;		
	}
	
	flg  = PETSC_FALSE;
	ierr = PetscOptionsGetBool(NULL,"-print_GMRF",&flg,NULL);CHKERRQ(ierr);
	if (flg) {ierr = PostProcs(Erho,"rho_mean.dat");CHKERRQ(ierr);}	
	
	flg  = PETSC_FALSE;
	ierr = PetscOptionsGetBool(NULL,"-print_sol",&flg,NULL);CHKERRQ(ierr);
	if (flg) {ierr = PostProcs(EU,"sol_mean.dat");CHKERRQ(ierr);}
			
	ierr = KSPDestroy(&kspSPDE);CHKERRQ(ierr);
	ierr = KSPDestroy(&kspGMRF);CHKERRQ(ierr);
	ierr = VecDestroy(&U);CHKERRQ(ierr);
	ierr = VecDestroy(&Um1);CHKERRQ(ierr);
	ierr = VecDestroy(&EU);CHKERRQ(ierr);
	ierr = VecDestroy(&VU);CHKERRQ(ierr);
	ierr = VecDestroy(&b);CHKERRQ(ierr);
	ierr = VecDestroy(&res);CHKERRQ(ierr);
	
	ierr = VecDestroy(&rho);CHKERRQ(ierr);
	ierr = VecDestroy(&rhom1);CHKERRQ(ierr);
	ierr = VecDestroy(&Erho);CHKERRQ(ierr);
	ierr = VecDestroy(&Vrho);CHKERRQ(ierr);			
	ierr = VecDestroy(&N01);CHKERRQ(ierr);

	ierr = MatDestroy(&A);CHKERRQ(ierr);
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
