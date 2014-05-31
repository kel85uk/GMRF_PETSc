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
//#include <random>
#include <cmath>
#include <Functions.hh>

#define nint(a) ((a) >= 0.0 ? (PetscInt)((a)+0.5) : (PetscInt)((a)-0.5))

int main(int argc,char **argv)
{
	Vec U, EUNm1, EUN, VUN, b, M2N, resU, rho, ErhoNm1, ErhoN, VrhoN, N01, M2Nr, resR;
	Mat A, L;
	KSP kspSPDE, kspGMRF;
	PetscReal		normE, normV;
	PetscInt		Ns;
	UserCTX		users;
	PetscScalar	result;
	PetscMPIInt    rank;
	PetscErrorCode ierr;
	PetscBool      flg = PETSC_FALSE;
	boost::variate_generator<boost::mt19937, boost::normal_distribution<> >	generator(boost::mt19937(time(0)),boost::normal_distribution<>(0.,1.));
	PetscInitialize(&argc,&argv,(char*)0,help);
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	ierr = GetOptions(users);
	users.dx = (users.x1 - users.x0)/users.m;
	users.dy = (users.y1 - users.y0)/users.n;
	users.nu = users.alpha - users.dim/2.0;
	users.kappa = sqrt(8.0*users.nu)/(users.lamb);	
	users.NGhost += PetscMax(PetscMax(nint(0.5*users.lamb/users.dx),nint(0.5*users.lamb/users.dy)),nint(0.5*users.lamb/(sqrt(users.dx*users.dx + users.dy*users.dy))));
	users.NI = users.m*users.n;
	users.NT = (users.m+2*users.NGhost)*(users.n+2*users.NGhost);
	PetscPrintf(PETSC_COMM_SELF,"NGhost = %d and I am Processor[%d] \n",users.NGhost,rank);
	/* Create all the vectors and matrices needed for calculation */
	ierr = VecCreate(PETSC_COMM_WORLD,&rho);CHKERRQ(ierr);
	ierr = VecSetSizes(rho,PETSC_DECIDE,users.NT);CHKERRQ(ierr);
	ierr = VecSetFromOptions(rho);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&ErhoNm1);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&ErhoN);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&M2Nr);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&VrhoN);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&N01);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&resR);CHKERRQ(ierr);
	ierr = MatCreate(PETSC_COMM_WORLD,&L);CHKERRQ(ierr); // Create matrix A residing in PETSC_COMM_WORLD
	ierr = MatSetSizes(L,PETSC_DECIDE,PETSC_DECIDE,users.NT,users.NT);CHKERRQ(ierr); // Set the size of the matrix A, and let PETSC decide the decomposition
	ierr = MatSetFromOptions(L);CHKERRQ(ierr);
	ierr = MatMPIAIJSetPreallocation(L,5,NULL,5,NULL);CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(L,5,NULL);CHKERRQ(ierr);
	ierr = MatSetUp(L);CHKERRQ(ierr);	
	ierr = KSPCreate(PETSC_COMM_WORLD,&kspGMRF);CHKERRQ(ierr);
	ierr = KSPSetTolerances(kspGMRF,1.e-3/(users.NT),1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
	ierr = KSPSetFromOptions(kspGMRF);CHKERRQ(ierr);
	ierr = VecCreate(PETSC_COMM_WORLD,&U);CHKERRQ(ierr);
	ierr = VecSetSizes(U,PETSC_DECIDE,users.NI);CHKERRQ(ierr);
	ierr = VecSetFromOptions(U);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&EUNm1);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&EUN);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&M2N);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&VUN);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&b);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&resU);CHKERRQ(ierr);
	ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr); // Create matrix A residing in PETSC_COMM_WORLD
	ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,users.NI,users.NI);CHKERRQ(ierr); // Set the size of the matrix A, and let PETSC decide the decomposition
	ierr = MatSetFromOptions(A);CHKERRQ(ierr);
	ierr = MatMPIAIJSetPreallocation(A,5,NULL,5,NULL);CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(A,5,NULL);CHKERRQ(ierr);
	ierr = MatSetUp(A);CHKERRQ(ierr);			
	ierr = KSPCreate(PETSC_COMM_WORLD,&kspSPDE);CHKERRQ(ierr);
	ierr = KSPSetTolerances(kspSPDE,1.e-3/(users.NI),1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
	ierr = KSPSetFromOptions(kspSPDE);CHKERRQ(ierr);


	for (Ns = 1; (Ns <= users.Nsamples) && (users.tol > users.TOL); ++Ns){
		ierr = SetRandSource(N01,users.NT,users.dx,users.dy,rank,generator);CHKERRQ(ierr);
		ierr = SetGMRFOperator(L,users.m,users.n,users.NGhost,users.dx,users.dy,users.kappa);CHKERRQ(ierr);
		ierr = KSPSetOperators(kspGMRF,L,L,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);			
		ierr = KSPSolve(kspGMRF,N01,rho);CHKERRQ(ierr);
		ierr = KSPGetIterationNumber(kspGMRF,&users.its);CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD,"Sample[%d]: GMRF solved in %d iterations \n",Ns,users.its);CHKERRQ(ierr);
		ierr = VecExp(rho);CHKERRQ(ierr);
		ierr = SetOperator(A,rho,users.m,users.n,users.NGhost,users.dx,users.dy);CHKERRQ(ierr);
		ierr = SetSource(b,rho,users.m,users.n,users.NGhost,users.dx,users.dy,users.UN,users.US,users.UE,users.UW,users.lamb);CHKERRQ(ierr);
		ierr = KSPSetOperators(kspSPDE,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
		ierr = KSPSolve(kspSPDE,b,U);CHKERRQ(ierr);
		ierr = KSPGetIterationNumber(kspSPDE,&users.its);CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD,"Sample[%d]: SPDE solved in %d iterations \n",Ns,users.its);CHKERRQ(ierr);
		
		// Start calculations for EUN	and VUN
		ierr = update_stats(EUN,VUN,EUNm1,M2N,users.tolU,U,Ns);CHKERRQ(ierr);
		// Start calculations for ErhoN	and VrhoN
		ierr = update_stats(ErhoN,VrhoN,ErhoNm1,M2Nr,users.tolr,rho,Ns);CHKERRQ(ierr);
		users.tol = PetscMax(users.tolU,users.tolr);
	}

	flg  = PETSC_FALSE;
	ierr = PetscOptionsGetBool(NULL,"-print_GMRF",&flg,NULL);CHKERRQ(ierr);
	if (flg) {ierr = PostProcs(ErhoN,"rho_mean.dat");CHKERRQ(ierr);}	

	flg  = PETSC_FALSE;
	ierr = PetscOptionsGetBool(NULL,"-print_sol",&flg,NULL);CHKERRQ(ierr);
	if (flg) {ierr = PostProcs(EUN,"sol_mean.dat");CHKERRQ(ierr);}
	ierr = KSPDestroy(&kspSPDE);CHKERRQ(ierr);
	ierr = KSPDestroy(&kspGMRF);CHKERRQ(ierr);

	ierr = VecDestroy(&U);CHKERRQ(ierr);
	ierr = VecDestroy(&M2N);CHKERRQ(ierr);
	ierr = VecDestroy(&EUN);CHKERRQ(ierr);
	ierr = VecDestroy(&EUNm1);CHKERRQ(ierr);
	ierr = VecDestroy(&VUN);CHKERRQ(ierr);
	ierr = VecDestroy(&b);CHKERRQ(ierr);
	ierr = VecDestroy(&resU);CHKERRQ(ierr);
	
	ierr = VecDestroy(&rho);CHKERRQ(ierr);
	ierr = VecDestroy(&M2Nr);CHKERRQ(ierr);
	ierr = VecDestroy(&ErhoNm1);CHKERRQ(ierr);
	ierr = VecDestroy(&ErhoN);CHKERRQ(ierr);
	ierr = VecDestroy(&VrhoN);CHKERRQ(ierr);	
	ierr = VecDestroy(&N01);CHKERRQ(ierr);
	ierr = VecDestroy(&resR);CHKERRQ(ierr);

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
