/* Test Program for Petsc */

static char help[] = "Solves a linear system in parallel with KSP.\n\
Input parameters include:\n\
  -random_exact_sol : use a random exact solution vector\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_n>       : number of mesh points in y-direction\n\n";

#include <Functions.hh>

int main(int argc,char **argv)
{
	Vec U, EUNm1, EUN, VUN, b, M2N, resU, rho, ErhoNm1, ErhoN, VrhoN, N01, M2Nr, resR, gmrf, EgmrfN, EgmrfNm1, VgmrfN, M2Ng;
	Mat A, L;
	KSP kspSPDE, kspGMRF;
	PetscReal		normE, normV;
	PetscInt		Ns;
	UserCTX		users;
	PetscScalar	result;
	PetscMPIInt    rank;
	PetscErrorCode ierr;
	PetscBool      flg = PETSC_FALSE;
	
	PetscInitialize(&argc,&argv,(char*)0,help);
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	srand(rank);
	std::default_random_engine generator(rand());	
	ierr = GetOptions(users);CHKERRQ(ierr);
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
	ierr = VecDuplicate(rho,&gmrf);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&EgmrfNm1);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&EgmrfN);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&VgmrfN);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&M2Ng);CHKERRQ(ierr);	
	ierr = MatCreate(PETSC_COMM_WORLD,&L);CHKERRQ(ierr); // Create matrix A residing in PETSC_COMM_WORLD
	ierr = MatSetSizes(L,PETSC_DECIDE,PETSC_DECIDE,users.NT,users.NT);CHKERRQ(ierr); // Set the size of the matrix A, and let PETSC decide the decomposition
	ierr = MatSetFromOptions(L);CHKERRQ(ierr);
	ierr = MatMPIAIJSetPreallocation(L,5,NULL,5,NULL);CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(L,5,NULL);CHKERRQ(ierr);
	ierr = MatSetUp(L);CHKERRQ(ierr);	
	ierr = KSPCreate(PETSC_COMM_WORLD,&kspGMRF);CHKERRQ(ierr);
	ierr = KSPSetTolerances(kspGMRF,1.e-7/(users.NT),1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
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
	ierr = KSPSetTolerances(kspSPDE,1.e-7/(users.NI),1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
	ierr = KSPSetFromOptions(kspSPDE);CHKERRQ(ierr);

	ierr = SetGMRFOperator(L,users.m,users.n,users.NGhost,users.dx,users.dy,users.kappa);CHKERRQ(ierr);
	ierr = KSPSetOperators(kspGMRF,L,L,SAME_PRECONDITIONER);CHKERRQ(ierr);	

	for (Ns = 1; (Ns <= users.Nsamples) && (users.tol > users.TOL); ++Ns){
		ierr = SetRandSource(N01,users.NT,users.dx,users.dy,rank,generator);CHKERRQ(ierr);	
		ierr = KSPSolve(kspGMRF,N01,rho);CHKERRQ(ierr);
		ierr = KSPGetIterationNumber(kspGMRF,&users.its);CHKERRQ(ierr);
		ierr = VecScale(rho,1.0/sqrt(users.tau2));CHKERRQ(ierr);
		ierr = VecCopy(rho,gmrf);CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD,"Sample[%d]: GMRF solved in %d iterations \n",Ns,users.its);CHKERRQ(ierr);
		ierr = VecExp(rho);CHKERRQ(ierr);
		ierr = SetOperator(A,rho,users.m,users.n,users.NGhost,users.dx,users.dy);CHKERRQ(ierr);
		ierr = SetSource(b,rho,users.m,users.n,users.NGhost,users.dx,users.dy,users.UN,users.US,users.UE,users.UW,users.lamb);CHKERRQ(ierr);
		ierr = KSPSetOperators(kspSPDE,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
		ierr = KSPSolve(kspSPDE,b,U);CHKERRQ(ierr);
		ierr = KSPGetIterationNumber(kspSPDE,&users.its);CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD,"Sample[%d]: SPDE solved in %d iterations \n",Ns,users.its);CHKERRQ(ierr);
		// Start calculations for ErhoN and VrhoN
		ierr = update_stats(ErhoN,VrhoN,ErhoNm1,M2Nr,users.tolr,rho,Ns);CHKERRQ(ierr);		
		// Start calculations for EUN and VUN
		ierr = update_stats(EUN,VUN,EUNm1,M2N,users.tolU,U,Ns);CHKERRQ(ierr);
		// Start calculations for EgmrfN and VgmrfN (Can be used to output covariance later)
		ierr = update_stats(EgmrfN,VgmrfN,EgmrfNm1,M2Ng,gmrf,Ns);CHKERRQ(ierr);		
		users.tol = PetscMax(users.tolU,users.tolr);
	}
	PetscPrintf(PETSC_COMM_WORLD,"NGhost = %d and I am Processor[%d] \n",users.NGhost,rank);
	PetscPrintf(PETSC_COMM_WORLD,"tau2 = %f \n",users.tau2);
	PetscPrintf(PETSC_COMM_WORLD,"kappa = %f \n",users.kappa);
	PetscPrintf(PETSC_COMM_WORLD,"nu = %f \n",users.nu);	
	
	flg  = PETSC_FALSE;
	ierr = PetscOptionsGetBool(NULL,"-print_rho_mean",&flg,NULL);CHKERRQ(ierr);
	if (flg) {ierr = VecPostProcs(ErhoN,"rho_mean.dat",rank);CHKERRQ(ierr);}	
	
	flg  = PETSC_FALSE;
	ierr = PetscOptionsGetBool(NULL,"-print_rho_var",&flg,NULL);CHKERRQ(ierr);
	if (flg) {ierr = VecPostProcs(VrhoN,"rho_var.dat",rank);CHKERRQ(ierr);}

	flg  = PETSC_FALSE;
	ierr = PetscOptionsGetBool(NULL,"-print_sol_mean",&flg,NULL);CHKERRQ(ierr);
	if (flg) {ierr = VecPostProcs(EUN,"sol_mean.dat",rank);CHKERRQ(ierr);}
	
	flg  = PETSC_FALSE;
	ierr = PetscOptionsGetBool(NULL,"-print_sol_var",&flg,NULL);CHKERRQ(ierr);
	if (flg) {ierr = VecPostProcs(VUN,"sol_var.dat",rank);CHKERRQ(ierr);}
	
	flg  = PETSC_FALSE;
	ierr = PetscOptionsGetBool(NULL,"-print_gmrf_mean",&flg,NULL);CHKERRQ(ierr);
	if (flg) {ierr = VecPostProcs(EgmrfN,"gmrf_mean.dat",rank);CHKERRQ(ierr);}
	
	flg  = PETSC_FALSE;
	ierr = PetscOptionsGetBool(NULL,"-print_gmrf_var",&flg,NULL);CHKERRQ(ierr);
	if (flg) {ierr = VecPostProcs(VgmrfN,"gmrf_var.dat",rank);CHKERRQ(ierr);}		
		
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
	
	ierr = VecDestroy(&gmrf);CHKERRQ(ierr);
	ierr = VecDestroy(&M2Ng);CHKERRQ(ierr);
	ierr = VecDestroy(&EgmrfNm1);CHKERRQ(ierr);
	ierr = VecDestroy(&EgmrfN);CHKERRQ(ierr);
	ierr = VecDestroy(&VgmrfN);CHKERRQ(ierr);	

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
