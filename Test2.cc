/* Test Program for entire GMRF + SLMC (Reorganized into a UnitSolve function) */

static char help[] = "Solves a linear system in parallel with KSP.\n\
Input parameters include:\n\
  -random_exact_sol : use a random exact solution vector\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_n>       : number of mesh points in y-direction\n\n";

#include <Functions.hh>
#include <Solver.hh>
#define MPI_WTIME_IS_GLOBAL
//#define VEC_OUTPUT

int main(int argc,char **argv)
{
	Vec U, EUNm1, EUN, VUN, b, M2N, resU, rho, ErhoNm1, ErhoN, VrhoN, N01, M2Nr, resR, gmrf, EgmrfN, EgmrfNm1, VgmrfN, M2Ng;
	Vec* Wrapalla[12] = {&rho, &ErhoNm1, &ErhoN, &VrhoN, &N01, &M2Nr, &resR, &gmrf, &EgmrfN, &EgmrfNm1, &VgmrfN, &M2Ng};
	Vec* Wrapallb[7] = {&U, &EUNm1, &EUN, &VUN, &b, &M2N, &resU};
	Mat A, L;
	KSP kspSPDE, kspGMRF;
	PetscInt		Ns;
	UserCTX		users;
	PetscMPIInt    rank;
	PetscErrorCode ierr;
	PetscBool      flg = PETSC_FALSE;
	PetscReal startTime, endTime;
	
	PetscInitialize(&argc,&argv,(char*)0,help);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	
	startTime = MPI_Wtime();
	srand(rank);
	std::default_random_engine generator(rand());	
	ierr = GetOptions(users);CHKERRQ(ierr);
	#ifndef VEC_OUTPUT
		users.tol = 0.0;
	#endif
	/* Create all the vectors and matrices needed for calculation */
	ierr = CreateVectors(*Wrapalla,12,users.NT);CHKERRQ(ierr);
	ierr = CreateVectors(*Wrapallb,7,users.NI);CHKERRQ(ierr);
	/* Create Matrices and Solver contexts */
	ierr = CreateSolvers(L,users.NT,kspGMRF,A,users.NI,kspSPDE);CHKERRQ(ierr);
	
	PetscScalar normU,EnormUN,VnormUN,EnormUNm1,M2NnU,tol = 1.0;
	ierr = SetGMRFOperator(L,users.m,users.n,users.NGhost,users.dx,users.dy,users.kappa);CHKERRQ(ierr);
	ierr = KSPSetOperators(kspGMRF,L,L,SAME_PRECONDITIONER);CHKERRQ(ierr);
	
	for (Ns = 1; (Ns <= users.Nsamples) && (tol > users.TOL); ++Ns){
		ierr = UnitSolver(rho,gmrf,N01,kspGMRF,U,b,A,kspSPDE,users,generator,rank,Ns,normU); CHKERRQ(ierr);
		update_stats(EnormUN,VnormUN,EnormUNm1,M2NnU,tol,normU,Ns);
		#ifdef VEC_OUTPUT
			// Start calculations for ErhoN and VrhoN
			ierr = update_stats(ErhoN,VrhoN,ErhoNm1,M2Nr,users.tolr,rho,Ns);CHKERRQ(ierr);		
			// Start calculations for EUN and VUN
			ierr = update_stats(EUN,VUN,EUNm1,M2N,users.tolU,U,Ns);CHKERRQ(ierr);
			// Start calculations for EgmrfN and VgmrfN (Can be used to output covariance later)
			ierr = update_stats(EgmrfN,VgmrfN,EgmrfNm1,M2Ng,gmrf,Ns);CHKERRQ(ierr);		
			users.tol = PetscMax(users.tolU,users.tolr);
		#endif
		tol = PetscMax(users.tol,tol);
	}
	endTime = MPI_Wtime();
	PetscPrintf(PETSC_COMM_WORLD,"Elapsed wall-clock time (sec)= %f \n",endTime - startTime);
	PetscPrintf(PETSC_COMM_WORLD,"NGhost = %d and I am Processor[%d] \n",users.NGhost,rank);
	PetscPrintf(PETSC_COMM_WORLD,"tau2 = %f \n",users.tau2);
	PetscPrintf(PETSC_COMM_WORLD,"kappa = %f \n",users.kappa);
	PetscPrintf(PETSC_COMM_WORLD,"nu = %f \n",users.nu);	
	
	#ifdef VEC_OUTPUT
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
	#endif
		
	ierr = KSPDestroy(&kspSPDE);CHKERRQ(ierr);
	ierr = KSPDestroy(&kspGMRF);CHKERRQ(ierr);
	
	ierr = DestroyVectors(*Wrapalla,12);CHKERRQ(ierr);
	ierr = DestroyVectors(*Wrapallb,7);CHKERRQ(ierr);

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
