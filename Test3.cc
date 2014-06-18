/* Test Program for entire GMRF + SLMC (Reorganized into a UnitSolve function) with Master-Slave */

static char help[] = "Solves a linear system in parallel with KSP.\n\
Input parameters include:\n\
  -random_exact_sol : use a random exact solution vector\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_n>       : number of mesh points in y-direction\n\n";

#include <Functions.hh>
#include <Solver.hh>
#define MPI_WTIME_IS_GLOBAL

int main(int argc,char **argv)
{
	Vec U, EUNm1, EUN, VUN, b, M2N, resU, rho, ErhoNm1, ErhoN, VrhoN, N01, M2Nr, resR, gmrf, EgmrfN, EgmrfNm1, VgmrfN, M2Ng;
	Vec* Wrapalla[12] = {&rho, &ErhoNm1, &ErhoN, &VrhoN, &N01, &M2Nr, &resR, &gmrf, &EgmrfN, &EgmrfNm1, &VgmrfN, &M2Ng};
	Vec* Wrapallb[7] = {&U, &EUNm1, &EUN, &VUN, &b, &M2N, &resU};
	Mat A, L;
	KSP kspSPDE, kspGMRF;
	PetscInt		Ns;
	UserCTX		users;
	PetscMPIInt    grank, lrank;
	PetscErrorCode ierr;
	PetscBool      flg = PETSC_FALSE;
	PetscReal startTime, endTime;
	int numprocs;
	int procpercolor = 2;
	MPI_Status status;
	int ranks[] = {0};

	MPI_Comm petsc_comm_slaves;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&grank); // Get the processor global rank
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	/* Split the communicator into PETSC_COMM_WORLD */
	int ncolors = (numprocs)/procpercolor;
	int color = grank/procpercolor;
	if ((color+1) > ncolors)
     color--;
	MPI_Comm_split(MPI_COMM_WORLD, color, grank, &petsc_comm_slaves);
	
	PETSC_COMM_WORLD = petsc_comm_slaves;

	PetscInitialize(&argc,&argv,(char*)0,help);
	
	MPI_Comm_rank(PETSC_COMM_WORLD,&lrank); // Get the processor local rank

	PetscPrintf(PETSC_COMM_WORLD,"I am processor %d \n",lrank);	
	/* Split the different communicators between root and workers */
	startTime = MPI_Wtime();
	srand(grank);
	std::default_random_engine generator(rand());	
	ierr = GetOptions(users);CHKERRQ(ierr);
	/* Create all the vectors and matrices needed for calculation */
	ierr = CreateVectors(*Wrapalla,12,users.NT);CHKERRQ(ierr);
	ierr = CreateVectors(*Wrapallb,7,users.NI);CHKERRQ(ierr);
	/* Create Matrices and Solver contexts */
	ierr = CreateSolvers(L,users.NT,kspGMRF,A,users.NI,kspSPDE);CHKERRQ(ierr);
	
	PetscScalar normU = 0.,EnormUN,VnormUN,EnormUNm1,M2NnU,tol = 1.0;
	PetscInt Nspc = users.Nsamples/ncolors + users.Nsamples%ncolors;

	ierr = SetGMRFOperator(L,users.m,users.n,users.NGhost,users.dx,users.dy,users.kappa);CHKERRQ(ierr);
	ierr = KSPSetOperators(kspGMRF,L,L,SAME_PRECONDITIONER);CHKERRQ(ierr);
	for (Ns = 1; (Ns <= Nspc) && (tol > users.TOL); ++Ns){
		ierr = UnitSolver(rho,gmrf,N01,kspGMRF,U,b,A,kspSPDE,users,generator,lrank,Ns,normU); CHKERRQ(ierr);
		update_stats(EnormUN,VnormUN,EnormUNm1,M2NnU,tol,normU,Ns);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	--Ns;
	if (lrank != 0){
		EnormUN = 0.;
		Ns = 0.;
	}
	PetscPrintf(PETSC_COMM_WORLD,"I did %d samples \n",Ns);
	PetscInt sendint = Ns; PetscScalar sendreal = EnormUN*Ns;
	MPI_Reduce(&sendreal,&normU,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&sendint,&Ns,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	PetscPrintf(MPI_COMM_WORLD,"Total colors = %d \n",ncolors);
	PetscPrintf(MPI_COMM_WORLD,"Total samples = %d \n",Ns);
	PetscPrintf(MPI_COMM_WORLD,"Expectation of ||U|| = %4.7E \n",normU/Ns);
	endTime = MPI_Wtime();
	PetscPrintf(MPI_COMM_WORLD,"Elapsed wall-clock time (sec)= %f \n",endTime - startTime);
	PetscPrintf(MPI_COMM_WORLD,"NGhost = %d and I am Processor[%d] \n",users.NGhost,lrank);
	PetscPrintf(MPI_COMM_WORLD,"tau2 = %f \n",users.tau2);
	PetscPrintf(MPI_COMM_WORLD,"kappa = %f \n",users.kappa);
	PetscPrintf(MPI_COMM_WORLD,"nu = %f \n",users.nu);
	
	MPI_Barrier(MPI_COMM_WORLD);
		
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
	MPI_Finalize();
	return 0;
}
