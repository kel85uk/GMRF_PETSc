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

#define WORKTAG 1
#define DIETAG 2

void Masterprocess(int&);
void Slaveprocess(Vec&,Vec&,Vec&,KSP&,Vec&,Vec&,Mat&,Vec&,UserCTX&,std::default_random_engine&,const PetscMPIInt&, const PetscInt&, PetscScalar&);

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

	MPI_Group worldgroup, slavegroup;
	MPI_Comm petsc_comm_slaves;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&grank); // Get the processor global rank
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_group(MPI_COMM_WORLD,&worldgroup);
	
	MPI_Group_excl(worldgroup,1,ranks,&slavegroup); // Exclude the root process
	MPI_Comm_create(MPI_COMM_WORLD,slavegroup,&petsc_comm_slaves);
	PETSC_COMM_WORLD = petsc_comm_slaves; // PETSC_COMM_WORLD does not have root process now
	/* To further split the PETSC communicator */
	int ncolors = (numprocs-1)/procpercolor;
	int color = grank/procpercolor;
	if ((color+1) > ncolors)
     color--;
	MPI_Comm_split(PETSC_COMM_WORLD, color, grank, &petsc_comm_slaves);
	
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
	
	PetscScalar normU,EnormUN,VnormUN,EnormUNm1,M2NnU,tol = 1.0;

	ierr = SetGMRFOperator(L,users.m,users.n,users.NGhost,users.dx,users.dy,users.kappa);CHKERRQ(ierr);
	ierr = KSPSetOperators(kspGMRF,L,L,SAME_PRECONDITIONER);CHKERRQ(ierr);
	if(grank == 0){
		Masterprocess();
	}
	else{
		Slaveprocess(rho,gmrf,N01,kspGMRF,U,b,A,kspSPDE,users,generator,lrank,Ns,normU);
	}
	endTime = MPI_Wtime();
	PetscPrintf(MPI_COMM_WORLD,"Elapsed wall-clock time (sec)= %f \n",endTime - startTime);
	PetscPrintf(MPI_COMM_WORLD,"NGhost = %d and I am Processor[%d] \n",users.NGhost,lrank);
	PetscPrintf(MPI_COMM_WORLD,"tau2 = %f \n",users.tau2);
	PetscPrintf(MPI_COMM_WORLD,"kappa = %f \n",users.kappa);
	PetscPrintf(MPI_COMM_WORLD,"nu = %f \n",users.nu);	
		
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

void Masterprocess(int& numprocs){
	MPI_Request request;
	int ii = 0;
	// Send task to slaves
	int whomax = numprocs - 1;
	if (whomax > users.Nsamples)
		whomax = users.Nsamples;
	for (int who = 1; who <= whomax; ++who, ++ii){
		MPI_Send(&ii,1,MPI_INT,who,WORKTAG,MPI_COMM_WORLD);
	}
	for (Ns = 1; (Ns <= users.Nsamples) && (tol > users.TOL); ++Ns){
		
		update_stats(EnormUN,VnormUN,EnormUNm1,M2NnU,tol,normU,Ns);
	}
}

void Slaveprocess(Vec& rho, Vec& gmrf, Vec& N01, KSP& kspGMRF, Vec& U, Vec& b, Mat& A, Vec& kspSPDE, UserCTX& users, std::default_random_engine& generator,const PetscMPIInt& lrank, const PetscInt& Ns, PetscScalar& normU){
	ierr = UnitSolver(rho,gmrf,N01,kspGMRF,U,b,A,kspSPDE,users,generator,lrank,Ns,normU); CHKERRQ(ierr);
}
