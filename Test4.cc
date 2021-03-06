/* Test Program for entire GMRF + SLMC (Reorganized into a UnitSolve function) with Master-Slave */

static char help[] = "Solves a linear system in parallel with KSP.\n\
Input parameters include:\n\
  -random_exact_sol : use a random exact solution vector\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_n>       : number of mesh points in y-direction\n\n";

#include <Functions.hh>
#include <Solver.hh>
#define MPI_WTIME_IS_GLOBAL 1
#define WORKTAG 1
#define DIETAG 2

int main(int argc,char **argv)
{
	Vec U, EUNm1, EUN, VUN, b, M2N, resU, rho, ErhoNm1, ErhoN, VrhoN, N01, M2Nr, resR, gmrf, EgmrfN, EgmrfNm1, VgmrfN, M2Ng;
	Vec* Wrapalla[12] = {&rho, &ErhoNm1, &ErhoN, &VrhoN, &N01, &M2Nr, &resR, &gmrf, &EgmrfN, &EgmrfNm1, &VgmrfN, &M2Ng};
	Vec* Wrapallb[7] = {&U, &EUNm1, &EUN, &VUN, &b, &M2N, &resU};
	Mat A, L;
	KSP kspSPDE, kspGMRF;
	PetscInt		Ns = 1, bufferInt;
	UserCTX		users;
	PetscMPIInt    grank, lrank, bufferRank;
	PetscErrorCode ierr;
	PetscBool      flg = PETSC_FALSE;
	bool bufferBool = false;
	PetscScalar startTime, endTime, bufferScalar;
	int numprocs;
	int procpercolor = 2;
	MPI_Status status;MPI_Request request;
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

	PetscScalar normU = 0.,EnormUN = 0.,VnormUN = 0.,EnormUNm1 = 0.,M2NnU = 0.,tol = 1.0;

	ierr = SetGMRFOperator(L,users.m,users.n,users.NGhost,users.dx,users.dy,users.kappa);CHKERRQ(ierr);
	ierr = KSPSetOperators(kspGMRF,L,L,SAME_PRECONDITIONER);CHKERRQ(ierr);

	// Managers report duty to the root processor
	if(lrank == 0) MPI_Isend(&grank,1,MPI_INT,0,lrank,MPI_COMM_WORLD,&request);
	// Start the ball rolling
	if(grank == 0){
		PetscInt Nmanagers = 0;std::vector<PetscMPIInt> masters;
		PetscInt received_answers = 1, who, whomax;
		// Receive all the managers and place in a list
		while(Nmanagers < ncolors){
			MPI_Recv(&bufferRank,1,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
			masters.push_back(bufferRank);
			++Nmanagers;
		}
		std::sort(masters.begin(),masters.end(),[](PetscMPIInt a,PetscMPIInt b){return (a<b);});
		std::for_each(masters.begin(),masters.end(),[](PetscMPIInt s){std::cout << s << std::endl;});
		PetscInt total_work = users.Nsamples + 1;
		PetscScalar bufferNormU;
		whomax = ncolors - 1;
		if(whomax > total_work)
			whomax = total_work;
		bufferBool = true;
		PetscPrintf(MPI_COMM_WORLD,"I am processor %d in world, %d in petsc \n",grank,lrank);
		for(who = 1; who <= whomax; ++who){
			MPI_Isend(&bufferBool,1,MPI_C_BOOL,masters[who],WORKTAG,MPI_COMM_WORLD,&request);
		}
		while ((received_answers <= total_work) && (tol > users.TOL)){
			MPI_Recv(&bufferNormU,1,MPI_DOUBLE,MPI_ANY_SOURCE,WORKTAG,MPI_COMM_WORLD,&status);
			who = status.MPI_SOURCE;
			PetscPrintf(MPI_COMM_WORLD,"Proc[%d]: Received norm from processor %d \n",grank,who);
			normU = bufferNormU;
			update_stats(EnormUN,VnormUN,EnormUNm1,M2NnU,tol,normU,received_answers);
			++received_answers;
			if (Ns < users.Nsamples || tol > users.TOL){
				MPI_Isend(&bufferBool,1,MPI_C_BOOL,who,WORKTAG,MPI_COMM_WORLD,&request);
				++Ns;
			}
		}
		for (who = 1; who <= whomax; ++who){
			bufferBool = false;
			MPI_Isend(&bufferBool,1,MPI_C_BOOL,masters[who],DIETAG,MPI_COMM_WORLD,&request);
			PetscPrintf(MPI_COMM_WORLD,"Proc[%d]: Sending kill signal to proc %d\n",grank,masters[who]);
		}
		PetscPrintf(MPI_COMM_WORLD,"Expectation of ||U|| = %4.8E\n",EnormUN);
	}
	if (grank != 0 || lrank == 0){
		int work_status = DIETAG;
		if(lrank == 0) {
			MPI_Recv(&bufferBool,1,MPI_C_BOOL,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
			work_status = status.MPI_TAG;
			MPI_Bcast(&work_status,1,MPI_INT,0,PETSC_COMM_WORLD);
			PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: I am broadcasting to my subordinates\n",grank);
		}
		else {
			MPI_Bcast(&work_status,1,MPI_INT,0,PETSC_COMM_WORLD);
			PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: I am receiving broadcast\n",grank);
		}
		if(work_status != DIETAG){
			while(true){		
				ierr = UnitSolver(rho,gmrf,N01,kspGMRF,U,b,A,kspSPDE,users,generator,lrank,Ns,bufferScalar); CHKERRQ(ierr);
				if(lrank == 0){
					MPI_Send(&bufferScalar,1,MPI_DOUBLE,0,WORKTAG,MPI_COMM_WORLD);
					MPI_Recv(&bufferBool,1,MPI_C_BOOL,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
					work_status = status.MPI_TAG;
					MPI_Bcast(&work_status,1,MPI_INT,0,PETSC_COMM_WORLD);
					PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: I am broadcasting to my subordinates with tag = %d\n",grank,work_status);
				}
				else{
					MPI_Bcast(&work_status,1,MPI_INT,0,PETSC_COMM_WORLD);
					PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: I am receiving broadcast with tag = %d\n",grank,work_status);
				}
				if(work_status == DIETAG){
					PetscPrintf(PETSC_COMM_WORLD,"Proc[%d]: We finished all work \n",grank);	
					break;
				}
			}
		}
	}

	endTime = MPI_Wtime();
	PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: All done! \n",grank);
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
