/* Test Program for entire GMRF + SLMC (Reorganized into a UnitSolve function) with Master-Slave */

static char help[] = "Solves a linear system in parallel with KSP.\n\
Input parameters include:\n\
  -random_exact_sol : use a random exact solution vector\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_n>       : number of mesh points in y-direction\n\n";

#include <Functions.hh>
#include <Solver.hh>
#include <climits>
#define DEBUG 0
#define MPI_WTIME_IS_GLOBAL 1
#define WORKTAG 1
#define DIETAG 2

int main(int argc,char **argv)
{
	Vec U, EUNm1, EUN, VUN, b, M2N, resU, rho, ErhoNm1, ErhoN, VrhoN, N01, M2Nr, resR, gmrf, EgmrfN, EgmrfNm1, VgmrfN, M2Ng;
	Vec* Wrapalla[12] = {&rho, &ErhoNm1, &ErhoN, &VrhoN, &N01, &M2Nr, &resR, &gmrf, &EgmrfN, &EgmrfNm1, &VgmrfN, &M2Ng};
	Vec* Wrapallb[7] = {&U, &EUNm1, &EUN, &VUN, &b, &M2N, &resU};
	Mat A, L;
	PetscScalar* timings = new PetscScalar[6];
	PetscScalar* buffer_timings = new PetscScalar[6];
	for (int i = 0; i < 6; ++i) timings[i] = 0;
	for (int i = 0; i < 6; ++i) buffer_timings[i] = 0;
	PetscScalar temp_time;
	KSP kspSPDE, kspGMRF;
	PetscInt		Ns = 0, bufferInt;
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
  PetscLogStage stage;
	MPI_Comm petsc_comm_slaves;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&grank); // Get the processor global rank
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	/* Split the communicator into PETSC_COMM_WORLD */
	int ncolors = (numprocs-1)/procpercolor;
	int color = (grank-1)/procpercolor;
	if(grank == 0) color = std::numeric_limits<PetscInt>::max();
	if ((color+1) > ncolors)
		color--;
	#if DEBUG
		printf("I am processor %d with color %d \n",grank,color);
	#endif
	MPI_Comm_split(MPI_COMM_WORLD, color, grank, &petsc_comm_slaves);
	
	PETSC_COMM_WORLD = petsc_comm_slaves;
	PetscInitialize(&argc,&argv,(char*)0,help);

	MPI_Comm_rank(PETSC_COMM_WORLD,&lrank); // Get the processor local rank
	
	/* Split the different communicators between root and workers */
	startTime = MPI_Wtime();
	srand(grank);
	std::default_random_engine generator(rand());
	ierr = PetscLogStageRegister("Get options and create all vectors", &stage);CHKERRQ(ierr);
	ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
	ierr = GetOptions(users);CHKERRQ(ierr);
	/* Create all the vectors and matrices needed for calculation */
	ierr = CreateVectors(*Wrapalla,12,users.NT);CHKERRQ(ierr);
	ierr = CreateVectors(*Wrapallb,7,users.NI);CHKERRQ(ierr);
	/* Create Matrices and Solver contexts */
	ierr = CreateSolvers(L,users.NT,kspGMRF,A,users.NI,kspSPDE);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
	PetscScalar normU = 0.,EnormUN = 0.,VnormUN = 0.,EnormUNm1 = 0.,M2NnU = 0.,tol = 1.0;
  
  ierr = PetscLogStageRegister("Set GMRF operator", &stage);CHKERRQ(ierr);
	ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
	temp_time = MPI_Wtime();
	ierr = SetGMRFOperator(L,users.m,users.n,users.NGhost,users.dx,users.dy,users.kappa);CHKERRQ(ierr);
	ierr = KSPSetOperators(kspGMRF,L,L,SAME_PRECONDITIONER);CHKERRQ(ierr);
	temp_time = MPI_Wtime() - temp_time;
	timings[1] += temp_time;
  ierr = PetscLogStagePop();CHKERRQ(ierr);
	// Managers report duty to the root processor
	if(lrank == 0) MPI_Isend(&grank,1,MPI_INT,0,lrank,MPI_COMM_WORLD,&request);
	// Start the ball rolling
	if(grank == 0){
		PetscInt Nmanagers = 0;std::vector<PetscMPIInt> masters;
		PetscScalar root_times[2] = {0.,0.};
		PetscInt received_answers = 1, who, whomax;
		// Receive all the managers and place in a list
		temp_time = MPI_Wtime();
		while(Nmanagers <= ncolors){
			MPI_Recv(&bufferRank,1,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
			masters.push_back(bufferRank);
			++Nmanagers;
		}
		temp_time = MPI_Wtime() - temp_time;
	  root_times[0] += temp_time;
		std::sort(masters.begin(),masters.end(),[](PetscMPIInt a,PetscMPIInt b){return (a<b);});
		std::for_each(masters.begin(),masters.end(),[](PetscMPIInt s){std::cout << s << std::endl;});
		PetscInt total_work = users.Nsamples + 1;
		PetscScalar bufferNormU;
		whomax = ncolors;
		if(whomax > total_work)
			whomax = total_work;
		bufferBool = true;
		#if DEBUG
			PetscPrintf(MPI_COMM_WORLD,"I am processor %d in world, %d in petsc \n",grank,lrank);
		#endif
    ierr = PetscLogStageRegister("Start distributing work", &stage);CHKERRQ(ierr);
    ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
    temp_time = MPI_Wtime();
		for(who = 1; who <= whomax; ++who){
			MPI_Isend(&bufferBool,1,MPI_C_BOOL,masters[who],WORKTAG,MPI_COMM_WORLD,&request);
		}
		temp_time = MPI_Wtime() - temp_time;
	  root_times[0] += temp_time;
		while ((received_answers <= total_work) && (tol > users.TOL)){
		  temp_time = MPI_Wtime();
			MPI_Recv(&bufferNormU,1,MPI_DOUBLE,MPI_ANY_SOURCE,WORKTAG,MPI_COMM_WORLD,&status);
			who = status.MPI_SOURCE;
			#if 1
				PetscPrintf(MPI_COMM_WORLD,"Proc[%d]: Received norm from processor %d \n",grank,who);
			#endif
			temp_time = MPI_Wtime() - temp_time;
	    root_times[0] += temp_time;
			normU = bufferNormU;
			temp_time = MPI_Wtime();
			update_stats(EnormUN,VnormUN,EnormUNm1,M2NnU,tol,normU,received_answers);
			temp_time = MPI_Wtime() - temp_time;
	    root_times[1] += temp_time;
			++received_answers;
			temp_time = MPI_Wtime();
			if (Ns < users.Nsamples || tol > users.TOL){
				MPI_Isend(&bufferBool,1,MPI_C_BOOL,who,WORKTAG,MPI_COMM_WORLD,&request);
				++Ns;
			}
      temp_time = MPI_Wtime() - temp_time;
      root_times[0] += temp_time;
		}
		temp_time = MPI_Wtime();
		for (who = 1; who <= whomax; ++who){
			bufferBool = false;
			MPI_Isend(&bufferBool,1,MPI_C_BOOL,masters[who],DIETAG,MPI_COMM_WORLD,&request);
			PetscPrintf(MPI_COMM_WORLD,"Proc[%d]: Sending kill signal to proc %d\n",grank,masters[who]);
		}
		MPI_Reduce(timings,buffer_timings,6,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
		temp_time = MPI_Wtime() - temp_time;
	  root_times[0] += temp_time;
		ierr = PetscLogStagePop();CHKERRQ(ierr);
		PetscPrintf(MPI_COMM_WORLD,"Expectation of ||U|| = %4.8E\n",EnormUN);
		PetscPrintf(MPI_COMM_WORLD,"Communication GMRF (s) = %4.8E\n",buffer_timings[0]);
		PetscPrintf(MPI_COMM_WORLD,"Setup GMRF (s) = %4.8E\n",buffer_timings[1]);
		PetscPrintf(MPI_COMM_WORLD,"Solving GMRF (s) = %4.8E\n",buffer_timings[2]);
		PetscPrintf(MPI_COMM_WORLD,"Communication SPDE (s) = %4.8E\n",buffer_timings[3]);
		PetscPrintf(MPI_COMM_WORLD,"Setup SPDE (s) = %4.8E\n",buffer_timings[4]);
		PetscPrintf(MPI_COMM_WORLD,"Solving SPDE (s) = %4.8E\n",buffer_timings[5]);
		PetscPrintf(MPI_COMM_WORLD,"Communication root (s) = %4.8E\n",root_times[0]);
		PetscPrintf(MPI_COMM_WORLD,"Computation root (s) = %4.8E\n",root_times[1]);
	}
	if (grank != 0){
		int work_status = DIETAG;
		ierr = PetscLogStageRegister("Receive and work on jobs", &stage);CHKERRQ(ierr);
    ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
		if(lrank == 0) {
			MPI_Recv(&bufferBool,1,MPI_C_BOOL,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
			work_status = status.MPI_TAG;
			MPI_Bcast(&work_status,1,MPI_INT,0,PETSC_COMM_WORLD);
			#if DEBUG
				PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: I am broadcasting to my subordinates\n",grank);
			#endif
		}
		else {
			MPI_Bcast(&work_status,1,MPI_INT,0,PETSC_COMM_WORLD);
			#if DEBUG
				PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: I am receiving broadcast\n",grank);
			#endif
		}
		if(work_status != DIETAG){
			while(true){
        ierr = UnitSolverTimings(rho,gmrf,N01,kspGMRF,U,b,A,kspSPDE,users,generator,lrank,Ns,bufferScalar,timings); CHKERRQ(ierr);
				++Ns;				
				if(lrank == 0){
					MPI_Send(&bufferScalar,1,MPI_DOUBLE,0,WORKTAG,MPI_COMM_WORLD);
					#if DEBUG
					PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: Waiting for work\n",grank);
					#endif
					MPI_Recv(&bufferBool,1,MPI_C_BOOL,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
					work_status = status.MPI_TAG;
					MPI_Bcast(&work_status,1,MPI_INT,0,PETSC_COMM_WORLD);
					#if DEBUG
					PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: I am broadcasting to my subordinates with tag = %d\n",grank,work_status);
					#endif
				}
				else{
					MPI_Bcast(&work_status,1,MPI_INT,0,PETSC_COMM_WORLD);
					#if DEBUG
					PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: I am receiving broadcast with tag = %d\n",grank,work_status);
					#endif
				}
				if(work_status == DIETAG){
					#if DEBUG
					PetscPrintf(PETSC_COMM_WORLD,"Proc[%d]: We finished all work \n",grank);	
					#endif
					MPI_Reduce(timings,buffer_timings,6,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
					break;
				}
			}
		  ierr = PetscLogStagePop();CHKERRQ(ierr);
		}
	}

	endTime = MPI_Wtime();
	PetscPrintf(MPI_COMM_WORLD,"Proc[%d]: All done! \n",grank);
	if (grank != 0) PetscPrintf(PETSC_COMM_WORLD,"Proc[%d]: We did %d samples \n",grank,(Ns-1));
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
	MPI_Finalize();
	return 0;
}
