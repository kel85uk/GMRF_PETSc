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
#include <mpe.h>
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
	MPE_Init_log();
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
	
	//PETSC_COMM_WORLD = petsc_comm_slaves;
	PetscInitialize(&argc,&argv,(char*)0,help);
	PetscLogBegin();
	MPI_Comm_rank(petsc_comm_slaves,&lrank); // Get the processor local rank
	PetscLogEvent COMMS, COMPS;
	PetscClassId communications, computations;
	PetscClassIdRegister("All Comms",&communications);
  PetscLogEventRegister("Communications",communications,&COMMS);
  PetscClassIdRegister("All Comps",&computations);
  PetscLogEventRegister("Computations",computations,&COMPS);
  std::vector<PetscLogEvent> events;
  events.push_back(COMMS);
  events.push_back(COMPS);
  
  int start_comp = MPE_Log_get_event_number();
  int end_comp = MPE_Log_get_event_number();
  int start_comm = MPE_Log_get_event_number();
  int end_comm = MPE_Log_get_event_number();
  MPE_Describe_state(start_comp,end_comp,"Comp","green:gray");
  MPE_Describe_state(start_comm,end_comm,"Comm","red:white");
	/* Split the different communicators between root and workers */
	startTime = MPI_Wtime();
	srand(grank);
	std::default_random_engine generator(rand());
	ierr = GetOptions(users);CHKERRQ(ierr);
	/* Create all the vectors and matrices needed for calculation */
	PetscLogEventBegin(events[1],0,0,0,0);
	ierr = CreateVectors(*Wrapalla,12,users.NT,petsc_comm_slaves);CHKERRQ(ierr);
	ierr = CreateVectors(*Wrapallb,7,users.NI,petsc_comm_slaves);CHKERRQ(ierr);
	/* Create Matrices and Solver contexts */
	ierr = CreateSolvers(L,users.NT,kspGMRF,A,users.NI,kspSPDE,petsc_comm_slaves);CHKERRQ(ierr);
  	PetscScalar normU = 0.,EnormUN = 0.,VnormUN = 0.,EnormUNm1 = 0.,M2NnU = 0.,tol = 1.0;
  PetscLogEventEnd(events[1],0,0,0,0);
	ierr = SetGMRFOperatorT(L,users.m,users.n,users.NGhost,users.dx,users.dy,users.kappa,events);CHKERRQ(ierr);
	PetscLogEventBegin(events[1],0,0,0,0);
	ierr = KSPSetOperators(kspGMRF,L,L,SAME_PRECONDITIONER);CHKERRQ(ierr);
	PetscLogEventEnd(events[1],0,0,0,0);
	// Managers report duty to the root processor
	if(lrank == 0) {
	  PetscLogEventBegin(events[0],0,0,0,0);
	  MPI_Isend(&grank,1,MPI_INT,0,lrank,MPI_COMM_WORLD,&request);
	  PetscLogEventEnd(events[0],0,0,0,0);
  }
	// Start the ball rolling
	if(grank == 0){
		PetscInt Nmanagers = 0;std::vector<PetscMPIInt> masters;
		PetscInt received_answers = 1, who, whomax;
		// Receive all the managers and place in a list
		while(Nmanagers <= ncolors){
		  PetscLogEventBegin(events[0],0,0,0,0);
		  MPE_Log_event(start_comm,0,"start-comm");
			MPI_Recv(&bufferRank,1,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
			MPE_Log_event(end_comm,0,"end-comm");
			PetscLogEventEnd(events[0],0,0,0,0);
			masters.push_back(bufferRank);
			++Nmanagers;
		}
		std::sort(masters.begin(),masters.end(),[](PetscMPIInt a,PetscMPIInt b){return (a<b);});
		std::for_each(masters.begin(),masters.end(),[](PetscMPIInt s){std::cout << s << std::endl;});
		PetscInt total_work = users.Nsamples;
		PetscScalar bufferNormU;
		whomax = ncolors;
		if(whomax > total_work)
			whomax = total_work;
		bufferBool = true;
		#if DEBUG
			PetscPrintf(MPI_COMM_WORLD,"I am processor %d in world, %d in petsc \n",grank,lrank);
		#endif
		for(who = 1; who <= whomax; ++who){
		  PetscLogEventBegin(events[0],0,0,0,0);
		  MPE_Log_event(start_comm,0,"start-comm");
			MPI_Isend(&bufferBool,1,MPI_C_BOOL,masters[who],WORKTAG,MPI_COMM_WORLD,&request);
			MPE_Log_event(end_comm,0,"end-comm");
			PetscLogEventEnd(events[0],0,0,0,0);
		}
		while ((received_answers <= total_work) && (tol > users.TOL)){
		  PetscLogEventBegin(events[0],0,0,0,0);
		  MPE_Log_event(start_comm,0,"start-comm");
			MPI_Recv(&bufferNormU,1,MPI_DOUBLE,MPI_ANY_SOURCE,WORKTAG,MPI_COMM_WORLD,&status);
			MPE_Log_event(end_comm,0,"end-comm");
			PetscLogEventEnd(events[0],0,0,0,0);
			who = status.MPI_SOURCE;
			#if 1
				PetscPrintf(MPI_COMM_WORLD,"Proc[%d]: Received norm from processor %d \n",grank,who);
			#endif
			normU = bufferNormU;
			PetscLogEventBegin(events[1],0,0,0,0);
			MPE_Log_event(start_comp,0,"start-comp");
			update_stats(EnormUN,VnormUN,EnormUNm1,M2NnU,tol,normU,received_answers);
			MPE_Log_event(end_comp,0,"end-comp");
			PetscLogEventEnd(events[1],0,0,0,0);
			++received_answers;
			if (Ns < users.Nsamples || tol > users.TOL){
			  PetscLogEventBegin(events[0],0,0,0,0);
			  MPE_Log_event(start_comm,0,"start-comm");
				MPI_Isend(&bufferBool,1,MPI_C_BOOL,who,WORKTAG,MPI_COMM_WORLD,&request);
				MPE_Log_event(start_comm,0,"end-comm");
				PetscLogEventEnd(events[0],0,0,0,0);
				++Ns;
			}
		}
		for (who = 1; who <= whomax; ++who){
			bufferBool = false;
			PetscLogEventBegin(events[0],0,0,0,0);
			MPE_Log_event(start_comm,0,"start-comm");
			MPI_Isend(&bufferBool,1,MPI_C_BOOL,masters[who],DIETAG,MPI_COMM_WORLD,&request);
			MPE_Log_event(end_comm,0,"end-comm");
			PetscLogEventEnd(events[0],0,0,0,0);
			PetscPrintf(MPI_COMM_WORLD,"Proc[%d]: Sending kill signal to proc %d\n",grank,masters[who]);
		}
		PetscPrintf(MPI_COMM_WORLD,"Expectation of ||U|| = %4.8E\n",EnormUN);
	}
	if (grank != 0){
		int work_status = DIETAG;
		if(lrank == 0) {
      MPE_Log_event(start_comm,0,"start-comm");
		  PetscLogEventBegin(events[0],0,0,0,0);
			MPI_Recv(&bufferBool,1,MPI_C_BOOL,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
			PetscLogEventEnd(events[0],0,0,0,0);
			MPE_Log_event(end_comm,0,"end-comm");
			work_status = status.MPI_TAG;
			PetscLogEventBegin(events[0],0,0,0,0);
			MPE_Log_event(start_comm,0,"start-comm");
			MPI_Bcast(&work_status,1,MPI_INT,0,petsc_comm_slaves);
			MPE_Log_event(end_comm,0,"end-comm");
			PetscLogEventEnd(events[0],0,0,0,0);
			#if DEBUG
				PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: I am broadcasting to my subordinates\n",grank);
			#endif
		}
		else {
      	PetscLogEventBegin(events[0],0,0,0,0);
      	MPE_Log_event(start_comm,0,"start-comm");
			MPI_Bcast(&work_status,1,MPI_INT,0,petsc_comm_slaves);
			MPE_Log_event(end_comm,0,"end-comm");
			PetscLogEventEnd(events[0],0,0,0,0);
			#if DEBUG
				PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: I am receiving broadcast\n",grank);
			#endif
		}
		if(work_status != DIETAG){
			while(true){
			  	MPE_Log_event(start_comp,0,"start-comp");
        ierr = UnitSolverTimings(rho,gmrf,N01,kspGMRF,U,b,A,kspSPDE,users,generator,lrank,Ns,bufferScalar,events,petsc_comm_slaves); CHKERRQ(ierr);
        	MPE_Log_event(end_comp,0,"end-comp");
				++Ns;				
				if(lrank == 0){
				  PetscLogEventBegin(events[0],0,0,0,0);
				  MPE_Log_event(start_comm,0,"start-comm");
					MPI_Send(&bufferScalar,1,MPI_DOUBLE,0,WORKTAG,MPI_COMM_WORLD);
					MPE_Log_event(end_comm,0,"start-comm");
					PetscLogEventEnd(events[0],0,0,0,0);
					#if DEBUG
					PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: Waiting for work\n",grank);
					#endif
					PetscLogEventBegin(events[0],0,0,0,0);
					MPE_Log_event(start_comm,0,"start-comm");
					MPI_Recv(&bufferBool,1,MPI_C_BOOL,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
					MPE_Log_event(end_comm,0,"end-comm");
					PetscLogEventEnd(events[0],0,0,0,0);
					work_status = status.MPI_TAG;
					PetscLogEventBegin(events[0],0,0,0,0);
					MPE_Log_event(start_comm,0,"start-comm");
					MPI_Bcast(&work_status,1,MPI_INT,0,petsc_comm_slaves);
					MPE_Log_event(end_comm,0,"end-comm");
					PetscLogEventEnd(events[0],0,0,0,0);
					#if DEBUG
					PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: I am broadcasting to my subordinates with tag = %d\n",grank,work_status);
					#endif
				}
				else{
				  PetscLogEventBegin(events[0],0,0,0,0);
				  MPE_Log_event(start_comm,0,"start-comm");
					MPI_Bcast(&work_status,1,MPI_INT,0,petsc_comm_slaves);
					MPE_Log_event(end_comm,0,"end-comm");
					PetscLogEventEnd(events[0],0,0,0,0);
					#if DEBUG
					PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: I am receiving broadcast with tag = %d\n",grank,work_status);
					#endif
				}
				if(work_status == DIETAG){
					#if DEBUG
					PetscPrintf(petsc_comm_slaves,"Proc[%d]: We finished all work \n",grank);	
					#endif
					break;
				}
			}
		}
	}

	endTime = MPI_Wtime();
	MPE_Finish_log("Test8");
	PetscPrintf(MPI_COMM_WORLD,"Proc[%d]: All done! \n",grank);
	if (grank != 0) PetscPrintf(petsc_comm_slaves,"Proc[%d]: We did %d samples \n",grank,(Ns-1));
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
