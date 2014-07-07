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
#define MPE_log
#define DEBUG 0
#define MPI_WTIME_IS_GLOBAL 1
#define WORKTAG 1
#define DIETAG 2

void CreateMPEEvents(std::vector<int>&);

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
	MPI_Comm petsc_comm_slaves;
	MPI_Init(&argc,&argv);

	MPI_Comm_rank(MPI_COMM_WORLD,&grank); // Get the processor global rank
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	/* Split the communicator into petsc_comm_slaves */
	int ncolors = (numprocs-1)/procpercolor;
	int color = (grank-1)/procpercolor;
	if(grank == 0) color = std::numeric_limits<PetscInt>::max();
	if ((color+1) > ncolors)
		color--;
	#if DEBUG
		printf("I am processor %d with color %d \n",grank,color);
	#endif
	MPI_Comm_split(MPI_COMM_WORLD, color, grank, &petsc_comm_slaves);
	
	PetscInitialize(&argc,&argv,(char*)0,help);
	MPI_Comm_rank(petsc_comm_slaves,&lrank); // Get the processor local rank
  std::vector<PetscLogEvent> petscevents;
	PetscLogEvent gmrf_COMMS, gmrf_SET, gmrf_SOLVE, spde_COMMS, spde_SET, spde_SOLVE, internode_COMMS, update_COMPS;
	PetscClassId gmrf_Comms, gmrf_Set, gmrf_Solve, spde_Comms, spde_Set, spde_Solve, internode_Comms, update_Comps;
	PetscClassIdRegister("GMRF Comms",&gmrf_Comms);
  PetscLogEventRegister("GMRF Communications",gmrf_Comms,&gmrf_COMMS);
  PetscClassIdRegister("GMRF Set",&gmrf_Set);
  PetscLogEventRegister("GMRF Setup",gmrf_Set,&gmrf_SET);
  PetscClassIdRegister("GMRF Sol",&gmrf_Solve);
  PetscLogEventRegister("GMRF Solve",gmrf_Solve,&gmrf_SOLVE);
  PetscClassIdRegister("SPDE Comms",&spde_Comms);
  PetscLogEventRegister("SPDE Communications",spde_Comms,&spde_COMMS);
  PetscClassIdRegister("SPDE Set",&spde_Set);
  PetscLogEventRegister("SPDE Setup",spde_Set,&spde_SET);
  PetscClassIdRegister("SPDE Sol",&spde_Solve);
  PetscLogEventRegister("SPDE Solve",spde_Solve,&spde_SOLVE);
  PetscClassIdRegister("MS Comms",&internode_Comms);
  PetscLogEventRegister("MS Communications",internode_Comms,&internode_COMMS);
  PetscClassIdRegister("Misc Comps",&update_Comps);
  PetscLogEventRegister("Misc Computations",update_Comps,&update_COMPS);
  petscevents.push_back(gmrf_COMMS);
  petscevents.push_back(gmrf_SET);
  petscevents.push_back(gmrf_SOLVE);
  petscevents.push_back(spde_COMMS);
  petscevents.push_back(spde_SET);
  petscevents.push_back(spde_SOLVE);
  petscevents.push_back(internode_COMMS);
  petscevents.push_back(update_COMPS);
  std::vector<int> MPE_events;
  	MPE_Init_log();
	PetscLogBegin();
  CreateMPEEvents(MPE_events);
	/* Split the different communicators between root and workers */
	startTime = MPI_Wtime();
	srand(grank);
	std::default_random_engine generator(rand());
	ierr = GetOptions(users);CHKERRQ(ierr);
	/* Create all the vectors and matrices needed for calculation */
	PetscLogEventBegin(petscevents[7],0,0,0,0);
	MPE_Log_event(MPE_events[14],0,"Misc Comp-start");
	ierr = CreateVectors(*Wrapalla,12,users.NT,petsc_comm_slaves);CHKERRQ(ierr);
	ierr = CreateVectors(*Wrapallb,7,users.NI,petsc_comm_slaves);CHKERRQ(ierr);
	/* Create Matrices and Solver contexts */
	ierr = CreateSolvers(L,users.NT,kspGMRF,A,users.NI,kspSPDE,petsc_comm_slaves);CHKERRQ(ierr);
  	PetscScalar normU = 0.,EnormUN = 0.,VnormUN = 0.,EnormUNm1 = 0.,M2NnU = 0.,tol = 1.0;
  PetscLogEventEnd(petscevents[7],0,0,0,0);
  MPE_Log_event(MPE_events[15],0,"Misc Comp-end");
	ierr = SetGMRFOperatorT(L,users.m,users.n,users.NGhost,users.dx,users.dy,users.kappa,petscevents,MPE_events);CHKERRQ(ierr);
	PetscLogEventBegin(petscevents[1],0,0,0,0);
	MPE_Log_event(MPE_events[2],0,"GMRF Set-start");
	ierr = KSPSetOperators(kspGMRF,L,L,SAME_PRECONDITIONER);CHKERRQ(ierr);
	PetscLogEventEnd(petscevents[1],0,0,0,0);
	MPE_Log_event(MPE_events[3],0,"GMRF Set-end");
	// Managers report duty to the root processor
	if(lrank == 0) {
	  PetscLogEventBegin(petscevents[6],0,0,0,0);
	  MPE_Log_event(MPE_events[12],0,"MS Comm-start");
	  MPI_Isend(&grank,1,MPI_INT,0,lrank,MPI_COMM_WORLD,&request);
	  PetscLogEventEnd(petscevents[6],0,0,0,0);
	  MPE_Log_event(MPE_events[13],0,"MS Comm-end");
  }
	// Start the ball rolling
	if(grank == 0){
		PetscInt Nmanagers = 0;std::vector<PetscMPIInt> masters;
		PetscInt received_answers = 1, who, whomax;
		// Receive all the managers and place in a list
		while(Nmanagers <= ncolors){
		  PetscLogEventBegin(petscevents[6],0,0,0,0);
		  MPE_Log_event(MPE_events[12],0,"MS Comm-start");
			MPI_Recv(&bufferRank,1,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
			PetscLogEventEnd(petscevents[6],0,0,0,0);
			MPE_Log_event(MPE_events[13],0,"MS Comm-end");
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
		  PetscLogEventBegin(petscevents[6],0,0,0,0);
		  MPE_Log_event(MPE_events[12],0,"MS Comm-start");
			MPI_Isend(&bufferBool,1,MPI_C_BOOL,masters[who],WORKTAG,MPI_COMM_WORLD,&request);
			PetscLogEventEnd(petscevents[6],0,0,0,0);
			MPE_Log_event(MPE_events[13],0,"MS Comm-end");
		}
		while ((received_answers <= total_work) && (tol > users.TOL)){
		  PetscLogEventBegin(petscevents[6],0,0,0,0);
		  MPE_Log_event(MPE_events[12],0,"MS Comm-start");
			MPI_Recv(&bufferNormU,1,MPI_DOUBLE,MPI_ANY_SOURCE,WORKTAG,MPI_COMM_WORLD,&status);
			PetscLogEventEnd(petscevents[6],0,0,0,0);
			MPE_Log_event(MPE_events[13],0,"MS Comm-end");
			who = status.MPI_SOURCE;
			#if 1
				PetscPrintf(MPI_COMM_WORLD,"Proc[%d]: Received norm from processor %d \n",grank,who);
			#endif
			normU = bufferNormU;
			PetscLogEventBegin(petscevents[7],0,0,0,0);
			MPE_Log_event(MPE_events[14],0,"Misc Comp-start");
			update_stats(EnormUN,VnormUN,EnormUNm1,M2NnU,tol,normU,received_answers);
			PetscLogEventEnd(petscevents[7],0,0,0,0);
			MPE_Log_event(MPE_events[15],0,"Misc Comp-end");
			++received_answers;
			if (Ns < users.Nsamples || tol > users.TOL){
			  PetscLogEventBegin(petscevents[6],0,0,0,0);
			  MPE_Log_event(MPE_events[12],0,"MS Comm-start");
				MPI_Isend(&bufferBool,1,MPI_C_BOOL,who,WORKTAG,MPI_COMM_WORLD,&request);
				PetscLogEventEnd(petscevents[6],0,0,0,0);
				MPE_Log_event(MPE_events[13],0,"MS Comm-end");
				++Ns;
			}
		}
		for (who = 1; who <= whomax; ++who){
			bufferBool = false;
			PetscLogEventBegin(petscevents[6],0,0,0,0);
			MPE_Log_event(MPE_events[12],0,"MS Comm-start");
			MPI_Isend(&bufferBool,1,MPI_C_BOOL,masters[who],DIETAG,MPI_COMM_WORLD,&request);
			PetscLogEventEnd(petscevents[6],0,0,0,0);
			MPE_Log_event(MPE_events[13],0,"MS Comm-end");
			PetscPrintf(MPI_COMM_WORLD,"Proc[%d]: Sending kill signal to proc %d\n",grank,masters[who]);
		}
		PetscPrintf(MPI_COMM_WORLD,"Expectation of ||U|| = %4.8E\n",EnormUN);
	}
	if (grank != 0){
		int work_status = DIETAG;
		if(lrank == 0) {
		  PetscLogEventBegin(petscevents[6],0,0,0,0);
		  MPE_Log_event(MPE_events[12],0,"MS Comm-start");
			MPI_Recv(&bufferBool,1,MPI_C_BOOL,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
			PetscLogEventEnd(petscevents[6],0,0,0,0);
			MPE_Log_event(MPE_events[13],0,"MS Comm-end");
			work_status = status.MPI_TAG;
			PetscLogEventBegin(petscevents[6],0,0,0,0);
			MPE_Log_event(MPE_events[12],0,"MS Comm-start");
			MPI_Bcast(&work_status,1,MPI_INT,0,petsc_comm_slaves);
			PetscLogEventEnd(petscevents[6],0,0,0,0);
			MPE_Log_event(MPE_events[13],0,"MS Comm-end");
			#if DEBUG
				PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: I am broadcasting to my subordinates\n",grank);
			#endif
		}
		else {
      	PetscLogEventBegin(petscevents[6],0,0,0,0);
      	MPE_Log_event(MPE_events[12],0,"MS Comm-start");
			MPI_Bcast(&work_status,1,MPI_INT,0,petsc_comm_slaves);
			PetscLogEventEnd(petscevents[6],0,0,0,0);
			MPE_Log_event(MPE_events[13],0,"MS Comm-end");
			#if DEBUG
				PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: I am receiving broadcast\n",grank);
			#endif
		}
		if(work_status != DIETAG){
			while(true){
        ierr = UnitSolverTimings(rho,gmrf,N01,kspGMRF,U,b,A,kspSPDE,users,generator,lrank,Ns,bufferScalar,petscevents,MPE_events,petsc_comm_slaves); CHKERRQ(ierr);
				++Ns;				
				if(lrank == 0){
				  PetscLogEventBegin(petscevents[6],0,0,0,0);
				  MPE_Log_event(MPE_events[12],0,"MS Comm-start");
					MPI_Send(&bufferScalar,1,MPI_DOUBLE,0,WORKTAG,MPI_COMM_WORLD);
					PetscLogEventEnd(petscevents[6],0,0,0,0);
					MPE_Log_event(MPE_events[13],0,"MS Comm-end");
					#if DEBUG
					PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: Waiting for work\n",grank);
					#endif
					PetscLogEventBegin(petscevents[6],0,0,0,0);
					MPE_Log_event(MPE_events[12],0,"MS Comm-start");
					MPI_Recv(&bufferBool,1,MPI_C_BOOL,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
					PetscLogEventEnd(petscevents[6],0,0,0,0);
					MPE_Log_event(MPE_events[13],0,"MS Comm-end");
					work_status = status.MPI_TAG;
					PetscLogEventBegin(petscevents[6],0,0,0,0);
					MPE_Log_event(MPE_events[12],0,"MS Comm-start");
					MPI_Bcast(&work_status,1,MPI_INT,0,petsc_comm_slaves);
					PetscLogEventEnd(petscevents[6],0,0,0,0);
					MPE_Log_event(MPE_events[13],0,"MS Comm-end");
					#if DEBUG
					PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: I am broadcasting to my subordinates with tag = %d\n",grank,work_status);
					#endif
				}
				else{
				  PetscLogEventBegin(petscevents[6],0,0,0,0);
          MPE_Log_event(MPE_events[12],0,"MS Comm-start");
					MPI_Bcast(&work_status,1,MPI_INT,0,petsc_comm_slaves);
					PetscLogEventEnd(petscevents[6],0,0,0,0);
					MPE_Log_event(MPE_events[13],0,"MS Comm-end");
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

void CreateMPEEvents(std::vector<int>& MPE_events){
  MPE_events.resize(16);
  for_each(MPE_events.begin(),MPE_events.end(),[](int& s){s = MPE_Log_get_event_number();});
  MPE_Describe_state(MPE_events[0],MPE_events[1],"GMRF Comm","purple4");
  MPE_Describe_state(MPE_events[2],MPE_events[3],"GMRF Setup","orange3");
  MPE_Describe_state(MPE_events[4],MPE_events[5],"GMRF Solve","cyan");
  MPE_Describe_state(MPE_events[6],MPE_events[7],"SPDE Comm","red");
  MPE_Describe_state(MPE_events[8],MPE_events[9],"SPDE Setup","yellow");
  MPE_Describe_state(MPE_events[10],MPE_events[11],"SPDE Solve","green");
  MPE_Describe_state(MPE_events[12],MPE_events[13],"MS Comm","maroon");
  MPE_Describe_state(MPE_events[14],MPE_events[15],"Misc Comp","blue");
}
