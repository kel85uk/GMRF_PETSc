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
	KSP kspSPDE, kspGMRF;
	PetscInt		Ns = 0, bufferInt;
	PetscInt  *buffer_iterations = new PetscInt[2];
	UserCTX		users;
	Buffer_messages sendrecvbuff;
	PetscMPIInt    grank, lrank, bufferRank;
	PetscErrorCode ierr;
	PetscBool      flg = PETSC_FALSE;
	bool bufferBool = false;
	PetscScalar startTime, endTime, bufferScalar;
	int numprocs, ncolors;
	MPI_Status status;MPI_Request request;
	int ranks[] = {0};
	MPI_Init(&argc,&argv);
	MPI_Aint extent, offsets[2];
  MPI_Type_extent(MPI_DOUBLE,&extent);
  offsets[0] = 0;
  offsets[1] = extent;
  int blockcounts[2];
  blockcounts[0] = 1;
  blockcounts[1] = 2;
  MPI_Datatype oldtypes[2],RESULT_SND_RECV;
  oldtypes[0] = MPI_DOUBLE;
  oldtypes[1] = MPI_INT;
  MPI_Type_create_struct(2,blockcounts,offsets,oldtypes,&RESULT_SND_RECV);
  MPI_Type_commit(&RESULT_SND_RECV);
  

	MPI_Comm petsc_comm_slaves;

	MPI_Comm_rank(MPI_COMM_WORLD,&grank); // Get the processor global rank
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	
	PetscInitialize(&argc,&argv,(char*)0,help);
	ierr = GetOptions(users);CHKERRQ(ierr);
	recolor(petsc_comm_slaves,ncolors,numprocs,users.procpercolor,grank,lrank);
	/* Split the different communicators between root and workers */
	startTime = MPI_Wtime();
	srand(grank);
	std::default_random_engine generator(rand());	

	/* Create all the vectors and matrices needed for calculation */
	ierr = CreateVectors(*Wrapalla,12,users.NT,petsc_comm_slaves);CHKERRQ(ierr);
	ierr = CreateVectors(*Wrapallb,7,users.NI,petsc_comm_slaves);CHKERRQ(ierr);
	/* Create Matrices and Solver contexts */
	ierr = CreateSolvers(L,users.NT,kspGMRF,A,users.NI,kspSPDE,petsc_comm_slaves);CHKERRQ(ierr);

	PetscScalar normU = 0.,EnormUN = 0.,VnormUN = 0.,EnormUNm1 = 0.,M2NnU = 0.,tol = 1.0;

	ierr = SetGMRFOperator(L,users.m,users.n,users.NGhost,users.dx,users.dy,users.kappa);CHKERRQ(ierr);
	ierr = KSPSetOperators(kspGMRF,L,L,SAME_PRECONDITIONER);CHKERRQ(ierr);

	// Managers report duty to the root processor
	if(lrank == 0) MPI_Isend(&grank,1,MPI_INT,0,lrank,MPI_COMM_WORLD,&request);
	// Start the ball rolling
	if(grank == 0){
		PetscInt Nmanagers = 0;std::vector<PetscMPIInt> masters;
		PetscInt received_answers = 1, who, whomax;
		PetscScalar *avg_iter = new PetscScalar[2];
		PetscInt *max_iter = new PetscInt[2], *min_iter = new PetscInt[2];
		for (int ii = 0; ii < (sizeof(max_iter)/sizeof(PetscInt)); ++ii){
      avg_iter[ii] = 0.;
      max_iter[ii] = 0;
      min_iter[ii] = 299792458;
	  }
		// Receive all the managers and place in a list
		while(Nmanagers <= ncolors){
			MPI_Recv(&bufferRank,1,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
			masters.push_back(bufferRank);
			++Nmanagers;
		}
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
		for(who = 1; who <= whomax; ++who){
			MPI_Isend(&bufferBool,1,MPI_C_BOOL,masters[who],WORKTAG,MPI_COMM_WORLD,&request);
		}
		while ((received_answers <= total_work) && (tol > users.TOL)){
			MPI_Recv(&sendrecvbuff,1,RESULT_SND_RECV,MPI_ANY_SOURCE,WORKTAG,MPI_COMM_WORLD,&status);
			who = status.MPI_SOURCE;
			#if 1
				PetscPrintf(MPI_COMM_WORLD,"Proc[%d]: Received norm from processor %d \n",grank,who);
			#endif
			normU = sendrecvbuff.normU;
			update_stats(EnormUN,VnormUN,EnormUNm1,M2NnU,tol,normU,received_answers);
			buffer_iterations[0] = sendrecvbuff.gmrf_iter;
			buffer_iterations[1] = sendrecvbuff.spde_iter;
			update_iters(avg_iter,max_iter,min_iter,buffer_iterations,Ns);
			for (int ii = 0; ii < sizeof(max_iter)/sizeof(PetscInt); ++ii)
        PetscPrintf(MPI_COMM_WORLD,"Iters[%d]: Avg(%g), Max(%d), Min(%d) \n",ii,avg_iter[ii],max_iter[ii],min_iter[ii]);
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
	if (grank != 0){
		int work_status = DIETAG;
		if(lrank == 0) {
			MPI_Recv(&bufferBool,1,MPI_C_BOOL,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
			work_status = status.MPI_TAG;
			MPI_Bcast(&work_status,1,MPI_INT,0,petsc_comm_slaves);
			#if DEBUG
				PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: I am broadcasting to my subordinates\n",grank);
			#endif
		}
		else {
			MPI_Bcast(&work_status,1,MPI_INT,0,petsc_comm_slaves);
			#if DEBUG
				PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: I am receiving broadcast\n",grank);
			#endif
		}
		if(work_status != DIETAG){
			while(true){
				ierr = UnitSolver(rho,gmrf,N01,kspGMRF,U,b,A,kspSPDE,users,generator,lrank,Ns,bufferScalar,buffer_iterations,petsc_comm_slaves); CHKERRQ(ierr);
				++Ns;
				sendrecvbuff.normU = bufferScalar;
				sendrecvbuff.gmrf_iter = buffer_iterations[0];
				sendrecvbuff.spde_iter = buffer_iterations[1];
				#if DEBUG
				PetscPrintf(petsc_comm_slaves,"Proc[%d]: GMRF iter = %d, SPDE iter = %d\n",grank,buffer_iterations[0],buffer_iterations[1]);
				#endif
				if(lrank == 0){
					MPI_Send(&sendrecvbuff,1,RESULT_SND_RECV,0,WORKTAG,MPI_COMM_WORLD);
					#if DEBUG
					PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: Waiting for work\n",grank);
					#endif
					MPI_Recv(&bufferBool,1,MPI_C_BOOL,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
					work_status = status.MPI_TAG;
					MPI_Bcast(&work_status,1,MPI_INT,0,petsc_comm_slaves);
					#if DEBUG
					PetscPrintf(PETSC_COMM_SELF,"Proc[%d]: I am broadcasting to my subordinates with tag = %d\n",grank,work_status);
					#endif
				}
				else{
					MPI_Bcast(&work_status,1,MPI_INT,0,petsc_comm_slaves);
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
