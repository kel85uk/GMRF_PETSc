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
//#include <boost/math/special_functions/bessel.hpp>
#define DEBUG 0
#define MPI_WTIME_IS_GLOBAL 1
#define WORKTAG 1
#define DIETAG 2

PetscErrorCode CovarMatCreate(Mat&, const UserCTX&);
PetscErrorCode GetCholFactor(PC&,const Mat&);
PetscScalar matern_cov(const UserCTX&,const PetscScalar&);
void set_coordinates(std::vector<PetscScalar>& XR,std::vector<PetscScalar>& YR,const UserCTX& users){
	PetscScalar yr,xr;
	PetscInt II;
	XR.resize(users.NT,0.);
	YR.resize(users.NT,0.);
	/* Set grid coordinates (with ghost cells)*/
	yr = users.y0 + (0.5-(PetscScalar)users.NGhost)*users.dy;
	for (int j=0; j < users.n + 2*users.NGhost; ++j) {
		xr = users.x0 + (0.5-(PetscScalar)users.NGhost)*users.dx;
	for (int i=0; i < users.m + 2*users.NGhost; ++i) {
		II = i + (users.m + 2*users.NGhost)*j;
		XR[II] = xr;
		YR[II] = yr;
		xr += users.dx;
	}
		yr += users.dy;
	}
}

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
	int procpercolor = 1;
	MPI_Status status;MPI_Request request;
	int ranks[] = {0};

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
	ierr = GetOptions(users);CHKERRQ(ierr);
	users.NGhost = 1; // Only change the Nghost for direct solvers (It's a hack!)
	users.NT = (users.m+2*users.NGhost)*(users.n+2*users.NGhost); // Ditto
	/* Create all the vectors and matrices needed for calculation */
	ierr = CreateVectors(*Wrapalla,12,users.NT);CHKERRQ(ierr);
	ierr = CreateVectors(*Wrapallb,7,users.NI);CHKERRQ(ierr);
	/* Create Matrices and Solver contexts */
	ierr = CreateSolvers(L,users.NT,kspGMRF,A,users.NI,kspSPDE);CHKERRQ(ierr);

	PetscScalar normU = 0.,EnormUN = 0.,VnormUN = 0.,EnormUNm1 = 0.,M2NnU = 0.,tol = 1.0;

	// Managers report duty to the root processor
	if(lrank == 0) MPI_Isend(&grank,1,MPI_INT,0,lrank,MPI_COMM_WORLD,&request);
	// Start the ball rolling
	if(grank == 0){
		PetscInt Nmanagers = 0;std::vector<PetscMPIInt> masters;
		PetscInt received_answers = 1, who, whomax;
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
			MPI_Recv(&bufferNormU,1,MPI_DOUBLE,MPI_ANY_SOURCE,WORKTAG,MPI_COMM_WORLD,&status);
			who = status.MPI_SOURCE;
			#if 1
				PetscPrintf(MPI_COMM_WORLD,"Proc[%d]: Received norm from processor %d \n",grank,who);
			#endif
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
	if (grank != 0){
		int work_status = DIETAG;
		Vec normrnds;
		Mat Covar;PC Chol_fac;
		ierr = VecDuplicate(rho,&normrnds); CHKERRQ(ierr);
		// Create the covariance matrix here
		ierr = CovarMatCreate(Covar,users); CHKERRQ(ierr);
		std::cout << "All ok! covar" << std::endl;
		// Add the Cholesky factorization routines here
		ierr = GetCholFactor(Chol_fac,Covar);CHKERRQ(ierr);
		std::cout << "All ok! cholfac" << std::endl;
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
				ierr = UnitSolverChol(rho,normrnds,Chol_fac,Covar,U,b,A,kspSPDE,users,generator,lrank,Ns,bufferScalar); CHKERRQ(ierr);
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
					ierr = VecDestroy(&normrnds); CHKERRQ(ierr);
					ierr = MatDestroy(&Covar); CHKERRQ(ierr);
					ierr = PCDestroy(&Chol_fac); CHKERRQ(ierr);
					break;
				}
			}
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

PetscErrorCode CovarMatCreate(Mat& Covar,const UserCTX& users){
  PetscErrorCode ierr;
  PetscBool flg;
  PetscInt Istart, Iend, Ii, Ij;
  PetscScalar cov, radius;
  std::vector<PetscScalar> XR,YR;
  set_coordinates(XR,YR,users);
  ierr = MatCreate(PETSC_COMM_WORLD,&Covar);CHKERRQ(ierr); // Create matrix A residing in PETSC_COMM_WORLD
	ierr = MatSetSizes(Covar,PETSC_DECIDE,PETSC_DECIDE,users.NT,users.NT);CHKERRQ(ierr); // Set the size of the matrix A, and let PETSC decide the decomposition
	ierr = MatSetFromOptions(Covar);CHKERRQ(ierr);
	ierr = MatMPIAIJSetPreallocation(Covar,users.NT,NULL,users.NT,NULL);CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(Covar,users.NT,NULL);CHKERRQ(ierr);
	ierr = MatSetUp(Covar);CHKERRQ(ierr);
	ierr = MatGetOwnershipRange(Covar,&Istart,&Iend);CHKERRQ(ierr);
	for (Ii = Istart; Ii < Iend; ++Ii)
	  for (Ij = 0; Ij < users.NT; ++Ij){
	    radius = sqrt((XR[Ii] - XR[Ij])*(XR[Ii] - XR[Ij]) + (YR[Ii] - YR[Ij])*(YR[Ii] - YR[Ij]));
	    cov    = matern_cov(users,radius);
	    ierr   = MatSetValue(Covar,Ii,Ij,cov,INSERT_VALUES);CHKERRQ(ierr);
    }
  ierr = MatAssemblyBegin(Covar,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Covar,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatIsSymmetric(Covar,1e-6,&flg); CHKERRQ(ierr);
  if (flg) PetscPrintf(PETSC_COMM_WORLD,"Covar matrix is symmetric \n");

  return ierr;
}

PetscErrorCode GetCholFactor(PC& Chol_fac,const Mat& Covar){
  PetscErrorCode ierr;
  PC& pcchol = Chol_fac;
  ierr = PCCreate(PETSC_COMM_WORLD,&pcchol);CHKERRQ(ierr);
  ierr = PCSetType(pcchol,PCCHOLESKY);CHKERRQ(ierr);
  ierr = PCSetOperators(pcchol,Covar,Covar,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverPackage(pcchol,MATSOLVERMUMPS);CHKERRQ(ierr);
  ierr = PCSetUp(pcchol); CHKERRQ(ierr);
  return ierr;
}

PetscScalar matern_cov(const UserCTX& users,const PetscScalar& rad_d){
  PetscScalar res;
/*  if (rad_d <= 0)
    res = users.sigma*users.sigma;
  else
//    res = boost::math::cyl_bessel_k(users.nu,users.kappa*rad_d);
    res = users.sigma*users.sigma/(std::pow(2.0,(users.nu-1.0))*tgamma(users.nu))*std::pow(users.kappa*rad_d,users.nu)*boost::math::cyl_bessel_k(users.nu,users.kappa*rad_d); */
  res = users.sigma*users.sigma*exp(-rad_d/users.lamb);
	return res;
}
