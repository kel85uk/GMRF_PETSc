/* Test Program for PDE routines */

static char help[] = "Solves a linear system in parallel with KSP.\n\
Input parameters include:\n\
  -random_exact_sol : use a random exact solution vector\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_n>       : number of mesh points in y-direction\n\n";

#include <Functions.hh>
#include <vector>

#define nint(a) ((a) >= 0.0 ? (PetscInt)((a)+0.5) : (PetscInt)((a)-0.5))


PetscScalar	rho_calculate(const PetscScalar& radius,const PetscScalar& RE){
	PetscScalar rho_mean = 0.;
	if(radius <= RE){
		rho_mean = (1.0 - 100.0)*radius/RE + 100.0;	//PetscSinScalar(2.*PETSC_PI*xr)*PetscSinScalar(2.*PETSC_PI*yr);
	}
	else{
		rho_mean = 1.0;
	}
	return rho_mean;
}

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
	Vec U, Z, rho;
	Mat L;
	PetscInt Ns, Istart, Iend;
	PetscScalar xr,yr, rho_mean, radius, RE = 0.25;
	std::vector<PetscScalar>		XR,YR; // Physical Coordinates reside in all processors
	UserCTX users;
	KSP kspPDE;
	PetscScalar	result, normU;
	PetscErrorCode ierr;
	PetscBool      flg = PETSC_FALSE;
	PetscMPIInt rank;	
	PetscInitialize(&argc,&argv,(char*)0,help);
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);


	ierr = GetOptions(users);CHKERRQ(ierr);
	set_coordinates(XR,YR,users);
	
	ierr = VecCreate(PETSC_COMM_WORLD,&U);CHKERRQ(ierr);
	ierr = VecSetSizes(U,PETSC_DECIDE,users.NI);CHKERRQ(ierr);
	ierr = VecSetFromOptions(U);CHKERRQ(ierr);
	ierr = VecSet(U,0.);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&Z);CHKERRQ(ierr);
	ierr = VecCopy(U,Z);CHKERRQ(ierr);
	
	ierr = VecCreate(PETSC_COMM_WORLD,&rho);CHKERRQ(ierr);
	ierr = VecSetSizes(rho,PETSC_DECIDE,users.NT);CHKERRQ(ierr);
	ierr = VecSetFromOptions(rho);CHKERRQ(ierr);
	ierr = VecSet(rho,1.);CHKERRQ(ierr);
	
	ierr = VecGetOwnershipRange(rho,&Istart,&Iend);CHKERRQ(ierr);
	for (int II = Istart; II < Iend; ++II){
		radius = std::sqrt((XR[II] - 0.5)*(XR[II] - 0.5) + (YR[II] - 0.5)*(YR[II] - 0.5));
		rho_mean = rho_calculate(radius,RE);
		ierr = VecSetValues(rho,1,&II,&rho_mean,INSERT_VALUES);CHKERRQ(ierr);
	}
	ierr = VecAssemblyBegin(rho);CHKERRQ(ierr);
		ierr = MatCreate(PETSC_COMM_WORLD,&L);CHKERRQ(ierr); // Create matrix A residing in PETSC_COMM_WORLD
		ierr = MatSetSizes(L,PETSC_DECIDE,PETSC_DECIDE,users.NI,users.NI);CHKERRQ(ierr); // Set the size of the matrix A, and let PETSC decide the decomposition
		ierr = MatSetFromOptions(L);CHKERRQ(ierr);
		ierr = MatMPIAIJSetPreallocation(L,5,NULL,5,NULL);CHKERRQ(ierr);
		ierr = MatSeqAIJSetPreallocation(L,5,NULL);CHKERRQ(ierr);
		ierr = MatSetUp(L);CHKERRQ(ierr);	
		ierr = KSPCreate(PETSC_COMM_WORLD,&kspPDE);CHKERRQ(ierr);
		ierr = KSPSetTolerances(kspPDE,1.e-7/(users.NI),1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
		ierr = KSPSetFromOptions(kspPDE);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(rho);CHKERRQ(ierr);
	
	ierr = SetOperator(L,rho,users.m,users.n,users.NGhost,users.dx,users.dy);CHKERRQ(ierr);
	ierr = SetSource(Z,rho,users.m,users.n,users.NGhost,users.dx,users.dy,users.UN,users.US,users.UE,users.UW,users.lamb);CHKERRQ(ierr);
	ierr = KSPSetOperators(kspPDE,L,L,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
	ierr = KSPSolve(kspPDE,Z,U);CHKERRQ(ierr);
	ierr = KSPGetIterationNumber(kspPDE,&users.its);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Sample[%d]: SPDE solved in %d iterations \n",Ns,users.its);CHKERRQ(ierr);
	ierr = VecNorm(U,NORM_2,&normU);CHKERRQ(ierr);
	normU /= sqrt(users.NI);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm-2 of U = %4.8E \n",normU);CHKERRQ(ierr);
	if(false){
	ierr = VecPostProcs(U,"sol_mean.dat",rank);CHKERRQ(ierr);
	ierr = VecPostProcs(rho,"rho_mean.dat",rank);CHKERRQ(ierr);
	VecPostProcs(XR,"XR.dat",rank);
	VecPostProcs(YR,"YR.dat",rank);
	}
	PetscPrintf(PETSC_COMM_WORLD,"NGhost = %d and I am Processor[%d] \n",users.NGhost,rank);
	PetscPrintf(PETSC_COMM_WORLD,"tau2 = %f \n",users.tau2);
	PetscPrintf(PETSC_COMM_WORLD,"kappa = %f \n",users.kappa);
	PetscPrintf(PETSC_COMM_WORLD,"nu = %f \n",users.nu);
	PetscPrintf(PETSC_COMM_WORLD,"dx = %f \n",users.dx);
	PetscPrintf(PETSC_COMM_WORLD,"dy = %f \n",users.dy);	

	ierr = VecDestroy(&U);CHKERRQ(ierr);
	ierr = VecDestroy(&rho);CHKERRQ(ierr);
	ierr = VecDestroy(&Z);CHKERRQ(ierr);
	
	ierr = KSPDestroy(&kspPDE);CHKERRQ(ierr);
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
