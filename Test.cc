/* Test Program for Petsc */

static char help[] = "Solves a linear system in parallel with KSP.\n\
Input parameters include:\n\
  -random_exact_sol : use a random exact solution vector\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_n>       : number of mesh points in y-direction\n\n";

#include <petsc.h>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <random>
#include <cmath>
#include <Functions.hh>

#define nint(a) ((a) >= 0.0 ? (PetscInt)((a)+0.5) : (PetscInt)((a)-0.5))

int main(int argc,char **argv)
{
	Vec U, EUNm1, EUN, VUN, VUNm1, b, resU, rho, ErhoNm1, ErhoN, VrhoN, VrhoNm1, N01, M2N, M2Nm1, dUN, dUNm1, M2Nr, M2Nm1r, drhoN, drhoNm1, resR;
	Mat A, L;
	KSP kspSPDE, kspGMRF;
	PetscReal		normE, normV, x0 = 0, x1 = 1, y0 = 0, y1 = 1, dx, dy;     /* norm of solution error */
	PetscReal		UN = 1, US = 10, UE = 5, UW = 3;
	PetscReal		dim = 2, alpha = 2, sigma = 0.3, lamb = 0.1, nu, kappa, tol = 1, TOL = 1e-4, tolE, tolV;
	PetscInt		m = 8, n = 7, its, NGhost = 2, NI = 0, NT = 0, Nsamples = 1, Ns = 1;
	PetscScalar	result;
	nu = alpha - dim/2;
	kappa = sqrt(8*nu)/(lamb);	
	PetscMPIInt    rank;
	dx = (x1 - x0)/m;
	dy = (y1 - y0)/n;
	PetscErrorCode ierr;
	PetscBool      flg = PETSC_FALSE;
	boost::variate_generator<boost::mt19937, boost::normal_distribution<> >	generator(boost::mt19937(time(0)),boost::normal_distribution<>(0.,1.));
	PetscInitialize(&argc,&argv,(char*)0,help);
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	ierr = GetOptions(m,n,x0,x1,y0,y1,UN,US,UE,UW,lamb);
	NGhost += PetscMax(PetscMax(nint(0.5*lamb/dx),nint(0.5*lamb/dy)),nint(0.5*lamb/(sqrt(dx*dx + dy*dy))));
	NI = m*n;
	NT = (m+2*NGhost)*(n+2*NGhost);
	PetscPrintf(PETSC_COMM_SELF,"NGhost = %d and I am Processor[%d] \n",NGhost,rank);
	/* Create all the vectors and matrices needed for calculation */
	ierr = VecCreate(PETSC_COMM_WORLD,&rho);CHKERRQ(ierr);
	ierr = VecSetSizes(rho,PETSC_DECIDE,NT);CHKERRQ(ierr);
	ierr = VecSetFromOptions(rho);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&ErhoNm1);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&ErhoN);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&M2Nr);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&M2Nm1r);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&drhoNm1);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&drhoN);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&VrhoN);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&VrhoNm1);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&N01);CHKERRQ(ierr);
	ierr = VecDuplicate(rho,&resR);CHKERRQ(ierr);
	ierr = MatCreate(PETSC_COMM_WORLD,&L);CHKERRQ(ierr); // Create matrix A residing in PETSC_COMM_WORLD
	ierr = MatSetSizes(L,PETSC_DECIDE,PETSC_DECIDE,NT,NT);CHKERRQ(ierr); // Set the size of the matrix A, and let PETSC decide the decomposition
	ierr = MatSetFromOptions(L);CHKERRQ(ierr);
	ierr = MatMPIAIJSetPreallocation(L,5,NULL,5,NULL);CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(L,5,NULL);CHKERRQ(ierr);
	ierr = MatSetUp(L);CHKERRQ(ierr);	
	ierr = KSPCreate(PETSC_COMM_WORLD,&kspGMRF);CHKERRQ(ierr);
	ierr = KSPSetTolerances(kspGMRF,1.e-3/(NT),1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
	ierr = KSPSetFromOptions(kspGMRF);CHKERRQ(ierr);
	ierr = VecCreate(PETSC_COMM_WORLD,&U);CHKERRQ(ierr);
	ierr = VecSetSizes(U,PETSC_DECIDE,NI);CHKERRQ(ierr);
	ierr = VecSetFromOptions(U);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&EUNm1);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&EUN);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&M2N);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&M2Nm1);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&dUNm1);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&dUN);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&VUN);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&VUNm1);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&b);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&resU);CHKERRQ(ierr);
	ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr); // Create matrix A residing in PETSC_COMM_WORLD
	ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,NI,NI);CHKERRQ(ierr); // Set the size of the matrix A, and let PETSC decide the decomposition
	ierr = MatSetFromOptions(A);CHKERRQ(ierr);
	ierr = MatMPIAIJSetPreallocation(A,5,NULL,5,NULL);CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(A,5,NULL);CHKERRQ(ierr);
	ierr = MatSetUp(A);CHKERRQ(ierr);			
	ierr = KSPCreate(PETSC_COMM_WORLD,&kspSPDE);CHKERRQ(ierr);
	ierr = KSPSetTolerances(kspSPDE,1.e-3/(NI),1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
	ierr = KSPSetFromOptions(kspSPDE);CHKERRQ(ierr);


	for (Ns = 1; (Ns <= Nsamples) && (tol > TOL); ++Ns){
		ierr = SetRandSource(N01,NT,dx,dy,Ns,generator);CHKERRQ(ierr);
		ierr = SetGMRFOperator(L,m,n,NGhost,dx,dy,kappa);CHKERRQ(ierr);
		ierr = KSPSetOperators(kspGMRF,L,L,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);			
		ierr = KSPSolve(kspGMRF,N01,rho);CHKERRQ(ierr);
		ierr = KSPGetIterationNumber(kspGMRF,&its);CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD,"Sample[%d]: GMRF solved in %d iterations \n",Ns,its);CHKERRQ(ierr);
		ierr = VecExp(rho);CHKERRQ(ierr);
		ierr = SetOperator(A,rho,m,n,NGhost,dx,dy);CHKERRQ(ierr);
		ierr = SetSource(b,rho,m,n,NGhost,dx,dy,UN,US,UE,UW,lamb);CHKERRQ(ierr);
		ierr = KSPSetOperators(kspSPDE,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
		ierr = KSPSolve(kspSPDE,b,U);CHKERRQ(ierr);
		ierr = KSPGetIterationNumber(kspSPDE,&its);CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD,"Sample[%d]: SPDE solved in %d iterations \n",Ns,its);CHKERRQ(ierr);
		
		// Start calculations for EUN
		
		ierr = VecAXPBYPCZ(EUN,(PetscScalar)(Ns-1)/(PetscScalar)Ns,1.0/(PetscScalar)Ns,0,EUNm1,U);CHKERRQ(ierr); // Calculate Expectation
		ierr = VecNorm(EUN,NORM_INFINITY,&normE);CHKERRQ(ierr);	
		ierr = VecWAXPY(dUNm1,-1,EUNm1,U);
		ierr = VecWAXPY(dUN,-1,EUN,U);
		ierr = VecPointwiseMult(M2N,dUN,dUNm1);CHKERRQ(ierr);
		ierr = VecAXPY(M2N,1,M2Nm1);
		ierr = VecAXPBY(VUN,1.0/(PetscScalar)(Ns),0,M2N);

		ierr = VecWAXPY(resU,-1,EUN,EUNm1);CHKERRQ(ierr);
		ierr = VecAbs(resU);CHKERRQ(ierr);
		ierr = VecNorm(resU,NORM_INFINITY,&tolE);CHKERRQ(ierr);
		ierr = VecWAXPY(resU,-1,VUN,VUNm1);CHKERRQ(ierr);
		ierr = VecAbs(resU);CHKERRQ(ierr);
		ierr = VecNorm(resU,NORM_INFINITY,&tolV);CHKERRQ(ierr);
		ierr = VecSum(VUN,&normV);CHKERRQ(ierr);
		tol = PetscMax(tolV,tolE);
		ierr = PetscPrintf(PETSC_COMM_WORLD,"Sample[%d]: Tol = %f, ||EU|| = %f, ||VU|| = %f \n",Ns,tol,normE,normV/NI);CHKERRQ(ierr);
		ierr = VecCopy(EUN,EUNm1); CHKERRQ(ierr);
		ierr = VecCopy(VUN,VUNm1); CHKERRQ(ierr);
		ierr = VecCopy(M2N,M2Nm1); CHKERRQ(ierr);
		
		// Start calculations for ErhoN
		ierr = VecAXPBYPCZ(ErhoN,(PetscScalar)(Ns-1)/(PetscScalar)Ns,1.0/(PetscScalar)Ns,0,ErhoNm1,N01);CHKERRQ(ierr); // Calculate Expectation
		ierr = VecNorm(ErhoN,NORM_INFINITY,&normE);CHKERRQ(ierr);	
		ierr = VecWAXPY(drhoNm1,-1,ErhoNm1,N01);
		ierr = VecWAXPY(drhoN,-1,ErhoN,N01);
		ierr = VecPointwiseMult(M2Nr,drhoN,drhoNm1);CHKERRQ(ierr);
		ierr = VecAXPY(M2Nr,1,M2Nm1r);
		ierr = VecAXPBY(VrhoN,1.0/(PetscScalar)(Ns),0,M2Nr);
		
		ierr = VecWAXPY(resR,-1,ErhoN,ErhoNm1);CHKERRQ(ierr);
		ierr = VecAbs(resR);CHKERRQ(ierr);
		ierr = VecNorm(resR,NORM_INFINITY,&tolE);CHKERRQ(ierr);
		ierr = VecWAXPY(resR,-1,VrhoN,VrhoNm1);CHKERRQ(ierr);
		ierr = VecAbs(resR);CHKERRQ(ierr);
		ierr = VecNorm(resR,NORM_INFINITY,&tolV);CHKERRQ(ierr);
		ierr = VecNorm(VrhoN,NORM_INFINITY,&normV);CHKERRQ(ierr);
//		ierr = VecSum(VrhoN,&normV);CHKERRQ(ierr);
		tol = PetscMax(tolV,PetscMax(tolE,tol));
		ierr = PetscPrintf(PETSC_COMM_WORLD,"Sample[%d]: Tol = %f, ||Erho|| = %f, ||Vrho|| = %f \n",Ns,tol,normE,normV);CHKERRQ(ierr);
		ierr = VecCopy(ErhoN,ErhoNm1); CHKERRQ(ierr);
		ierr = VecCopy(VrhoN,VrhoNm1); CHKERRQ(ierr);		
		
	}

	flg  = PETSC_FALSE;
	ierr = PetscOptionsGetBool(NULL,"-print_GMRF",&flg,NULL);CHKERRQ(ierr);
	if (flg) {ierr = PostProcs(rho,"rho_mean.dat");CHKERRQ(ierr);}	

	flg  = PETSC_FALSE;
	ierr = PetscOptionsGetBool(NULL,"-print_sol",&flg,NULL);CHKERRQ(ierr);
	if (flg) {ierr = PostProcs(U,"sol_mean.dat");CHKERRQ(ierr);}
	ierr = KSPDestroy(&kspSPDE);CHKERRQ(ierr);
	ierr = KSPDestroy(&kspGMRF);CHKERRQ(ierr);

	ierr = VecDestroy(&U);CHKERRQ(ierr);
	ierr = VecDestroy(&M2N);CHKERRQ(ierr);
	ierr = VecDestroy(&M2Nm1);CHKERRQ(ierr);
	ierr = VecDestroy(&dUN);CHKERRQ(ierr);
	ierr = VecDestroy(&dUNm1);CHKERRQ(ierr);
	ierr = VecDestroy(&EUN);CHKERRQ(ierr);
	ierr = VecDestroy(&EUNm1);CHKERRQ(ierr);
	ierr = VecDestroy(&VUN);CHKERRQ(ierr);
	ierr = VecDestroy(&VUNm1);CHKERRQ(ierr);
	ierr = VecDestroy(&b);CHKERRQ(ierr);
	ierr = VecDestroy(&resU);CHKERRQ(ierr);
	
	ierr = VecDestroy(&rho);CHKERRQ(ierr);
	ierr = VecDestroy(&M2Nr);CHKERRQ(ierr);
	ierr = VecDestroy(&M2Nm1r);CHKERRQ(ierr);
	ierr = VecDestroy(&drhoN);CHKERRQ(ierr);
	ierr = VecDestroy(&drhoNm1);CHKERRQ(ierr);
	ierr = VecDestroy(&ErhoNm1);CHKERRQ(ierr);
	ierr = VecDestroy(&ErhoN);CHKERRQ(ierr);
	ierr = VecDestroy(&VrhoN);CHKERRQ(ierr);	
	ierr = VecDestroy(&VrhoNm1);CHKERRQ(ierr);		
	ierr = VecDestroy(&N01);CHKERRQ(ierr);
	ierr = VecDestroy(&resR);CHKERRQ(ierr);

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
