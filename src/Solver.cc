#include <Solver.hh>

PetscErrorCode UnitSolver(Vec*& WrapVecA, Vec*& WrapVecB, Mat*& WrapMats, KSP*& WrapKSPs, UserCTX& users, std::default_random_engine& generator, const PetscMPIInt& rank, const PetscInt& Ns){
	PetscErrorCode ierr;
	ierr = SetRandSource(WrapVecA[4],users.NT,users.dx,users.dy,rank,generator);CHKERRQ(ierr);	
	ierr = KSPSolve(WrapKSPs[0],WrapVecA[4],WrapVecA[0]);CHKERRQ(ierr);
	ierr = KSPGetIterationNumber(WrapKSPs[0],&users.its);CHKERRQ(ierr);
	ierr = VecScale(WrapVecA[0],1.0/sqrt(users.tau2));CHKERRQ(ierr);
	ierr = VecCopy(WrapVecA[0],WrapVecA[7]);CHKERRQ(ierr);
//	ierr = PetscPrintf(PETSC_COMM_WORLD,"Sample[%d]: GMRF solved in %d iterations \n",Ns,users.its);CHKERRQ(ierr);
	ierr = VecExp(WrapVecA[0]);CHKERRQ(ierr);
	ierr = SetOperator(WrapMats[1],WrapVecA[0],users.m,users.n,users.NGhost,users.dx,users.dy);CHKERRQ(ierr);
	ierr = SetSource(WrapVecB[4],WrapVecA[0],users.m,users.n,users.NGhost,users.dx,users.dy,users.UN,users.US,users.UE,users.UW,users.lamb);CHKERRQ(ierr);
	ierr = KSPSetOperators(WrapKSPs[1],WrapMats[1],WrapMats[1],DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
	ierr = KSPSolve(WrapKSPs[1],WrapVecB[4],WrapVecB[0]);CHKERRQ(ierr);
	ierr = KSPGetIterationNumber(WrapKSPs[1],&users.its);CHKERRQ(ierr);
/*	ierr = VecNorm(WrapVecB[0],NORM_2,&normU);CHKERRQ(ierr);
	normU /= users.NI;
//	ierr = PetscPrintf(PETSC_COMM_WORLD,"Sample[%d]: SPDE solved in %d iterations \n",Ns,users.its);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Sample[%d]: 2-Norm = %f \n",normU); */
	return ierr;
}

PetscErrorCode UnitSolver(Vec& rho, Vec& gmrf, Vec& N01, KSP& kspGMRF, Vec& U, Vec& b, Mat& A, KSP& kspSPDE, UserCTX& users, std::default_random_engine& generator, const PetscMPIInt& rank, const PetscInt& Ns){
	PetscErrorCode ierr;
	ierr = SetRandSource(N01,users.NT,users.dx,users.dy,rank,generator);CHKERRQ(ierr);	
	ierr = KSPSolve(kspGMRF,N01,rho);CHKERRQ(ierr);
	ierr = KSPGetIterationNumber(kspGMRF,&users.its);CHKERRQ(ierr);
	ierr = VecScale(rho,1.0/sqrt(users.tau2));CHKERRQ(ierr);
	ierr = VecCopy(rho,gmrf);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Sample[%d]: GMRF solved in %d iterations \n",Ns,users.its);CHKERRQ(ierr);
	ierr = VecExp(rho);CHKERRQ(ierr);
	ierr = SetOperator(A,rho,users.m,users.n,users.NGhost,users.dx,users.dy);CHKERRQ(ierr);
	ierr = SetSource(b,rho,users.m,users.n,users.NGhost,users.dx,users.dy,users.UN,users.US,users.UE,users.UW,users.lamb);CHKERRQ(ierr);
	ierr = KSPSetOperators(kspSPDE,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
	ierr = KSPSolve(kspSPDE,b,U);CHKERRQ(ierr);
	ierr = KSPGetIterationNumber(kspSPDE,&users.its);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Sample[%d]: SPDE solved in %d iterations \n",Ns,users.its);
	return ierr;
}

PetscErrorCode UnitSolver(Vec& rho, Vec& gmrf, Vec& N01, KSP& kspGMRF, Vec& U, Vec& b, Mat& A, KSP& kspSPDE, UserCTX& users, std::default_random_engine& generator, const PetscMPIInt& rank, const PetscInt& Ns, PetscScalar& normU){
	PetscErrorCode ierr;
	ierr = SetRandSource(N01,users.NT,users.dx,users.dy,rank,generator);CHKERRQ(ierr);	
	ierr = KSPSolve(kspGMRF,N01,rho);CHKERRQ(ierr);
	ierr = KSPGetIterationNumber(kspGMRF,&users.its);CHKERRQ(ierr);
	ierr = VecScale(rho,1.0/sqrt(users.tau2));CHKERRQ(ierr);
	ierr = VecCopy(rho,gmrf);CHKERRQ(ierr);
	ierr = VecExp(rho);CHKERRQ(ierr);
	ierr = SetOperator(A,rho,users.m,users.n,users.NGhost,users.dx,users.dy);CHKERRQ(ierr);
	ierr = SetSource(b,rho,users.m,users.n,users.NGhost,users.dx,users.dy,users.UN,users.US,users.UE,users.UW,users.lamb);CHKERRQ(ierr);
	ierr = KSPSetOperators(kspSPDE,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
	ierr = KSPSolve(kspSPDE,b,U);CHKERRQ(ierr);
	ierr = KSPGetIterationNumber(kspSPDE,&users.its);CHKERRQ(ierr);
	ierr = VecNorm(U,NORM_2,&normU);CHKERRQ(ierr);
	normU /= sqrt(users.NI);
//	ierr = PetscPrintf(PETSC_COMM_WORLD,"Sample[%d] from Processor %d: 2-Norm = %f \n",Ns,rank,normU);
	return ierr;
}

PetscErrorCode UnitSolverChol(Vec& rho, Vec& normrnds,const Mat& Chol_fac, Vec& U, Vec& b, Mat& A, KSP& kspSPDE, UserCTX& users, std::default_random_engine& generator, const PetscMPIInt& rank, const PetscInt& Ns, PetscScalar& normU){
	PetscErrorCode ierr;
	PetscScalar x;
	PetscInt Ii = 0, Istart = 0, Iend = 0;
	std::normal_distribution<PetscScalar> distribution(0.0,1.0);
	ierr = VecGetOwnershipRange(normrnds,&Istart,&Iend);CHKERRQ(ierr);
	for (Ii = Istart; Ii < Iend; ++Ii){
	  x = distribution(generator);
	  ierr = VecSetValues(normrnds,1,&Ii,&x,INSERT_VALUES);CHKERRQ(ierr);
	}
	ierr = VecAssemblyBegin(normrnds);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(normrnds);CHKERRQ(ierr);
	ierr = MatMult(Chol_fac,normrnds,rho);CHKERRQ(ierr); // rho = Chol_fac*normrnds => y = Lx
	ierr = VecExp(rho);CHKERRQ(ierr);
	ierr = SetOperator(A,rho,users.m,users.n,users.NGhost,users.dx,users.dy);CHKERRQ(ierr);
	ierr = SetSource(b,rho,users.m,users.n,users.NGhost,users.dx,users.dy,users.UN,users.US,users.UE,users.UW,users.lamb);CHKERRQ(ierr);
	ierr = KSPSetOperators(kspSPDE,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
	ierr = KSPSolve(kspSPDE,b,U);CHKERRQ(ierr);
	ierr = KSPGetIterationNumber(kspSPDE,&users.its);CHKERRQ(ierr);
	ierr = VecNorm(U,NORM_2,&normU);CHKERRQ(ierr);
	normU /= sqrt(users.NI);
//	ierr = PetscPrintf(PETSC_COMM_WORLD,"Sample[%d] from Processor %d: 2-Norm = %f \n",Ns,rank,normU);
	return ierr;
}
