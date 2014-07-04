#ifndef SOLVE_S
#define SOLVE_S
#include <Functions.hh>

PetscErrorCode UnitSolver(Vec*&,Vec*&,Mat*&,KSP*&,UserCTX&,std::default_random_engine&,const PetscMPIInt&,const PetscInt&);

PetscErrorCode UnitSolver(Vec&,Vec&,Vec&,KSP&,Vec&,Vec&,Mat&,KSP&,UserCTX&,std::default_random_engine&,const PetscMPIInt&,const PetscInt&,PetscScalar&);

PetscErrorCode UnitSolverTimings(Vec&,Vec&,Vec&,KSP&,Vec&,Vec&,Mat&,KSP&,UserCTX&,std::default_random_engine&,const PetscMPIInt&,const PetscInt&,PetscScalar&,PetscScalar*&);

PetscErrorCode UnitSolver(Vec&,Vec&,Vec&,KSP&,Vec&,Vec&,Mat&,KSP&,UserCTX&,std::default_random_engine&,const PetscMPIInt&,const PetscInt&);

PetscErrorCode UnitSolverChol(Vec&,Vec&,const Mat&,Vec&,Vec&,Mat&,KSP&,UserCTX&,std::default_random_engine&,const PetscMPIInt&,const PetscInt&,PetscScalar&);

PetscErrorCode UnitSolverChol(Vec&,Vec&,const PC&,const Mat&,Vec&,Vec&,Mat&,KSP&,UserCTX&,std::default_random_engine&,const PetscMPIInt&,const PetscInt&,PetscScalar&);

#endif
