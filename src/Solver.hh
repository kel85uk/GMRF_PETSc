#ifndef SOLVE_S
#define SOLVE_S
#include <Functions.hh>

PetscErrorCode UnitSolver(Vec*&,Vec*&,Mat*&,KSP*&,UserCTX&,std::default_random_engine&,const PetscMPIInt&,const PetscInt&);

PetscErrorCode UnitSolver(Vec&,Vec&,Vec&,KSP&,Vec&,Vec&,Mat&,KSP&,UserCTX&,std::default_random_engine&,const PetscMPIInt&,const PetscInt&,PetscScalar&);

PetscErrorCode UnitSolver(Vec&,Vec&,Vec&,KSP&,Vec&,Vec&,Mat&,KSP&,UserCTX&,std::default_random_engine&,const PetscMPIInt&,const PetscInt&,PetscScalar&,const MPI_Comm&);

#ifdef MPE_log
PetscErrorCode UnitSolverTimings(Vec&,Vec&,Vec&,KSP&,Vec&,Vec&,Mat&,KSP&,UserCTX&,std::default_random_engine&,const PetscMPIInt&,const PetscInt&,PetscScalar&,std::vector<PetscLogEvent>&,std::vector<int>&,const MPI_Comm&);
#endif
PetscErrorCode UnitSolver(Vec&,Vec&,Vec&,KSP&,Vec&,Vec&,Mat&,KSP&,UserCTX&,std::default_random_engine&,const PetscMPIInt&,const PetscInt&);

PetscErrorCode UnitSolverChol(Vec&,Vec&,const Mat&,Vec&,Vec&,Mat&,KSP&,UserCTX&,std::default_random_engine&,const PetscMPIInt&,const PetscInt&,PetscScalar&);

PetscErrorCode UnitSolverChol(Vec&,Vec&,const PC&,const Mat&,Vec&,Vec&,Mat&,KSP&,UserCTX&,std::default_random_engine&,const PetscMPIInt&,const PetscInt&,PetscScalar&);

#endif
