#ifndef SOLVE_S
#define SOLVE_S
#include <Functions.hh>

PetscErrorCode UnitSolver(Vec*&,Vec*&,Mat*&,KSP*&,UserCTX&,std::default_random_engine&,const PetscMPIInt&,const PetscInt&);

#endif
