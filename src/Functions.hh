#ifndef FUNCTION_S
#define FUNCTION_S

#include <petsc.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>


typedef struct{
	PetscReal 	x0 = 0, x1 = 1, y0 = 0, y1 = 1, dx, dy;     /* norm of solution error */
	PetscReal		UN = 1, US = 10, UE = 5, UW = 3;
	PetscReal		dim = 2, alpha = 2, sigma = 0.3, lamb = 0.1, nu, kappa, tau2, tol = 1, TOL = 1e-4, tolU, tolr;
	PetscInt		m = 8, n = 7, its, NGhost = 2, NI = 0, NT = 0, Nsamples = 100, Ns = 1;
} UserCTX;

#define nint(a) ((a) >= 0.0 ? (PetscInt)((a)+0.5) : (PetscInt)((a)-0.5))

PetscErrorCode GetOptions(UserCTX&);
PetscErrorCode CreateVectors(Vec*&,const PetscInt&,const PetscInt&);
PetscErrorCode DestroyVectors(Vec*&,const PetscInt&);
PetscErrorCode CreateSolvers(Mat&,const PetscInt&,KSP&,Mat&,const PetscInt&,KSP&);


PetscErrorCode SetGMRFOperator(Mat&, const PetscInt&, const PetscInt&, const PetscInt&, const PetscReal&, const PetscReal&, const PetscReal&);

PetscErrorCode SetOperator(Mat&, const PetscInt&, const PetscInt&, const PetscInt&, const PetscReal&, const PetscReal&);
PetscErrorCode SetOperator(Mat&, const Vec&, const PetscInt&, const PetscInt&, const PetscInt&, const PetscReal&, const PetscReal&);

PetscErrorCode SetRandSource(Vec&,const PetscInt&, const PetscReal&, const PetscReal&, const PetscMPIInt&,std::default_random_engine&);

PetscErrorCode SetSource(Vec&,const PetscInt&,const PetscInt&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscBool&);
PetscErrorCode SetSource(Vec&,const Vec&,const PetscInt&,const PetscInt&,const PetscInt&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&);

void VecPostProcs(const std::vector<PetscScalar>&, const char*, const PetscMPIInt&);
PetscErrorCode VecPostProcs(const Vec&, const char*);
PetscErrorCode VecPostProcs(const Vec&, const char*,const PetscMPIInt&);

PetscErrorCode MatPostProcs(const Mat&, const char*);

PetscErrorCode update_stats(Vec&,Vec&,Vec&,Vec&,PetscReal&,const Vec&,const PetscInt&);
PetscErrorCode update_stats(Vec&,Vec&,Vec&,Vec&,const Vec&,const PetscInt&);
PetscErrorCode update_stats(PetscScalar&,PetscScalar&,PetscScalar&,PetscScalar&,PetscScalar&,const PetscScalar&,const PetscInt&);

PetscErrorCode VecSetMean(Vec&,const Vec&);

void global_local_Nelements(PetscInt&, PetscInt&, PetscInt&, const PetscInt&, const PetscInt&, const PetscInt&, const PetscInt&);

#endif
