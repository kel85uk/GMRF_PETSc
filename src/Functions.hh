#ifndef FUNCTION_S
#define FUNCTION_S

#ifdef SLEPC
#include <slepc.h>
#endif
#include <petsc.h>
#ifdef MPE_log
#include <mpe.h>
#endif
#include <fstream>
#include <vector>
#include <new>
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>


typedef struct{
	PetscReal 	x0 = 0, x1 = 1, y0 = 0, y1 = 1, dx, dy;     /* norm of solution error */
	PetscReal		UN = 1, US = 10, UE = 5, UW = 3;
	PetscReal		dim = 2, alpha = 2, sigma = 0.3, lamb = 0.1, nu, kappa, tau2, tol = 1, TOL = 1e-4, tolU, tolr;
	PetscInt		m = 8, n = 7, its, NGhost = 2, NI = 0, NT = 0, Nsamples = 100, Ns = 1;
	int procpercolor = PROCS;
} UserCTX;

#define nint(a) ((a) >= 0.0 ? (PetscInt)((a)+0.5) : (PetscInt)((a)-0.5))

PetscErrorCode GetOptions(UserCTX&);
PetscErrorCode CreateVectors(Vec*&,const PetscInt&,const PetscInt&);
PetscErrorCode CreateVectors(Vec*&,const PetscInt&,const PetscInt&,const MPI_Comm&);
PetscErrorCode DestroyVectors(Vec*&,const PetscInt&);
PetscErrorCode CreateSolvers(Mat&,const PetscInt&,KSP&,Mat&,const PetscInt&,KSP&);
PetscErrorCode CreateSolvers(Mat&,const PetscInt&,KSP&,Mat&,const PetscInt&,KSP&,const MPI_Comm&);

PetscErrorCode Interp2D(Vec&,const Vec&);

PetscErrorCode SetGMRFOperator(Mat&, const PetscInt&, const PetscInt&, const PetscInt&, const PetscReal&, const PetscReal&, const PetscReal&);
#ifdef MPE_log
PetscErrorCode SetGMRFOperatorT(Mat&, const PetscInt&, const PetscInt&, const PetscInt&, const PetscReal&, const PetscReal&, const PetscReal&,std::vector<PetscLogEvent>&,std::vector<int>&);
#endif

PetscErrorCode SetOperator(Mat&, const PetscInt&, const PetscInt&, const PetscInt&, const PetscReal&, const PetscReal&);
PetscErrorCode SetOperator(Mat&, const Vec&, const PetscInt&, const PetscInt&, const PetscInt&, const PetscReal&, const PetscReal&);
PetscErrorCode SetOperator(Mat&, const Vec&, const PetscInt&, const PetscInt&, const PetscInt&, const PetscReal&, const PetscReal&,const MPI_Comm&);
#ifdef MPE_log
PetscErrorCode SetOperatorT(Mat&, const Vec&, const PetscInt&, const PetscInt&, const PetscInt&, const PetscReal&, const PetscReal&,std::vector<PetscLogEvent>&,std::vector<int>&,const MPI_Comm&);
#endif

PetscErrorCode SetRandSource(Vec&,const PetscInt&, const PetscReal&, const PetscReal&, const PetscMPIInt&,std::default_random_engine&);

PetscErrorCode SetSource(Vec&,const PetscInt&,const PetscInt&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscBool&);
PetscErrorCode SetSource(Vec&,const Vec&,const PetscInt&,const PetscInt&,const PetscInt&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&);
PetscErrorCode SetSource(Vec&,const Vec&,const PetscInt&,const PetscInt&,const PetscInt&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const MPI_Comm&);
#ifdef MPE_log
PetscErrorCode SetSourceT(Vec&,const Vec&,const PetscInt&,const PetscInt&,const PetscInt&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,std::vector<PetscLogEvent>&,std::vector<int>&,const MPI_Comm&);
#endif

void VecPostProcs(const std::vector<PetscScalar>&, const char*, const PetscMPIInt&);
PetscErrorCode VecPostProcs(const Vec&, const char*);
PetscErrorCode VecPostProcs(const Vec&, const char*,const PetscMPIInt&);

PetscErrorCode MatPostProcs(const Mat&, const char*);

PetscErrorCode update_stats(Vec&,Vec&,Vec&,Vec&,PetscReal&,const Vec&,const PetscInt&);
PetscErrorCode update_stats(Vec&,Vec&,Vec&,Vec&,const Vec&,const PetscInt&);
PetscErrorCode update_stats(PetscScalar&,PetscScalar&,PetscScalar&,PetscScalar&,PetscScalar&,const PetscScalar&,const PetscInt&);

PetscErrorCode VecSetMean(Vec&,const Vec&);

/** Function which generates the SVD decomposition into orthogonal basis vector matrices U, and V, and diagonal matrix of singular values (Requires Slepc) **/
PetscErrorCode SVD_Decomp(Mat&,Mat&,Mat&,const Mat&);

void recolor(MPI_Comm&,int&,const int&,const int&,const int&);

void global_local_Nelements(PetscInt&, PetscInt&, PetscInt&, const PetscInt&, const PetscInt&, const PetscInt&, const PetscInt&);

#endif
