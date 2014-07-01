/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Singular value decomposition of the Lauchli matrix.\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = matrix dimension.\n"
  "  -mu <mu>, where <mu> = subdiagonal value.\n\n";

#include <slepc.h>
#include <Functions.hh>
#include <vector>
#include <new>

#define nint(a) ((a) >= 0.0 ? (PetscInt)((a)+0.5) : (PetscInt)((a)-0.5))

PetscErrorCode SVD_Decomp(Mat& U, Mat& V, Mat& S, const Mat& A){
	PetscErrorCode ierr;
	Vec            u,v;             /* left and right singular vectors */
  	SVD            svd;             /* singular value problem solver context */
  	PetscScalar    sigma;
  PetscInt       nconv,Am,An;
  PetscInt*      IdxU, IdxV;
  PetscScalar*   utemp, vtemp;
  	ierr = MatGetVecs(A,&v,&u);CHKERRQ(ierr);
  	ierr = VecGetSize(v,&An); CHKERRQ(ierr);
  	ierr = VecGetSize(u,&Am); CHKERRQ(ierr);
  ierr = SVDCreate(PETSC_COMM_WORLD,&svd);CHKERRQ(ierr);
  ierr = SVDSetOperator(svd,A);CHKERRQ(ierr);
  ierr = SVDSetType(svd,SVDTRLANCZOS);CHKERRQ(ierr);
  ierr = SVDSetFromOptions(svd);CHKERRQ(ierr);
  ierr = SVDSolve(svd);CHKERRQ(ierr);  	
  ierr = SVDGetConverged(svd,&nconv);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&S);CHKERRQ(ierr); // Create matrix S residing in PETSC_COMM_WORLD
  ierr = MatSetSizes(S,PETSC_DECIDE,PETSC_DECIDE,nconv,nconv);CHKERRQ(ierr); // Set the size of the matrix S, and let PETSC decide the decomposition
  ierr = MatSetFromOptions(S);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(S,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(S,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(S);CHKERRQ(ierr);	
  ierr = MatCreate(PETSC_COMM_WORLD,&U);CHKERRQ(ierr); // Create matrix U residing in PETSC_COMM_WORLD
  ierr = MatSetSizes(U,PETSC_DECIDE,PETSC_DECIDE,Am,nconv);CHKERRQ(ierr); // Set the size of the matrix U, and let PETSC decide the decomposition
  ierr = MatSetFromOptions(U);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&V);CHKERRQ(ierr); // Create matrix V residing in PETSC_COMM_WORLD
  ierr = MatSetSizes(V,PETSC_DECIDE,PETSC_DECIDE,An,nconv);CHKERRQ(ierr); // Set the size of the matrix V, and let PETSC decide the decomposition
  ierr = MatSetFromOptions(V);CHKERRQ(ierr);
  IdxU = std::new PetscInt [Am];
  IdxV = std::new PetscInt [An];
  for (int ii = 0; ii < Am; ++ii) IdxU[ii] = ii;
  for (int ii = 0; ii < An; ++ii) IdxV[ii] = ii;
  for (int i=0;i<nconv;i++) {
    ierr = SVDGetSingularTriplet(svd,i,&sigma,u,v);CHKERRQ(ierr);
    ierr = MatSetValue(S,i,i,sigma,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecGetArray(u,&utemp); CHKERRQ(ierr);
    ierr = VecGetArray(v,&vtemp); CHKERRQ(ierr);
    ierr = MatSetValues(U,Am,IdxU,1,&i,&utemp,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatSetValues(V,An,IdxV,1,&i,&vtemp,INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecRestoreArray(u,&utemp); CHKERRQ(ierr);
    ierr = VecRestoreArray(v,&vtemp); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(U,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(V,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = SVDDestroy(&svd);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(U,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(V,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(S,MAT_FINAL_ASSEMBLY);
  return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A, U, S, V, Vt;               /* operator matrix */
  //UserCTX        users;
  //std::vector<PetscScalar>		XR,YR;
  PetscInt       i,j,Istart,Iend,n;
  PetscErrorCode ierr;
  PetscScalar    mu;
  SlepcInitialize(&argc,&argv,(char*)0,help);
  //ierr = GetOptions(users);CHKERRQ(ierr);
  //set_coordinates(XR,YR,users);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          Build the Covariance matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-mu",&mu,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nLauchli singular value decomposition, (%D x %D) mu=%G\n\n",n+1,n,mu);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n+1,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    if (i == 0) {
      for (j=0;j<n;j++) {
        ierr = MatSetValue(A,0,j,1.0,INSERT_VALUES);CHKERRQ(ierr);
      }
    } else {
      ierr = MatSetValue(A,i,i-1,mu,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  ierr = SVD_Decomp(U,V,S,A);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&U);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);
  ierr = MatDestroy(&V);CHKERRQ(ierr);
  ierr = MatDestroy(&Vt);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return 0;
}
