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

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A, U, S, V, Vt, result, resultf;               /* operator matrix */
  //UserCTX        users;
  //std::vector<PetscScalar>		XR,YR;
  PetscInt       i,j,Istart,Iend,n;
  PetscErrorCode ierr;
  PetscScalar    mu = 0.1;
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
  PetscPrintf(PETSC_COMM_WORLD,"All good\n");
  ierr = SVD_Decomp(U,V,S,A);CHKERRQ(ierr);
  MatView(S,PETSC_VIEWER_STDOUT_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,"All good 2\n");
  MatView(U,PETSC_VIEWER_STDOUT_WORLD);
  MatView(V,PETSC_VIEWER_STDOUT_WORLD);
  MatTranspose(V,MAT_INITIAL_MATRIX,&Vt);
  MatMatMult(S,Vt,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&result);
  MatMatMult(U,result,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&resultf);
  MatView(resultf,PETSC_VIEWER_STDOUT_WORLD);
  MatView(A,PETSC_VIEWER_STDOUT_WORLD);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&U);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);
  ierr = MatDestroy(&V);CHKERRQ(ierr);
  ierr = MatDestroy(&Vt);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return 0;
}
