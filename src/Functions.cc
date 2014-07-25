#include <Functions.hh>

PetscErrorCode CreateVectors(Vec*& Wrapall,const PetscInt& N,const PetscInt& veclen){
	PetscErrorCode ierr;
	ierr = VecCreate(PETSC_COMM_WORLD,&Wrapall[0]);CHKERRQ(ierr);
	ierr = VecSetSizes(Wrapall[0],PETSC_DECIDE,veclen);CHKERRQ(ierr);
	ierr = VecSetFromOptions(Wrapall[0]);CHKERRQ(ierr);
	for (int NN = 1; NN < N; ++NN)
		ierr = VecDuplicate(Wrapall[0],&Wrapall[NN]);CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode CreateVectors(Vec*& Wrapall,const PetscInt& N,const PetscInt& veclen, const MPI_Comm& petsc_comm){
	PetscErrorCode ierr;
	ierr = VecCreate(petsc_comm,&Wrapall[0]);CHKERRQ(ierr);
	ierr = VecSetSizes(Wrapall[0],PETSC_DECIDE,veclen);CHKERRQ(ierr);
	ierr = VecSetFromOptions(Wrapall[0]);CHKERRQ(ierr);
	for (int NN = 1; NN < N; ++NN)
		ierr = VecDuplicate(Wrapall[0],&Wrapall[NN]);CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode DestroyVectors(Vec*& Wrapall,const PetscInt& N){
	PetscErrorCode ierr;
	for (int NN = 0; NN < N; ++NN)
		ierr = VecDestroy(&Wrapall[NN]);CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode CreateSolvers(Mat& L, const PetscInt& NT, KSP& kspGMRF, Mat& A,const PetscInt& NI, KSP& kspSPDE){
	PetscErrorCode ierr;
	ierr = MatCreate(PETSC_COMM_WORLD,&L);CHKERRQ(ierr); // Create matrix A residing in PETSC_COMM_WORLD
	ierr = MatSetSizes(L,PETSC_DECIDE,PETSC_DECIDE,NT,NT);CHKERRQ(ierr); // Set the size of the matrix A, and let PETSC decide the decomposition
	ierr = MatSetFromOptions(L);CHKERRQ(ierr);
	ierr = MatMPIAIJSetPreallocation(L,5,NULL,5,NULL);CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(L,5,NULL);CHKERRQ(ierr);
	ierr = MatSetUp(L);CHKERRQ(ierr);	
	ierr = KSPCreate(PETSC_COMM_WORLD,&kspGMRF);CHKERRQ(ierr);
	ierr = KSPSetTolerances(kspGMRF,1.e-7/(NT),1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
	ierr = KSPSetFromOptions(kspGMRF);CHKERRQ(ierr);

	ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr); // Create matrix A residing in PETSC_COMM_WORLD
	ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,NI,NI);CHKERRQ(ierr); // Set the size of the matrix A, and let PETSC decide the decomposition
	ierr = MatSetFromOptions(A);CHKERRQ(ierr);
	ierr = MatMPIAIJSetPreallocation(A,5,NULL,5,NULL);CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(A,5,NULL);CHKERRQ(ierr);
	ierr = MatSetUp(A);CHKERRQ(ierr);			
	ierr = KSPCreate(PETSC_COMM_WORLD,&kspSPDE);CHKERRQ(ierr);
	ierr = KSPSetTolerances(kspSPDE,1.e-7/(NI),1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
	ierr = KSPSetFromOptions(kspSPDE);CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode CreateSolvers(Mat& L, const PetscInt& NT, KSP& kspGMRF, Mat& A,const PetscInt& NI, KSP& kspSPDE,const MPI_Comm& petsc_comm){
	PetscErrorCode ierr;
	ierr = MatCreate(petsc_comm,&L);CHKERRQ(ierr); // Create matrix A residing in PETSC_COMM_WORLD
	ierr = MatSetSizes(L,PETSC_DECIDE,PETSC_DECIDE,NT,NT);CHKERRQ(ierr); // Set the size of the matrix A, and let PETSC decide the decomposition
	ierr = MatSetFromOptions(L);CHKERRQ(ierr);
	ierr = MatMPIAIJSetPreallocation(L,5,NULL,5,NULL);CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(L,5,NULL);CHKERRQ(ierr);
	ierr = MatSetUp(L);CHKERRQ(ierr);	
	ierr = KSPCreate(petsc_comm,&kspGMRF);CHKERRQ(ierr);
	ierr = KSPSetTolerances(kspGMRF,1.e-7/(NT),1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
	ierr = KSPSetFromOptions(kspGMRF);CHKERRQ(ierr);

	ierr = MatCreate(petsc_comm,&A);CHKERRQ(ierr); // Create matrix A residing in PETSC_COMM_WORLD
	ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,NI,NI);CHKERRQ(ierr); // Set the size of the matrix A, and let PETSC decide the decomposition
	ierr = MatSetFromOptions(A);CHKERRQ(ierr);
	ierr = MatMPIAIJSetPreallocation(A,5,NULL,5,NULL);CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(A,5,NULL);CHKERRQ(ierr);
	ierr = MatSetUp(A);CHKERRQ(ierr);			
	ierr = KSPCreate(petsc_comm,&kspSPDE);CHKERRQ(ierr);
	ierr = KSPSetTolerances(kspSPDE,1.e-7/(NI),1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
	ierr = KSPSetFromOptions(kspSPDE);CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode Interp2D(Vec& output,const Vec& input,const int& m_new,const int& m_old)
{
	PetscErrorCode ierr;
	PetscInt Nout, Nin;
	PetscScalar *Inputl_arr;
	ierr = VecGetSize(output,&Nout); CHKERRQ(ierr);
	ierr = VecGetSize(input,&Nin); CHKERRQ(ierr);
	PetscInt n_new = Nout/m_new;
	PetscInt n_old = Nin/m_old;
  if ( n_new == n_old  && m_new == m_old ){
    ierr = VecCopy(input,output); CHKERRQ(ierr);
  }
  VecScatter sc_ctx;
  Vec Input_loc;
  ierr = VecScatterCreateToAll(input,&sc_ctx,&Input_loc); CHKERRQ(ierr);
  VecScatterBegin(sc_ctx,input,Input_loc,INSERT_VALUES,SCATTER_FORWARD);
  PetscScalar scaleX = PetscScalar( m_old ) / PetscScalar(m_new);
  PetscScalar scaleY = PetscScalar( n_old ) / PetscScalar(n_new);
  PetscInt Ii, Istart, Iend, oind, oindp1, oindpm, oindp1m, x, y;
  VecScatterEnd(sc_ctx,input,Input_loc,INSERT_VALUES,SCATTER_FORWARD);
  VecGetArray(Input_loc,&Inputl_arr);
  
  ierr = VecGetOwnershipRange(output,&Istart,&Iend);CHKERRQ(ierr);
	for (Ii = Istart; Ii < Iend; ++Ii){
    y = (PetscInt) Ii/m_new; x = Ii - y*m_new;
    PetscScalar oldX = x * scaleX;
    PetscScalar oldY = y * scaleY;
    PetscInt oldXi = static_cast<PetscInt>( oldX );
    PetscInt oldYi = static_cast<PetscInt>( oldY );
    PetscScalar xDiff = oldX - oldXi;
    PetscScalar yDiff = oldY - oldYi;

    // bilinear interpolation
    oind   = oldXi + oldYi*m_old;
    oindp1 = (oldXi + 1) + oldYi*m_old;
    oindpm = (oldXi) + (oldYi + 1)*m_old;
    oindp1m = (oldXi + 1) + (oldYi + 1)*m_old;
    PetscScalar value =
            (1 - xDiff) * (1 - yDiff ) * Inputl_arr[oind] /*getElement( oldXi    , oldYi    )*/ +
                 xDiff  * (1 - yDiff ) * Inputl_arr[oindp1] /*getElement( oldXi + 1, oldYi    )*/ +
            (1 - xDiff) *      yDiff   * Inputl_arr[oindpm] /*getElement( oldXi    , oldYi + 1)*/ +
                 xDiff  *      yDiff   * Inputl_arr[oindp1m]; //getElement( oldXi + 1, oldYi + 1)
    ierr = VecSetValue(output,Ii,value,INSERT_VALUES); CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(output); CHKERRQ(ierr);
    ierr = VecDestroy(&Input_loc); CHKERRQ(ierr);
    ierr = VecRestoreArray(Input_loc,&Inputl_arr);CHKERRQ(ierr);	
		ierr = VecScatterDestroy(&sc_ctx);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(output);
  return ierr;
}

PetscErrorCode SetGMRFOperator(Mat& L, const PetscInt& m,const PetscInt& n,const PetscInt& NGhost, const PetscReal& dx,const PetscReal& dy, const PetscReal& kappa){
	PetscInt			i,j,Ii,J,Istart,Iend, M = (m + 2*NGhost), N = (n + 2*NGhost);
	PetscReal			dxdy = 1.0, dxidy = 1.0/dy/dy, dyidx = 1.0/dx/dx;
	PetscReal			vD, vN = -dxidy, vS = -dxidy, vE = -dyidx, vW = -dyidx;
	PetscErrorCode ierr;
	
	ierr = MatGetOwnershipRange(L,&Istart,&Iend);CHKERRQ(ierr);

	for (Ii=Istart; Ii<Iend; Ii++) {
		vD = 0.; j = (PetscInt) Ii/M; i = Ii - j*M;
		if (i>0)   {J = Ii - 1; ierr = MatSetValues(L,1,&Ii,1,&J,&vW,INSERT_VALUES);CHKERRQ(ierr);}
		if (i<M-1) {J = Ii + 1; ierr = MatSetValues(L,1,&Ii,1,&J,&vE,INSERT_VALUES);CHKERRQ(ierr);}
		if (j>0)   {J = Ii - M; ierr = MatSetValues(L,1,&Ii,1,&J,&vS,INSERT_VALUES);CHKERRQ(ierr);}
		if (j<N-1) {J = Ii + M; ierr = MatSetValues(L,1,&Ii,1,&J,&vN,INSERT_VALUES);CHKERRQ(ierr);}
		vD = -(vW + vE + vS + vN) + kappa*kappa;
		if (j == 0){
      			vD			+=		vS;
			}
      if (j == N-1){
				vD			+=		vN;
			}
      if (i == 0){
				vD			+=		vW;
			}
      if (i == M-1){
				vD			+=		vE;
			}		
		ierr = MatSetValues(L,1,&Ii,1,&Ii,&vD,INSERT_VALUES);CHKERRQ(ierr);
	}
	
	ierr = MatAssemblyBegin(L,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(L,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	return ierr;
}
#ifdef MPE_log
PetscErrorCode SetGMRFOperatorT(Mat& L, const PetscInt& m,const PetscInt& n,const PetscInt& NGhost, const PetscReal& dx,const PetscReal& dy, const PetscReal& kappa,std::vector<PetscLogEvent>& petscevents,std::vector<int>& MPE_events){
	PetscInt			i,j,Ii,J,Istart,Iend, M = (m + 2*NGhost), N = (n + 2*NGhost);
	PetscReal			dxdy = 1.0, dxidy = 1.0/dy/dy, dyidx = 1.0/dx/dx;
	PetscReal			vD, vN = -dxidy, vS = -dxidy, vE = -dyidx, vW = -dyidx;
	PetscErrorCode ierr;

	ierr = MatGetOwnershipRange(L,&Istart,&Iend);CHKERRQ(ierr);
  PetscLogEventBegin(petscevents[1],0,0,0,0);
  MPE_Log_event(MPE_events[2],0,"GMRF Setup-start");
	for (Ii=Istart; Ii<Iend; Ii++) {
		vD = 0.; j = (PetscInt) Ii/M; i = Ii - j*M;
		if (i>0)   {J = Ii - 1; ierr = MatSetValues(L,1,&Ii,1,&J,&vW,INSERT_VALUES);CHKERRQ(ierr);}
		if (i<M-1) {J = Ii + 1; ierr = MatSetValues(L,1,&Ii,1,&J,&vE,INSERT_VALUES);CHKERRQ(ierr);}
		if (j>0)   {J = Ii - M; ierr = MatSetValues(L,1,&Ii,1,&J,&vS,INSERT_VALUES);CHKERRQ(ierr);}
		if (j<N-1) {J = Ii + M; ierr = MatSetValues(L,1,&Ii,1,&J,&vN,INSERT_VALUES);CHKERRQ(ierr);}
		vD = -(vW + vE + vS + vN) + kappa*kappa;
		if (j == 0){
      			vD			+=		vS;
			}
      if (j == N-1){
				vD			+=		vN;
			}
      if (i == 0){
				vD			+=		vW;
			}
      if (i == M-1){
				vD			+=		vE;
			}		
		ierr = MatSetValues(L,1,&Ii,1,&Ii,&vD,INSERT_VALUES);CHKERRQ(ierr);
	}
	PetscLogEventEnd(petscevents[1],0,0,0,0);
	MPE_Log_event(MPE_events[3],0,"GMRF Setup-end");
	PetscLogEventBegin(petscevents[0],0,0,0,0);
	MPE_Log_event(MPE_events[0],0,"GMRF Comm-start");
	ierr = MatAssemblyBegin(L,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(L,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	PetscLogEventEnd(petscevents[0],0,0,0,0);
	MPE_Log_event(MPE_events[1],0,"GMRF Comm-start");
	return ierr;
}
#endif
PetscErrorCode SetOperator(Mat& A, const PetscInt& m,const PetscInt& n,const PetscInt& NGhost, const PetscReal& dx,const PetscReal& dy){
	PetscInt			i,j,Ii,J,Istart,Iend;
	PetscReal			idx2, idy2;
	PetscScalar		vN, vS, vE, vW, vD;
	PetscErrorCode ierr;
	
	ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);

	idx2 = 1.0/(dx*dx);
	idy2 = 1.0/(dy*dy);
	for (Ii=Istart; Ii<Iend; Ii++) {
		vD = 0.; j = (PetscInt) Ii/m; i = Ii - j*m;
		vW = 0.; vE = 0.; vS = 0.; vN = 0.;
		if (i>0)   {J = Ii - 1; vW = -1.0*idx2; ierr = MatSetValues(A,1,&Ii,1,&J,&vW,INSERT_VALUES);CHKERRQ(ierr);}
		if (i<m-1) {J = Ii + 1; vE = -1.0*idx2; ierr = MatSetValues(A,1,&Ii,1,&J,&vE,INSERT_VALUES);CHKERRQ(ierr);}
		if (j>0)   {J = Ii - m; vS = -1.0*idy2; ierr = MatSetValues(A,1,&Ii,1,&J,&vS,INSERT_VALUES);CHKERRQ(ierr);}
		if (j<n-1) {J = Ii + m; vN = -1.0*idy2; ierr = MatSetValues(A,1,&Ii,1,&J,&vN,INSERT_VALUES);CHKERRQ(ierr);}
		vW = 1.0*idx2;vE = 1.0*idx2;vS = 1.0*idy2;vN = 1.0*idy2;
		vD = vW + vE + vS + vN;
		if (j == 0){
      			vD			+=		vS;
			}
      if (j == n-1){
				vD			+=		vN;
			}
      if (i == 0){
				vD			+=		vW;
			}
      if (i == m-1){
				vD			+=		vE;
			}		
		ierr = MatSetValues(A,1,&Ii,1,&Ii,&vD,INSERT_VALUES);CHKERRQ(ierr);
	}
	
	ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode SetOperator(Mat& A, const Vec& rho, const PetscInt& m,const PetscInt& n,const PetscInt& NGhost, const PetscReal& dx,const PetscReal& dy){
	PetscInt			i,j,Ii,II,Ji,JJ,Istart,Iend, Nrhol,ILs,ILe;
	IS					local_indices, global_indices;
	VecScatter		rho_scatter_ctx;
	PetscReal			idx2, idy2;
	PetscScalar		vN, vS, vE, vW, vD, rhoII, rhoJJ, rhoN, rhoS, rhoE, rhoW;
	PetscScalar*	   rhol_arr;
	Vec					rhol_vec;
	PetscErrorCode ierr;
	
	ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
	
	global_local_Nelements(Nrhol,ILs,ILe,Istart,Iend,NGhost,m);
	ierr = VecCreate(PETSC_COMM_SELF,&rhol_vec);CHKERRQ(ierr);
	ierr = VecSetSizes(rhol_vec,PETSC_DECIDE,Nrhol);CHKERRQ(ierr);
	ierr = VecSetFromOptions(rhol_vec);CHKERRQ(ierr);	

	ierr = ISCreateStride(PETSC_COMM_WORLD,Nrhol,ILs,1,&global_indices);CHKERRQ(ierr); // Get the indices for global rho vector
	ierr = ISCreateStride(PETSC_COMM_SELF,Nrhol,0,1,&local_indices);CHKERRQ(ierr); // Indices for local rhol vector
	ierr = VecScatterCreate(rho,global_indices,rhol_vec,local_indices,&rho_scatter_ctx);
	// Copy elements from rho to rhol_vec
	ierr = VecScatterBegin(rho_scatter_ctx,rho,rhol_vec,INSERT_VALUES,SCATTER_FORWARD);
	ierr = VecScatterEnd(rho_scatter_ctx,rho,rhol_vec,INSERT_VALUES,SCATTER_FORWARD);
	ierr = VecGetArray(rhol_vec,&rhol_arr);CHKERRQ(ierr);
	
	idx2 = 1.0/(dx*dx);
	idy2 = 1.0/(dy*dy);
	for (Ii=Istart; Ii<Iend; Ii++) {
		vD = 0.; j = (PetscInt) Ii/m; i = Ii - j*m;
		II = (i + NGhost) + (m+2*NGhost)*(j + NGhost);
		rhoII = rhol_arr[II-ILs];
		vW = 0.; vE = 0.; vS = 0.; vN = 0.; rhoW = 0.; rhoE = 0.; rhoS = 0.; rhoN = 0.;
		JJ = II - 1;
		rhoJJ = rhol_arr[JJ-ILs];
		rhoW = .5*(rhoII + rhoJJ);
		if (i>0)   {
			Ji = Ii - 1;
			vW = -rhoW*idx2;
			ierr = MatSetValues(A,1,&Ii,1,&Ji,&vW,INSERT_VALUES);CHKERRQ(ierr);
		}
		JJ = II + 1;
		rhoJJ = rhol_arr[JJ-ILs];
		rhoE = .5*(rhoII + rhoJJ);
		if (i<m-1) {
			Ji = Ii + 1; 
			vE = -rhoE*idx2;
			ierr = MatSetValues(A,1,&Ii,1,&Ji,&vE,INSERT_VALUES);CHKERRQ(ierr);
		}
		JJ = II - (m + 2*NGhost);
		rhoJJ = rhol_arr[JJ-ILs];
		rhoS = .5*(rhoII + rhoJJ);
		if (j>0)   {
			Ji = Ii - m;
			vS = -rhoS*idy2;
			ierr = MatSetValues(A,1,&Ii,1,&Ji,&vS,INSERT_VALUES);CHKERRQ(ierr);
		}
		JJ = II + (m + 2*NGhost);
		rhoJJ = rhol_arr[JJ-ILs];
		rhoN = .5*(rhoII + rhoJJ);
		if (j<n-1) {
			Ji = Ii + m; 
			vN = -rhoN*idy2;
			ierr = MatSetValues(A,1,&Ii,1,&Ji,&vN,INSERT_VALUES);CHKERRQ(ierr);
		}
		vW = rhoW*idx2;
		vE = rhoE*idx2;
		vS = rhoS*idy2;
		vN = rhoN*idy2;
		vD = vW + vE + vS + vN;
		if (j == 0){
      			vD			+=		vS;
		}
      if (j == n-1){
				vD			+=		vN;
		}
      if (i == 0){
				vD			+=		vW;
		}
      if (i == m-1){
				vD			+=		vE;
		}		
		ierr = MatSetValues(A,1,&Ii,1,&Ii,&vD,INSERT_VALUES);CHKERRQ(ierr);
	}

	ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
		ierr = VecRestoreArray(rhol_vec,&rhol_arr);CHKERRQ(ierr);	
		ierr = VecDestroy(&rhol_vec);CHKERRQ(ierr);
		ierr = VecScatterDestroy(&rho_scatter_ctx);CHKERRQ(ierr);
		ierr = ISDestroy(&local_indices);CHKERRQ(ierr);
		ierr = ISDestroy(&global_indices);CHKERRQ(ierr);		
	ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode SetOperator(Mat& A, const Vec& rho, const PetscInt& m,const PetscInt& n,const PetscInt& NGhost, const PetscReal& dx,const PetscReal& dy,const MPI_Comm& petsc_comm){
	PetscInt			i,j,Ii,II,Ji,JJ,Istart,Iend, Nrhol,ILs,ILe;
	IS					local_indices, global_indices;
	VecScatter		rho_scatter_ctx;
	PetscReal			idx2, idy2;
	PetscScalar		vN, vS, vE, vW, vD, rhoII, rhoJJ, rhoN, rhoS, rhoE, rhoW;
	PetscScalar*	   rhol_arr;
	Vec					rhol_vec;
	PetscErrorCode ierr;
	
	ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
	
	global_local_Nelements(Nrhol,ILs,ILe,Istart,Iend,NGhost,m);
	ierr = VecCreate(PETSC_COMM_SELF,&rhol_vec);CHKERRQ(ierr);
	ierr = VecSetSizes(rhol_vec,PETSC_DECIDE,Nrhol);CHKERRQ(ierr);
	ierr = VecSetFromOptions(rhol_vec);CHKERRQ(ierr);	

	ierr = ISCreateStride(petsc_comm,Nrhol,ILs,1,&global_indices);CHKERRQ(ierr); // Get the indices for global rho vector
	ierr = ISCreateStride(PETSC_COMM_SELF,Nrhol,0,1,&local_indices);CHKERRQ(ierr); // Indices for local rhol vector
	ierr = VecScatterCreate(rho,global_indices,rhol_vec,local_indices,&rho_scatter_ctx);
	// Copy elements from rho to rhol_vec
	ierr = VecScatterBegin(rho_scatter_ctx,rho,rhol_vec,INSERT_VALUES,SCATTER_FORWARD);
	ierr = VecScatterEnd(rho_scatter_ctx,rho,rhol_vec,INSERT_VALUES,SCATTER_FORWARD);
	ierr = VecGetArray(rhol_vec,&rhol_arr);CHKERRQ(ierr);
	
	idx2 = 1.0/(dx*dx);
	idy2 = 1.0/(dy*dy);
	for (Ii=Istart; Ii<Iend; Ii++) {
		vD = 0.; j = (PetscInt) Ii/m; i = Ii - j*m;
		II = (i + NGhost) + (m+2*NGhost)*(j + NGhost);
		rhoII = rhol_arr[II-ILs];
		vW = 0.; vE = 0.; vS = 0.; vN = 0.; rhoW = 0.; rhoE = 0.; rhoS = 0.; rhoN = 0.;
		JJ = II - 1;
		rhoJJ = rhol_arr[JJ-ILs];
		rhoW = .5*(rhoII + rhoJJ);
		if (i>0)   {
			Ji = Ii - 1;
			vW = -rhoW*idx2;
			ierr = MatSetValues(A,1,&Ii,1,&Ji,&vW,INSERT_VALUES);CHKERRQ(ierr);
		}
		JJ = II + 1;
		rhoJJ = rhol_arr[JJ-ILs];
		rhoE = .5*(rhoII + rhoJJ);
		if (i<m-1) {
			Ji = Ii + 1; 
			vE = -rhoE*idx2;
			ierr = MatSetValues(A,1,&Ii,1,&Ji,&vE,INSERT_VALUES);CHKERRQ(ierr);
		}
		JJ = II - (m + 2*NGhost);
		rhoJJ = rhol_arr[JJ-ILs];
		rhoS = .5*(rhoII + rhoJJ);
		if (j>0)   {
			Ji = Ii - m;
			vS = -rhoS*idy2;
			ierr = MatSetValues(A,1,&Ii,1,&Ji,&vS,INSERT_VALUES);CHKERRQ(ierr);
		}
		JJ = II + (m + 2*NGhost);
		rhoJJ = rhol_arr[JJ-ILs];
		rhoN = .5*(rhoII + rhoJJ);
		if (j<n-1) {
			Ji = Ii + m; 
			vN = -rhoN*idy2;
			ierr = MatSetValues(A,1,&Ii,1,&Ji,&vN,INSERT_VALUES);CHKERRQ(ierr);
		}
		vW = rhoW*idx2;
		vE = rhoE*idx2;
		vS = rhoS*idy2;
		vN = rhoN*idy2;
		vD = vW + vE + vS + vN;
		if (j == 0){
      			vD			+=		vS;
		}
      if (j == n-1){
				vD			+=		vN;
		}
      if (i == 0){
				vD			+=		vW;
		}
      if (i == m-1){
				vD			+=		vE;
		}		
		ierr = MatSetValues(A,1,&Ii,1,&Ii,&vD,INSERT_VALUES);CHKERRQ(ierr);
	}

	ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
		ierr = VecRestoreArray(rhol_vec,&rhol_arr);CHKERRQ(ierr);	
		ierr = VecDestroy(&rhol_vec);CHKERRQ(ierr);
		ierr = VecScatterDestroy(&rho_scatter_ctx);CHKERRQ(ierr);
		ierr = ISDestroy(&local_indices);CHKERRQ(ierr);
		ierr = ISDestroy(&global_indices);CHKERRQ(ierr);		
	ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	return ierr;
}

#ifdef MPE_log
PetscErrorCode SetOperatorT(Mat& A, const Vec& rho, const PetscInt& m,const PetscInt& n,const PetscInt& NGhost, const PetscReal& dx,const PetscReal& dy,std::vector<PetscLogEvent>& petscevents,std::vector<int>& MPE_events,const MPI_Comm& petsc_comm){
	PetscInt			i,j,Ii,II,Ji,JJ,Istart,Iend, Nrhol,ILs,ILe;
	IS					local_indices, global_indices;
	VecScatter		rho_scatter_ctx;
	PetscReal			idx2, idy2;
	PetscScalar		vN, vS, vE, vW, vD, rhoII, rhoJJ, rhoN, rhoS, rhoE, rhoW;
	PetscScalar*	   rhol_arr;
	Vec					rhol_vec;
	PetscErrorCode ierr;
	PetscLogEventBegin(petscevents[4],0,0,0,0);
	MPE_Log_event(MPE_events[8],0,"SPDE Setup-start");
	ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
	global_local_Nelements(Nrhol,ILs,ILe,Istart,Iend,NGhost,m);
	ierr = VecCreate(PETSC_COMM_SELF,&rhol_vec);CHKERRQ(ierr);
	ierr = VecSetSizes(rhol_vec,PETSC_DECIDE,Nrhol);CHKERRQ(ierr);
	ierr = VecSetFromOptions(rhol_vec);CHKERRQ(ierr);	
	ierr = ISCreateStride(petsc_comm,Nrhol,ILs,1,&global_indices);CHKERRQ(ierr); // Get the indices for global rho vector
	ierr = ISCreateStride(PETSC_COMM_SELF,Nrhol,0,1,&local_indices);CHKERRQ(ierr); // Indices for local rhol vector
	ierr = VecScatterCreate(rho,global_indices,rhol_vec,local_indices,&rho_scatter_ctx);
	PetscLogEventEnd(petscevents[4],0,0,0,0);
	MPE_Log_event(MPE_events[9],0,"SPDE Setup-end");
	// Copy elements from rho to rhol_vec
  PetscLogEventBegin(petscevents[3],0,0,0,0);
  MPE_Log_event(MPE_events[6],0,"SPDE Comm-start");
	ierr = VecScatterBegin(rho_scatter_ctx,rho,rhol_vec,INSERT_VALUES,SCATTER_FORWARD);
	ierr = VecScatterEnd(rho_scatter_ctx,rho,rhol_vec,INSERT_VALUES,SCATTER_FORWARD);
  PetscLogEventEnd(petscevents[3],0,0,0,0);
  MPE_Log_event(MPE_events[7],0,"SPDE Comm-end");
  PetscLogEventBegin(petscevents[4],0,0,0,0);
  MPE_Log_event(MPE_events[8],0,"SPDE Setup-start");
	ierr = VecGetArray(rhol_vec,&rhol_arr);CHKERRQ(ierr);
	
	idx2 = 1.0/(dx*dx);
	idy2 = 1.0/(dy*dy);
	for (Ii=Istart; Ii<Iend; Ii++) {
		vD = 0.; j = (PetscInt) Ii/m; i = Ii - j*m;
		II = (i + NGhost) + (m+2*NGhost)*(j + NGhost);
		rhoII = rhol_arr[II-ILs];
		vW = 0.; vE = 0.; vS = 0.; vN = 0.; rhoW = 0.; rhoE = 0.; rhoS = 0.; rhoN = 0.;
		JJ = II - 1;
		rhoJJ = rhol_arr[JJ-ILs];
		rhoW = .5*(rhoII + rhoJJ);
		if (i>0)   {
			Ji = Ii - 1;
			vW = -rhoW*idx2;
			ierr = MatSetValues(A,1,&Ii,1,&Ji,&vW,INSERT_VALUES);CHKERRQ(ierr);
		}
		JJ = II + 1;
		rhoJJ = rhol_arr[JJ-ILs];
		rhoE = .5*(rhoII + rhoJJ);
		if (i<m-1) {
			Ji = Ii + 1; 
			vE = -rhoE*idx2;
			ierr = MatSetValues(A,1,&Ii,1,&Ji,&vE,INSERT_VALUES);CHKERRQ(ierr);
		}
		JJ = II - (m + 2*NGhost);
		rhoJJ = rhol_arr[JJ-ILs];
		rhoS = .5*(rhoII + rhoJJ);
		if (j>0)   {
			Ji = Ii - m;
			vS = -rhoS*idy2;
			ierr = MatSetValues(A,1,&Ii,1,&Ji,&vS,INSERT_VALUES);CHKERRQ(ierr);
		}
		JJ = II + (m + 2*NGhost);
		rhoJJ = rhol_arr[JJ-ILs];
		rhoN = .5*(rhoII + rhoJJ);
		if (j<n-1) {
			Ji = Ii + m; 
			vN = -rhoN*idy2;
			ierr = MatSetValues(A,1,&Ii,1,&Ji,&vN,INSERT_VALUES);CHKERRQ(ierr);
		}
		vW = rhoW*idx2;
		vE = rhoE*idx2;
		vS = rhoS*idy2;
		vN = rhoN*idy2;
		vD = vW + vE + vS + vN;
		if (j == 0){
      			vD			+=		vS;
		}
      if (j == n-1){
				vD			+=		vN;
		}
      if (i == 0){
				vD			+=		vW;
		}
      if (i == m-1){
				vD			+=		vE;
		}		
		ierr = MatSetValues(A,1,&Ii,1,&Ii,&vD,INSERT_VALUES);CHKERRQ(ierr);
	}
  PetscLogEventEnd(petscevents[4],0,0,0,0);
  MPE_Log_event(MPE_events[9],0,"SPDE Setup-end");
  PetscLogEventBegin(petscevents[3],0,0,0,0);
  MPE_Log_event(MPE_events[6],0,"SPDE Comm-start");
	ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
		ierr = VecRestoreArray(rhol_vec,&rhol_arr);CHKERRQ(ierr);	
		ierr = VecDestroy(&rhol_vec);CHKERRQ(ierr);
		ierr = VecScatterDestroy(&rho_scatter_ctx);CHKERRQ(ierr);
		ierr = ISDestroy(&local_indices);CHKERRQ(ierr);
		ierr = ISDestroy(&global_indices);CHKERRQ(ierr);		
	ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	PetscLogEventEnd(petscevents[3],0,0,0,0);
	MPE_Log_event(MPE_events[7],0,"SPDE Comm-end");
	return ierr;
}
#endif
PetscErrorCode SetRandSource(Vec& Z,const PetscInt& NT, const PetscReal& dx, const PetscReal& dy, const PetscMPIInt& rank, std::default_random_engine& generator){
	PetscErrorCode ierr;
	PetscInt Ii = 0, Istart = 0, Iend = 0;
	PetscScalar x, result;
	std::normal_distribution<PetscScalar> distribution(0.0,1.0);
	ierr = VecGetOwnershipRange(Z,&Istart,&Iend);CHKERRQ(ierr);
	for (Ii = Istart; Ii < Iend; ++Ii){
		x = distribution(generator);
		result = x/sqrt(dx*dy);
		ierr = VecSetValues(Z,1,&Ii,&result,INSERT_VALUES);CHKERRQ(ierr);
	}
	ierr = VecAssemblyBegin(Z);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(Z);CHKERRQ(ierr);
	return ierr;	
}

PetscErrorCode SetSource(Vec& b,const PetscInt& m,const PetscInt& N,const PetscReal& dx,const PetscReal& dy,const PetscReal& UN,const PetscReal& US,const PetscReal& UE,const PetscReal& UW,const PetscReal& lamb,const PetscBool& flg){
	PetscErrorCode ierr;
	PetscInt Ii, Istart, Iend, n, i, j;
	PetscRandom    randomvec;
	PetscScalar userb;
	PetscReal ihx2,ihy2;
	ihx2 = 1.0/dx/dx;
	ihy2 = 1.0/dy/dy;
	n = N/m;
	ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
	ierr = VecSetSizes(b,PETSC_DECIDE,N);CHKERRQ(ierr);
	ierr = VecSetFromOptions(b);CHKERRQ(ierr);
	if (flg) {
		ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randomvec);CHKERRQ(ierr);
		ierr = PetscRandomSetFromOptions(randomvec);CHKERRQ(ierr);
		ierr = VecSetRandom(b,randomvec);CHKERRQ(ierr);
		ierr = PetscRandomDestroy(&randomvec);CHKERRQ(ierr);
	} else {
		ierr = VecSet(b,0.0);CHKERRQ(ierr);
	}
	ierr = VecGetOwnershipRange(b,&Istart,&Iend);CHKERRQ(ierr);
	for (Ii = Istart; Ii < Iend; ++Ii){
		j = (PetscInt) Ii/m; i = Ii - j*m;
		if (i == 0){
			userb = UW*2*ihx2;
			ierr = VecSetValues(b,1,&Ii,&userb,ADD_VALUES);CHKERRQ(ierr);
		}
		else if (i == (m-1)){
			userb = UE*2*ihx2;
			ierr = VecSetValues(b,1,&Ii,&userb,ADD_VALUES);CHKERRQ(ierr);
		}
		else if (j == 0){
			userb = US*2*ihy2;
			ierr = VecSetValues(b,1,&Ii,&userb,ADD_VALUES);CHKERRQ(ierr);
		}
		else if (j == (n-1)){
			userb = UN*2*ihy2;
			ierr = VecSetValues(b,1,&Ii,&userb,ADD_VALUES);CHKERRQ(ierr);
		}
	}
	ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode SetSource(Vec& b,const Vec& rho,const PetscInt& m,const PetscInt& n,const PetscInt& NGhost,const PetscReal& dx,const PetscReal& dy,const PetscReal& UN,const PetscReal& US,const PetscReal& UE,const PetscReal& UW,const PetscReal& lamb){
	PetscErrorCode ierr;
	PetscInt Ii, II, JJ, Istart, Iend, i, j, Nrhol,ILs,ILe;
	PetscScalar userb, rhoII, rhoJJ;
	Vec					rhol_vec;
	PetscScalar*		rhol_arr;
	IS					local_indices, global_indices;
	VecScatter		rho_scatter_ctx;
	PetscReal ihx2,ihy2;
	ihx2 = 1.0/dx/dx;
	ihy2 = 1.0/dy/dy;
	ierr = VecGetOwnershipRange(b,&Istart,&Iend);CHKERRQ(ierr);
	ierr = VecSet(b,0.);CHKERRQ(ierr);
	global_local_Nelements(Nrhol,ILs,ILe,Istart,Iend,NGhost,m);
	ierr = VecCreate(PETSC_COMM_SELF,&rhol_vec);CHKERRQ(ierr);
	ierr = VecSetSizes(rhol_vec,PETSC_DECIDE,Nrhol);CHKERRQ(ierr);
	ierr = VecSetFromOptions(rhol_vec);CHKERRQ(ierr);	

	ierr = ISCreateStride(PETSC_COMM_WORLD,Nrhol,ILs,1,&global_indices);CHKERRQ(ierr); // Get the indices for global rho vector
	ierr = ISCreateStride(PETSC_COMM_SELF,Nrhol,0,1,&local_indices);CHKERRQ(ierr); // Indices for local rhol vector
	ierr = VecScatterCreate(rho,global_indices,rhol_vec,local_indices,&rho_scatter_ctx);
	// Copy elements from rho to rhol_vec
	ierr = VecScatterBegin(rho_scatter_ctx,rho,rhol_vec,INSERT_VALUES,SCATTER_FORWARD);
	ierr = VecScatterEnd(rho_scatter_ctx,rho,rhol_vec,INSERT_VALUES,SCATTER_FORWARD);
	ierr = VecGetArray(rhol_vec,&rhol_arr);CHKERRQ(ierr);	
	
	
	for (Ii = Istart; Ii < Iend; ++Ii){
		j = (PetscInt) Ii/m; i = Ii - j*m;
		II = (i + NGhost) + (m+2*NGhost)*(j + NGhost);
		rhoII = rhol_arr[II-ILs];
		if (rhoII == 0) SETERRQ(PETSC_COMM_WORLD,1,"Something wrong b with rhoII");
		if (i == 0){
			JJ = II - 1;
			rhoJJ = rhol_arr[JJ-ILs];
			if (rhoJJ == 0) SETERRQ(PETSC_COMM_WORLD,1,"Something wrong b with rhoJJ-1");
			userb = (0.5)*(rhoII + rhoJJ)*UW*2*ihx2;
			ierr = VecSetValues(b,1,&Ii,&userb,ADD_VALUES);CHKERRQ(ierr);
		}
		else if (i == (m-1)){
			JJ = II + 1;
			rhoJJ = rhol_arr[JJ-ILs];
			if (rhoJJ == 0) SETERRQ(PETSC_COMM_WORLD,1,"Something wrong b with rhoJJ+1");
			userb = (0.5)*(rhoII + rhoJJ)*UE*2*ihx2;
			ierr = VecSetValues(b,1,&Ii,&userb,ADD_VALUES);CHKERRQ(ierr);
		}
		else if (j == 0){
			JJ = II - (m + 2*NGhost);
			rhoJJ = rhol_arr[JJ-ILs];
			if (rhoJJ == 0) SETERRQ(PETSC_COMM_WORLD,1,"Something wrong b with rhoJJ-m");
			userb = (0.5)*(rhoII + rhoJJ)*US*2*ihy2;
			ierr = VecSetValues(b,1,&Ii,&userb,ADD_VALUES);CHKERRQ(ierr);
		}
		else if (j == (n-1)){
			JJ = II + (m + 2*NGhost);
			rhoJJ = rhol_arr[JJ-ILs];
			if (rhoJJ == 0) SETERRQ(PETSC_COMM_WORLD,1,"Something wrong b with rhoJJ+m");
			userb = (0.5)*(rhoII + rhoJJ)*UN*2*ihy2;
			ierr = VecSetValues(b,1,&Ii,&userb,ADD_VALUES);CHKERRQ(ierr);
		}
	}
	ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
		ierr = VecRestoreArray(rhol_vec,&rhol_arr);CHKERRQ(ierr);	
		ierr = VecDestroy(&rhol_vec);CHKERRQ(ierr);
		ierr = VecScatterDestroy(&rho_scatter_ctx);CHKERRQ(ierr);
		ierr = ISDestroy(&local_indices);CHKERRQ(ierr);
		ierr = ISDestroy(&global_indices);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
	
	return ierr;
}
PetscErrorCode SetSource(Vec& b,const Vec& rho,const PetscInt& m,const PetscInt& n,const PetscInt& NGhost,const PetscReal& dx,const PetscReal& dy,const PetscReal& UN,const PetscReal& US,const PetscReal& UE,const PetscReal& UW,const PetscReal& lamb,const MPI_Comm& petsc_comm){
	PetscErrorCode ierr;
	PetscInt Ii, II, JJ, Istart, Iend, i, j, Nrhol,ILs,ILe;
	PetscScalar userb, rhoII, rhoJJ;
	Vec					rhol_vec;
	PetscScalar*		rhol_arr;
	IS					local_indices, global_indices;
	VecScatter		rho_scatter_ctx;
	PetscReal ihx2,ihy2;
	ihx2 = 1.0/dx/dx;
	ihy2 = 1.0/dy/dy;
	ierr = VecGetOwnershipRange(b,&Istart,&Iend);CHKERRQ(ierr);
	ierr = VecSet(b,0.);CHKERRQ(ierr);
	global_local_Nelements(Nrhol,ILs,ILe,Istart,Iend,NGhost,m);
	ierr = VecCreate(PETSC_COMM_SELF,&rhol_vec);CHKERRQ(ierr);
	ierr = VecSetSizes(rhol_vec,PETSC_DECIDE,Nrhol);CHKERRQ(ierr);
	ierr = VecSetFromOptions(rhol_vec);CHKERRQ(ierr);	

	ierr = ISCreateStride(petsc_comm,Nrhol,ILs,1,&global_indices);CHKERRQ(ierr); // Get the indices for global rho vector
	ierr = ISCreateStride(PETSC_COMM_SELF,Nrhol,0,1,&local_indices);CHKERRQ(ierr); // Indices for local rhol vector
	ierr = VecScatterCreate(rho,global_indices,rhol_vec,local_indices,&rho_scatter_ctx);
	// Copy elements from rho to rhol_vec
	ierr = VecScatterBegin(rho_scatter_ctx,rho,rhol_vec,INSERT_VALUES,SCATTER_FORWARD);
	ierr = VecScatterEnd(rho_scatter_ctx,rho,rhol_vec,INSERT_VALUES,SCATTER_FORWARD);
	ierr = VecGetArray(rhol_vec,&rhol_arr);CHKERRQ(ierr);	
	
	
	for (Ii = Istart; Ii < Iend; ++Ii){
		j = (PetscInt) Ii/m; i = Ii - j*m;
		II = (i + NGhost) + (m+2*NGhost)*(j + NGhost);
		rhoII = rhol_arr[II-ILs];
		if (rhoII == 0) SETERRQ(petsc_comm,1,"Something wrong b with rhoII");
		if (i == 0){
			JJ = II - 1;
			rhoJJ = rhol_arr[JJ-ILs];
			if (rhoJJ == 0) SETERRQ(petsc_comm,1,"Something wrong b with rhoJJ-1");
			userb = (0.5)*(rhoII + rhoJJ)*UW*2*ihx2;
			ierr = VecSetValues(b,1,&Ii,&userb,ADD_VALUES);CHKERRQ(ierr);
		}
		else if (i == (m-1)){
			JJ = II + 1;
			rhoJJ = rhol_arr[JJ-ILs];
			if (rhoJJ == 0) SETERRQ(petsc_comm,1,"Something wrong b with rhoJJ+1");
			userb = (0.5)*(rhoII + rhoJJ)*UE*2*ihx2;
			ierr = VecSetValues(b,1,&Ii,&userb,ADD_VALUES);CHKERRQ(ierr);
		}
		else if (j == 0){
			JJ = II - (m + 2*NGhost);
			rhoJJ = rhol_arr[JJ-ILs];
			if (rhoJJ == 0) SETERRQ(petsc_comm,1,"Something wrong b with rhoJJ-m");
			userb = (0.5)*(rhoII + rhoJJ)*US*2*ihy2;
			ierr = VecSetValues(b,1,&Ii,&userb,ADD_VALUES);CHKERRQ(ierr);
		}
		else if (j == (n-1)){
			JJ = II + (m + 2*NGhost);
			rhoJJ = rhol_arr[JJ-ILs];
			if (rhoJJ == 0) SETERRQ(petsc_comm,1,"Something wrong b with rhoJJ+m");
			userb = (0.5)*(rhoII + rhoJJ)*UN*2*ihy2;
			ierr = VecSetValues(b,1,&Ii,&userb,ADD_VALUES);CHKERRQ(ierr);
		}
	}
	ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
		ierr = VecRestoreArray(rhol_vec,&rhol_arr);CHKERRQ(ierr);	
		ierr = VecDestroy(&rhol_vec);CHKERRQ(ierr);
		ierr = VecScatterDestroy(&rho_scatter_ctx);CHKERRQ(ierr);
		ierr = ISDestroy(&local_indices);CHKERRQ(ierr);
		ierr = ISDestroy(&global_indices);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
	
	return ierr;
}
#ifdef MPE_log
PetscErrorCode SetSourceT(Vec& b,const Vec& rho,const PetscInt& m,const PetscInt& n,const PetscInt& NGhost,const PetscReal& dx,const PetscReal& dy,const PetscReal& UN,const PetscReal& US,const PetscReal& UE,const PetscReal& UW,const PetscReal& lamb,std::vector<PetscLogEvent>& petscevents,std::vector<int>& MPE_events,const MPI_Comm& petsc_comm){
	PetscErrorCode ierr;
	PetscInt Ii, II, JJ, Istart, Iend, i, j, Nrhol,ILs,ILe;
	PetscScalar userb, rhoII, rhoJJ;
	PetscScalar startTime1;
	Vec					rhol_vec;
	PetscScalar*		rhol_arr;
	IS					local_indices, global_indices;
	VecScatter		rho_scatter_ctx;
	PetscReal ihx2,ihy2;
	ihx2 = 1.0/dx/dx;
	ihy2 = 1.0/dy/dy;
	PetscLogEventBegin(petscevents[4],0,0,0,0);
	MPE_Log_event(MPE_events[8],0,"SPDE Setup-start");
	ierr = VecGetOwnershipRange(b,&Istart,&Iend);CHKERRQ(ierr);
	ierr = VecSet(b,0.);CHKERRQ(ierr);
	global_local_Nelements(Nrhol,ILs,ILe,Istart,Iend,NGhost,m);
	ierr = VecCreate(PETSC_COMM_SELF,&rhol_vec);CHKERRQ(ierr);
	ierr = VecSetSizes(rhol_vec,PETSC_DECIDE,Nrhol);CHKERRQ(ierr);
	ierr = VecSetFromOptions(rhol_vec);CHKERRQ(ierr);	

	ierr = ISCreateStride(petsc_comm,Nrhol,ILs,1,&global_indices);CHKERRQ(ierr); // Get the indices for global rho vector
	ierr = ISCreateStride(PETSC_COMM_SELF,Nrhol,0,1,&local_indices);CHKERRQ(ierr); // Indices for local rhol vector
	ierr = VecScatterCreate(rho,global_indices,rhol_vec,local_indices,&rho_scatter_ctx);
	// Copy elements from rho to rhol_vec
	PetscLogEventEnd(petscevents[4],0,0,0,0);
	MPE_Log_event(MPE_events[9],0,"SPDE Setup-end");
	PetscLogEventBegin(petscevents[3],0,0,0,0);
	MPE_Log_event(MPE_events[6],0,"SPDE Comm-start");
	ierr = VecScatterBegin(rho_scatter_ctx,rho,rhol_vec,INSERT_VALUES,SCATTER_FORWARD);
	ierr = VecScatterEnd(rho_scatter_ctx,rho,rhol_vec,INSERT_VALUES,SCATTER_FORWARD);
	PetscLogEventEnd(petscevents[3],0,0,0,0);
	MPE_Log_event(MPE_events[7],0,"SPDE Comm-end");
	PetscLogEventBegin(petscevents[4],0,0,0,0);
	MPE_Log_event(MPE_events[8],0,"SPDE Setup-start");
	ierr = VecGetArray(rhol_vec,&rhol_arr);CHKERRQ(ierr);	
	
	
	for (Ii = Istart; Ii < Iend; ++Ii){
		j = (PetscInt) Ii/m; i = Ii - j*m;
		II = (i + NGhost) + (m+2*NGhost)*(j + NGhost);
		rhoII = rhol_arr[II-ILs];
		if (rhoII == 0) SETERRQ(petsc_comm,1,"Something wrong b with rhoII");
		if (i == 0){
			JJ = II - 1;
			rhoJJ = rhol_arr[JJ-ILs];
			if (rhoJJ == 0) SETERRQ(petsc_comm,1,"Something wrong b with rhoJJ-1");
			userb = (0.5)*(rhoII + rhoJJ)*UW*2*ihx2;
			ierr = VecSetValues(b,1,&Ii,&userb,ADD_VALUES);CHKERRQ(ierr);
		}
		else if (i == (m-1)){
			JJ = II + 1;
			rhoJJ = rhol_arr[JJ-ILs];
			if (rhoJJ == 0) SETERRQ(petsc_comm,1,"Something wrong b with rhoJJ+1");
			userb = (0.5)*(rhoII + rhoJJ)*UE*2*ihx2;
			ierr = VecSetValues(b,1,&Ii,&userb,ADD_VALUES);CHKERRQ(ierr);
		}
		else if (j == 0){
			JJ = II - (m + 2*NGhost);
			rhoJJ = rhol_arr[JJ-ILs];
			if (rhoJJ == 0) SETERRQ(petsc_comm,1,"Something wrong b with rhoJJ-m");
			userb = (0.5)*(rhoII + rhoJJ)*US*2*ihy2;
			ierr = VecSetValues(b,1,&Ii,&userb,ADD_VALUES);CHKERRQ(ierr);
		}
		else if (j == (n-1)){
			JJ = II + (m + 2*NGhost);
			rhoJJ = rhol_arr[JJ-ILs];
			if (rhoJJ == 0) SETERRQ(petsc_comm,1,"Something wrong b with rhoJJ+m");
			userb = (0.5)*(rhoII + rhoJJ)*UN*2*ihy2;
			ierr = VecSetValues(b,1,&Ii,&userb,ADD_VALUES);CHKERRQ(ierr);
		}
	}
	PetscLogEventEnd(petscevents[4],0,0,0,0);
	MPE_Log_event(MPE_events[9],0,"SPDE Setup-end");
	PetscLogEventBegin(petscevents[3],0,0,0,0);
	MPE_Log_event(MPE_events[6],0,"SPDE Comm-start");
	ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
		ierr = VecRestoreArray(rhol_vec,&rhol_arr);CHKERRQ(ierr);	
		ierr = VecDestroy(&rhol_vec);CHKERRQ(ierr);
		ierr = VecScatterDestroy(&rho_scatter_ctx);CHKERRQ(ierr);
		ierr = ISDestroy(&local_indices);CHKERRQ(ierr);
		ierr = ISDestroy(&global_indices);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
	PetscLogEventBegin(petscevents[3],0,0,0,0);
	MPE_Log_event(MPE_events[7],0,"SPDE Comm-end");
	return ierr;
}
#endif
PetscErrorCode GetOptions(UserCTX& users){ // Not sure if it's also affecting MPI_COMM_WORLD. See if errors
	PetscErrorCode ierr;
	ierr = PetscOptionsGetInt(NULL,"-m",&users.m,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(NULL,"-n",&users.n,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-x0",&users.x0,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-x1",&users.x1,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-y0",&users.y0,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-y1",&users.y1,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-UN",&users.UN,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-US",&users.US,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-UE",&users.UE,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-UW",&users.UW,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(NULL,"-Nsamples",&users.Nsamples,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-TOL",&users.TOL,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-lamb",&users.lamb,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-alpha",&users.alpha,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-dim",&users.dim,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-sigma",&users.sigma,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(NULL,"-ppc",&	users.procpercolor,NULL);CHKERRQ(ierr);
	users.dx = (users.x1 - users.x0)/(PetscReal)users.m;
	users.dy = (users.y1 - users.y0)/(PetscReal)users.n;
	users.nu = users.alpha - users.dim/2.0;
	users.kappa = sqrt(8.0*users.nu)/(users.lamb);
	users.tau2 = (tgamma(users.nu)/(tgamma(users.nu + 0.5)*pow((4.0*M_PI),(users.dim/2.0))*pow(users.kappa,(2.0*users.nu))*users.sigma*users.sigma));	
	users.NGhost += PetscMax(PetscMax(nint(0.5*users.lamb/users.dx),nint(0.5*users.lamb/users.dy)),nint(0.5*users.lamb/(sqrt(users.dx*users.dx + users.dy*users.dy))));
	users.NI = users.m*users.n;
	users.NT = (users.m+2*users.NGhost)*(users.n+2*users.NGhost);	
	return ierr;
}

PetscErrorCode VecPostProcs(const Vec& U,const char* filename){
	PetscViewer solmview;
	PetscErrorCode ierr;
	ierr = PetscViewerCreate(PETSC_COMM_WORLD,&solmview);CHKERRQ(ierr);
	ierr = PetscViewerSetType(solmview,PETSCVIEWERASCII);CHKERRQ(ierr);
	ierr = PetscViewerSetFormat(solmview,PETSC_VIEWER_DEFAULT);
	ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename,&solmview);CHKERRQ(ierr);
	ierr = VecView(U,solmview);CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&solmview);CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode VecPostProcs(const Vec& U,const char* filename,const PetscMPIInt& rank){
	Vec buffer;
	PetscScalar* buffer_arr;
	PetscInt N;
	VecScatter ctx;
	PetscErrorCode ierr;
	ierr = VecScatterCreateToZero(U,&ctx,&buffer); CHKERRQ(ierr);
	ierr = VecGetSize(U,&N); CHKERRQ(ierr);
	// scatter as many times as you need
	ierr = VecScatterBegin(ctx,U,buffer,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
	ierr = VecScatterEnd(ctx,U,buffer,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
	if (rank == 0){
		std::ofstream vecout;
		vecout.open(filename);
		ierr = VecGetArray(buffer,&buffer_arr);CHKERRQ(ierr);
		for (int i = 0; i < N; ++i)
			vecout << buffer_arr[i] << std::endl;
		vecout.close();
	}
	// destroy scatter context and local vector when no longer needed
	ierr = VecScatterDestroy(&ctx); CHKERRQ(ierr);
	ierr = VecDestroy(&buffer); CHKERRQ(ierr);
	return ierr;
}

void VecPostProcs(const std::vector<PetscScalar>& U,const char* filename,const PetscMPIInt& rank){
	if (rank == 0){
		std::ofstream vecout;
		vecout.open(filename);
		std::for_each(U.begin(),U.end(),[&vecout](PetscScalar result){vecout << result << std::endl;});
		vecout.close();
	}
}

PetscErrorCode MatPostProcs(const Mat& U,const char* filename){
	PetscViewer solmview;
	PetscErrorCode ierr;
	ierr = PetscViewerCreate(PETSC_COMM_WORLD,&solmview);CHKERRQ(ierr);
	ierr = PetscViewerSetType(solmview,PETSCVIEWERASCII);CHKERRQ(ierr);
	ierr = PetscViewerSetFormat(solmview,PETSC_VIEWER_ASCII_DENSE);
	ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename,&solmview);CHKERRQ(ierr);
	ierr = MatView(U,solmview);CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&solmview);CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode update_stats(Vec& EUN, Vec& VUN, Vec& EUNm1, Vec& M2N, PetscReal& tol,const Vec& U, const PetscInt& Ns){
	PetscErrorCode ierr;
	PetscReal normE, normV, tolE, tolV, tolrE, tolrV;
	Vec VUNm1, dUN, dUNm1, resU, M2Nm1;

	ierr = VecDuplicate(U,&M2Nm1);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&dUNm1);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&dUN);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&VUNm1);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&resU);CHKERRQ(ierr);

	ierr = VecCopy(VUN,VUNm1); CHKERRQ(ierr); // VUNm1 = VUN;
	ierr = VecCopy(EUN,EUNm1); CHKERRQ(ierr); // EUNm1 = EUN;
	ierr = VecCopy(M2N,M2Nm1); CHKERRQ(ierr); // M2Nm1 = M2N;
	ierr = VecWAXPY(dUNm1,-1,EUNm1,U); CHKERRQ(ierr); // Calculate dUNm1 = Un - mu_n-1
	ierr = VecWAXPY(EUN,1.0/(PetscScalar)Ns,dUNm1,EUNm1); CHKERRQ(ierr); // mu_n = mu_n-1 + 1/n * dUNm1
	ierr = VecWAXPY(dUN,-1,EUN,U); CHKERRQ(ierr); // dUN = Un - mu_n
	ierr = VecPointwiseMult(M2N,dUN,dUNm1);CHKERRQ(ierr); // M2N = dUN * dUNm1;
	ierr = VecAXPY(M2N,1,M2Nm1); // M2N = M2Nm1 + M2N;
	ierr = VecCopy(M2N,VUN); // VUN = M2N;
	ierr = VecScale(VUN,1.0/(PetscScalar)Ns);
	
	ierr = VecNorm(EUN,NORM_INFINITY,&normE);CHKERRQ(ierr);
	ierr = VecNorm(VUN,NORM_INFINITY,&normV);CHKERRQ(ierr);
	
	ierr = VecWAXPY(resU,-1,EUN,EUNm1);CHKERRQ(ierr);
	ierr = VecAbs(resU);CHKERRQ(ierr);
	ierr = VecNorm(resU,NORM_INFINITY,&tolE);CHKERRQ(ierr);
	ierr = VecWAXPY(resU,-1,VUN,VUNm1);CHKERRQ(ierr);
	ierr = VecAbs(resU);CHKERRQ(ierr);
	ierr = VecNorm(resU,NORM_INFINITY,&tolV);CHKERRQ(ierr);
	tolrE = tolE/(normE + 1.e-12); tolrV = tolV/(normV + 1.e-12);
	tolrV = PetscMax(tolrV,tolrE);
	tol  = PetscMax(tolE,tolV);
	tol  = PetscMax(tolrV,tol);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Sample[%d]: Tol = %f, Mean = %f, Var = %f \n",Ns,tol,normE,normV);CHKERRQ(ierr);
	ierr = VecDestroy(&dUN);CHKERRQ(ierr);
	ierr = VecDestroy(&M2Nm1);CHKERRQ(ierr);
	ierr = VecDestroy(&dUNm1);CHKERRQ(ierr);	
	ierr = VecDestroy(&VUNm1);CHKERRQ(ierr);
	ierr = VecDestroy(&resU);CHKERRQ(ierr);	
	return ierr;
}

PetscErrorCode update_stats(Vec& EUN, Vec& VUN, Vec& EUNm1, Vec& M2N, const Vec& U, const PetscInt& Ns){
	PetscErrorCode ierr;
	PetscReal normE, normV;
	Vec VUNm1, dUN, dUNm1, M2Nm1;

	ierr = VecDuplicate(U,&M2Nm1);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&dUNm1);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&dUN);CHKERRQ(ierr);
	ierr = VecDuplicate(U,&VUNm1);CHKERRQ(ierr);

	ierr = VecCopy(VUN,VUNm1); CHKERRQ(ierr); // VUNm1 = VUN;
	ierr = VecCopy(EUN,EUNm1); CHKERRQ(ierr); // EUNm1 = EUN;
	ierr = VecCopy(M2N,M2Nm1); CHKERRQ(ierr); // M2Nm1 = M2N;
	ierr = VecWAXPY(dUNm1,-1,EUNm1,U); CHKERRQ(ierr); // Calculate dUNm1 = Un - mu_n-1
	ierr = VecWAXPY(EUN,1.0/(PetscScalar)Ns,dUNm1,EUNm1); CHKERRQ(ierr); // mu_n = mu_n-1 + 1/n * dUNm1
	ierr = VecWAXPY(dUN,-1,EUN,U); CHKERRQ(ierr); // dUN = Un - mu_n
	ierr = VecPointwiseMult(M2N,dUN,dUNm1);CHKERRQ(ierr); // M2N = dUN * dUNm1;
	ierr = VecAXPY(M2N,1,M2Nm1); // M2N = M2Nm1 + M2N;
	ierr = VecCopy(M2N,VUN); // VUN = M2N;
	ierr = VecScale(VUN,1.0/(PetscScalar)Ns);
	
	ierr = VecNorm(EUN,NORM_INFINITY,&normE);CHKERRQ(ierr);
	ierr = VecNorm(VUN,NORM_INFINITY,&normV);CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Sample[%d]: Mean = %f, Var = %f \n",Ns,normE,normV);CHKERRQ(ierr);
	ierr = VecDestroy(&dUN);CHKERRQ(ierr);
	ierr = VecDestroy(&M2Nm1);CHKERRQ(ierr);
	ierr = VecDestroy(&dUNm1);CHKERRQ(ierr);	
	ierr = VecDestroy(&VUNm1);CHKERRQ(ierr);	
	return ierr;
}

PetscErrorCode update_stats(PetscScalar& EUN,PetscScalar& VUN,PetscScalar& EUNm1,PetscScalar& M2N,PetscScalar& tol,const PetscScalar& U,const PetscInt& Ns){
	PetscScalar tolrE;
	PetscErrorCode ierr;
	EUNm1 = EUN;
	EUN = EUNm1 + (U - EUNm1)/(PetscScalar) Ns;
	M2N += (U - EUNm1)*(U - EUN);
	VUN = M2N/(PetscScalar) Ns;
	if (Ns <= 1){
	tol = std::abs(EUN - EUNm1);
	tolrE = tol/(EUN + 1.e-12);
	tol  = PetscMax(tolrE,tol);
	}
	else tol  = sqrt(VUN/(PetscScalar) Ns);
	ierr = PetscPrintf(MPI_COMM_WORLD,"Sample[%d]: Expectation = %3.8E\tVariance = %3.8E\tTol = %3.8E \n",Ns,EUN,VUN,tol);
	return ierr;
}

void update_iters(PetscScalar* & avg_iter, PetscInt* & max_iter, PetscInt* & min_iter, PetscInt* & iter,const PetscInt& Ns){
  int N = sizeof(iter)/sizeof(PetscInt);
  int *avg_iterm1 = new PetscInt[2];
  for (int i = 0; i< N; ++i){
    avg_iterm1[i] = avg_iter[i];
    avg_iter[i] = avg_iterm1[i] + ((PetscScalar)iter[i] - avg_iterm1[i])/(PetscScalar) Ns;
    max_iter[i] = (max_iter[i] < iter[i])? iter[i] : max_iter[i];
    min_iter[i] = (min_iter[i] > iter[i])? iter[i] : min_iter[i];
  }
}

PetscErrorCode VecSetMean(Vec&,const Vec&){
	PetscErrorCode ierr;
	return ierr;
}
#ifdef SLEPC
PetscErrorCode SVD_Decomp(Mat& Ut, Mat& Vt, Mat& S, const Mat& A){
	PetscErrorCode ierr;
	Vec            u,v;             /* left and right singular vectors */
  	SVD            svd;             /* singular value problem solver context */
  	PetscScalar    sigma;
  PetscInt       nconv,Am,An, Istart, Iend;
  const PetscScalar    *utemp, *vtemp;
  	ierr = MatGetVecs(A,&v,&u);CHKERRQ(ierr);
  	ierr = MatGetSize(A,&Am,&An); CHKERRQ(ierr);
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
  ierr = MatCreate(PETSC_COMM_WORLD,&Ut);CHKERRQ(ierr); // Create matrix Ut residing in PETSC_COMM_WORLD
  ierr = MatSetSizes(Ut,PETSC_DECIDE,PETSC_DECIDE,nconv,Am);CHKERRQ(ierr); // Set the size of the matrix U, and let PETSC decide the decomposition
  ierr = MatSetFromOptions(Ut);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Ut,Am,NULL,Am,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Ut,Am,NULL);CHKERRQ(ierr);  
  ierr = MatSetUp(Ut);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&Vt);CHKERRQ(ierr); // Create matrix Vt residing in PETSC_COMM_WORLD
  ierr = MatSetSizes(Vt,PETSC_DECIDE,PETSC_DECIDE,nconv,An);CHKERRQ(ierr); // Set the size of the matrix V, and let PETSC decide the decomposition
  ierr = MatSetFromOptions(Vt);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Vt,An,NULL,An,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Vt,An,NULL);CHKERRQ(ierr);   
  ierr = MatSetUp(Vt);CHKERRQ(ierr);
  //for (int ii = 0; ii < Am; ++ii) IdxU[ii] = ii;
  //for (int ii = 0; ii < An; ++ii) IdxV[ii] = ii;
  
  for (int i=0;i<nconv;i++) {
    ierr = SVDGetSingularTriplet(svd,i,&sigma,u,v);CHKERRQ(ierr);
    ierr = MatSetValue(S,i,i,sigma,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecGetArrayRead(u,&utemp); CHKERRQ(ierr);
    ierr = VecGetArrayRead(v,&vtemp); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(u,&Istart,&Iend);
    for (int j=Istart; j < Iend; ++j)
      ierr = MatSetValue(Ut,i,j,utemp[j-Istart],INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(v,&Istart,&Iend);
    for (int j=Istart; j < Iend; ++j)
      ierr = MatSetValue(Vt,i,j,vtemp[j-Istart],INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(u,&utemp); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(v,&vtemp); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Ut,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Vt,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Ut,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Vt,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(S,MAT_FINAL_ASSEMBLY);
  SVDView(svd,PETSC_VIEWER_STDOUT_WORLD	);
  ierr = SVDDestroy(&svd);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);  
  return ierr;
}
#endif

void recolor(MPI_Comm& petsc_comm,int& ncolors,const int& numprocs,const int& procpercolor,const int& grank,int& lrank){
//  PetscErrorCode ierr;
  /* Split the communicator into petsc_comm_slaves */
	ncolors = (numprocs-1)/procpercolor;
	int color = (grank-1)/procpercolor;
	if(grank == 0) color = std::numeric_limits<PetscInt>::max();
	if ((color+1) > ncolors)
		color--;
	#if DEBUG
		printf("I am processor %d with color %d \n",grank,color);
	#endif
	MPI_Comm_split(MPI_COMM_WORLD, color, grank, &petsc_comm);
	MPI_Comm_rank(petsc_comm,&lrank); // Get the processor local rank in petsc_comm_slaves [Logically must be after recolor!]
//  return ierr;
}

void global_local_Nelements(PetscInt& Nel, PetscInt& ILs, PetscInt& ILe, const PetscInt& Istart, const PetscInt& Iend, const PetscInt& NGhost, const PetscInt& m){
	PetscInt j_start, i_start, j_end, i_end, Iendm1 = Iend-1, IIe, IIs;
	j_start = (PetscInt) Istart/m; i_start = Istart - j_start*m;
	j_end = (PetscInt) Iendm1/m; i_end = Iendm1 - j_end*m;
	IIs = i_start + NGhost + (m + 2*NGhost)*(j_start + NGhost);
	IIe = i_end + NGhost + (m + 2*NGhost)*(j_end + NGhost);
	ILs = IIs - (m + 2*NGhost);
	ILe = IIe + (m + 2*NGhost);
	Nel = (ILe - ILs + 1);
}
