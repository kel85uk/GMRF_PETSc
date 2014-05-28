PetscErrorCode GetOptions(PetscInt&,PetscInt&,PetscReal&,PetscReal&,PetscReal&,PetscReal&,PetscReal&,PetscReal&,PetscReal&,PetscReal&,PetscReal&);

PetscErrorCode SetGMRFOperator(Mat&, const PetscInt&, const PetscInt&, const PetscInt&, const PetscReal&, const PetscReal&, const PetscReal&);
PetscErrorCode SetOperator(Mat&, const PetscInt&, const PetscInt&, const PetscInt&, const PetscReal&, const PetscReal&);
PetscErrorCode SetOperator(Mat&, const Vec&, const PetscInt&, const PetscInt&, const PetscInt&, const PetscReal&, const PetscReal&);
PetscErrorCode SetRandSource(Vec&,const PetscInt&, const PetscReal&, const PetscReal&, const PetscInt&);
PetscErrorCode SetSource(Vec&,const PetscInt&,const PetscInt&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscBool&);
PetscErrorCode SetSource(Vec&,const Vec&,const PetscInt&,const PetscInt&,const PetscInt&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&,const PetscReal&);
PetscErrorCode PostProcs(const Vec&, const char*);

PetscErrorCode NormRand_BM(Vec&,Vec&,const PetscInt&);
void global_local_Nelements(PetscInt&, PetscInt&, PetscInt&, const PetscInt&, const PetscInt&, const PetscInt&, const PetscInt&);


PetscErrorCode SetGMRFOperator(Mat& L, const PetscInt& m,const PetscInt& n,const PetscInt& NGhost, const PetscReal& dx,const PetscReal& dy, const PetscReal& kappa){
	PetscInt			i,j,Ii,J,Istart,Iend, M = (m + 2*NGhost), N = (n + 2*NGhost);
	PetscReal			dxdy = dx*dy, dxidy = dx/dy, dyidx = dy/dx;
	PetscScalar		vD, vN = -dxidy, vS = -dxidy, vE = -dyidx, vW = -dyidx;
	PetscErrorCode ierr;
	
	ierr = MatGetOwnershipRange(L,&Istart,&Iend);CHKERRQ(ierr);

	for (Ii=Istart; Ii<Iend; Ii++) {
		vD = 0.; j = (PetscInt) Ii/M; i = Ii - j*M;
		if (i>0)   {J = Ii - 1; ierr = MatSetValues(L,1,&Ii,1,&J,&vW,INSERT_VALUES);CHKERRQ(ierr);}
		if (i<M-1) {J = Ii + 1; ierr = MatSetValues(L,1,&Ii,1,&J,&vE,INSERT_VALUES);CHKERRQ(ierr);}
		if (j>0)   {J = Ii - M; ierr = MatSetValues(L,1,&Ii,1,&J,&vS,INSERT_VALUES);CHKERRQ(ierr);}
		if (j<N-1) {J = Ii + M; ierr = MatSetValues(L,1,&Ii,1,&J,&vN,INSERT_VALUES);CHKERRQ(ierr);}
		vD = -(vW + vE + vS + vN) + kappa*dxdy;
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
//	PostProcs(rhol_vec,"rhol_vec.dat");
	ierr = VecGetArray(rhol_vec,&rhol_arr);CHKERRQ(ierr);
	
	idx2 = 1.0/(dx*dx);
	idy2 = 1.0/(dy*dy);
	for (Ii=Istart; Ii<Iend; Ii++) {
		vD = 0.; j = (PetscInt) Ii/m; i = Ii - j*m;
		II = (i + NGhost) + (m+2*NGhost)*(j + NGhost);
		rhoII = rhol_arr[II-ILs];
//		if (rhoII == 0) SETERRQ3(PETSC_COMM_WORLD,1,"Something wrong with rhoII in %d out of %d with NR = %d",Ii,Istart,Nrhol);
		vW = 0.; vE = 0.; vS = 0.; vN = 0.; rhoW = 0.; rhoE = 0.; rhoS = 0.; rhoN = 0.;
		JJ = II - 1;
		rhoJJ = rhol_arr[JJ-ILs];
//		if (rhoJJ == 0) SETERRQ3(PETSC_COMM_WORLD,1,"Something wrong with rhoJJ-1 in %d out of %d with NR = %d",Ii,Istart,Nrhol);
		rhoW = .5*(rhoII + rhoJJ);
		if (i>0)   {
			Ji = Ii - 1;
			vW = -rhoW*idx2;
			ierr = MatSetValues(A,1,&Ii,1,&Ji,&vW,INSERT_VALUES);CHKERRQ(ierr);
		}
		JJ = II + 1;
		rhoJJ = rhol_arr[JJ-ILs];
//		if (rhoJJ == 0) SETERRQ3(PETSC_COMM_WORLD,1,"Something wrong with rhoJJ+1 in %d out of %d with NR = %d",Ii,Istart,Nrhol);
		rhoE = .5*(rhoII + rhoJJ);
		if (i<m-1) {
			Ji = Ii + 1; 
			vE = -rhoE*idx2;
			ierr = MatSetValues(A,1,&Ii,1,&Ji,&vE,INSERT_VALUES);CHKERRQ(ierr);
		}
		JJ = II - (m + 2*NGhost);
		rhoJJ = rhol_arr[JJ-ILs];
//		if (rhoJJ == 0) SETERRQ3(PETSC_COMM_WORLD,1,"Something wrong with rhoJJ-m in %d out of %d with NR = %d",Ii,Istart,Nrhol);
		rhoS = .5*(rhoII + rhoJJ);
		if (j>0)   {
			Ji = Ii - m;
			vS = -rhoS*idy2;
			ierr = MatSetValues(A,1,&Ii,1,&Ji,&vS,INSERT_VALUES);CHKERRQ(ierr);
		}
		JJ = II + (m + 2*NGhost);
		rhoJJ = rhol_arr[JJ-ILs];
//		if (rhoJJ == 0) SETERRQ3(PETSC_COMM_WORLD,1,"Something wrong with rhoJJ+m in %d seeking %d with NR = %d",Ii,JJ-ILs,Nrhol);
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

PetscErrorCode SetRandSource(Vec& Z,const PetscInt& NT, const PetscReal& dx, const PetscReal& dy, const PetscInt& Seed){
	PetscErrorCode ierr;
	Vec		v1, v2;
/* Test to check out variable and routines for now: Need to change to parallel */
	PetscInt Ii = 0, Istart = 0, Iend = 0;
	PetscScalar x, result;
	ierr = VecDuplicate(Z,&v1);CHKERRQ(ierr);
	ierr = VecDuplicate(Z,&v2);CHKERRQ(ierr);
	ierr = NormRand_BM(v1,v2,Seed);CHKERRQ(ierr);
	ierr = VecGetOwnershipRange(Z,&Istart,&Iend);CHKERRQ(ierr);
	for (Ii = Istart; Ii < Iend; ++Ii){
//		boost::variate_generator<boost::mt19937, boost::normal_distribution<> >	generator(boost::mt19937(time(0)),boost::normal_distribution<>(0,1));
/*		std::uniform_int_distribution<int> unif(-NT+1,NT+1);
		std::normal_distribution<double> normal(0,1.0);
		std::mt19937 rnd(Ii);
		std::mt19937_64 rand_engine(unif(rnd)); // mt19937 is a good pseudo-random number 
                                          // generator.

		x = normal(rand_engine)*sqrt(dx*dy); */
		ierr = VecGetValues(v1,1,&Ii,&x);CHKERRQ(ierr);
		result = x*sqrt(dx*dy);
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

PetscErrorCode GetOptions(PetscInt& m,PetscInt& n,PetscReal& x0,PetscReal& x1,PetscReal& y0,PetscReal& y1,PetscReal& UN,PetscReal& US,PetscReal& UE,PetscReal& UW,PetscReal& lamb){
	PetscErrorCode ierr;
	ierr = PetscOptionsGetInt(NULL,"-m",&m,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-x0",&x0,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-x1",&x1,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-y0",&y0,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-y1",&y1,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-UN",&UN,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-US",&US,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-UE",&UE,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-UW",&UW,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-lamb",&lamb,NULL);CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode PostProcs(const Vec& U,const char* filename){
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

PetscErrorCode NormRand_BM(Vec& V1,Vec& V2,const PetscInt& Seed){
	PetscErrorCode ierr;
	PetscBool flg;
	PetscInt	Ii, Istart, Iend;
	PetscScalar	result, v1, v2;
	PetscRandom    randomvec;
	ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randomvec);CHKERRQ(ierr);
	ierr = PetscRandomSetType(randomvec,PETSCRAND);CHKERRQ(ierr);
	ierr = PetscRandomSetSeed(randomvec,Seed);CHKERRQ(ierr);
	ierr = PetscRandomSeed(randomvec);CHKERRQ(ierr);
	ierr = VecSetRandom(V1,randomvec);CHKERRQ(ierr);
	ierr = PetscRandomDestroy(&randomvec);CHKERRQ(ierr);	
	ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randomvec);CHKERRQ(ierr);
	ierr = PetscRandomSetType(randomvec,PETSCRAND);CHKERRQ(ierr);
	ierr = PetscRandomSetSeed(randomvec,Seed);CHKERRQ(ierr);
	ierr = PetscRandomSeed(randomvec);CHKERRQ(ierr);
	ierr = VecSetRandom(V2,randomvec);CHKERRQ(ierr);
	ierr = PetscRandomDestroy(&randomvec);CHKERRQ(ierr);
	VecEqual(V1,V2,&flg);
	if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"NormRand_BM: V1 == V2!");
	ierr = PetscRandomDestroy(&randomvec);CHKERRQ(ierr);
	ierr = VecGetOwnershipRange(V1,&Istart,&Iend);CHKERRQ(ierr);
	for (Ii = Istart; Ii < Iend; ++Ii){
		ierr = VecGetValues(V1,1,&Ii,&v1);CHKERRQ(ierr);
		ierr = VecGetValues(V2,1,&Ii,&v2);CHKERRQ(ierr);
		result = sqrt(-2.0*log(v1))*cos(2.0*M_PI*v2);
		ierr = VecSetValue(V1,Ii,result,INSERT_VALUES);CHKERRQ(ierr);
	}
	ierr = VecAssemblyBegin(V1);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(V1);CHKERRQ(ierr);
	return ierr;
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
