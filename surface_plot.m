%
function surface_plot(m,n,NGhost)
U_data = dlmread('sol_mean.dat');                       
data2 = reshape(U_data,m,n);                        
[X,Y] = meshgrid(linspace(0,1,m),linspace(0,1,n));
figure(1);
surfc(X,Y,data2');                                    
view([0 90]);                                         
shading interp;

%U_data = dlmread('sol_var.dat');                       
%data2 = reshape(U_data,m,n);                        
%[X,Y] = meshgrid(linspace(0,1,m),linspace(0,1,n));
%figure(2);
%surfc(X,Y,data2');                                    
%view([0 90]);                                         
%shading interp;

gmrf_data = dlmread('rho_mean.dat');
gmrf_data2 = reshape(gmrf_data,m+2*NGhost,n+2*NGhost);
figure(3);
surfc(X,Y,gmrf_data2(1+NGhost:1:end-NGhost,1+NGhost:1:end-NGhost)');
view([0 90]);
shading interp;

%gmrf_data = dlmread('rho_var.dat');
%gmrf_data2 = reshape(gmrf_data,m+2*NGhost,n+2*NGhost);
%figure(4);
%surfc(X,Y,gmrf_data2(1+NGhost:1:end-NGhost,1+NGhost:1:end-NGhost)');
%view([0 90]);

%gmrf_data = dlmread('gmrf_mean.dat');
%gmrf_data2 = reshape(gmrf_data,m+2*NGhost,n+2*NGhost);
%figure(5);
%surfc(X,Y,gmrf_data2(1+NGhost:1:end-NGhost,1+NGhost:1:end-NGhost)');
%view([0 90]);
%shading interp;

%gmrf_data = dlmread('gmrf_var.dat');
%gmrf_data2 = reshape(gmrf_data,m+2*NGhost,n+2*NGhost);
%figure(6);
%surfc(X,Y,gmrf_data2(1+NGhost:1:end-NGhost,1+NGhost:1:end-NGhost)');
%view([0 90]);

gmrf_data = dlmread('XR.dat');
gmrf_data2 = reshape(gmrf_data,m+2*NGhost,n+2*NGhost);
figure(7);
surfc(X,Y,gmrf_data2(1+NGhost:1:end-NGhost,1+NGhost:1:end-NGhost)');
view([0 90]);
shading interp;

gmrf_data = dlmread('YR.dat');
gmrf_data2 = reshape(gmrf_data,m+2*NGhost,n+2*NGhost);
figure(8);
surfc(X,Y,gmrf_data2(1+NGhost:1:end-NGhost,1+NGhost:1:end-NGhost)');
view([0 90]);


end
