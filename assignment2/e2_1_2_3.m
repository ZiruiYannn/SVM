clc;
clear;


%%

load gamsig2.mat

sig2 = sig2list_sim(1);
gam = gamlist_sim(1);

X = 6.* rand (100 , 3) - 3;
Y = sinc ( X (: ,1) ) + 0.1.* randn (100 ,1) ;
[ selected , ranking ] = bay_lssvmARD ({ X , Y , 'f', gam , sig2 }) ;

figure
plot(X(:,1),Y,'o')

figure
plot(X(:,2),Y,'o')

figure
plot(X(:,3),Y,'o')