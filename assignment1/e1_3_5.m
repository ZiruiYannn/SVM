
load iris

algr='simplex';
%algr='gridsearch';

%[ gam , sig2 , cost ] = tunelssvm ({ Xtrain , Ytrain , 'c', [] , [] ,'RBF_kernel'} , algr, 'crossvalidatelssvm',{10 , 'misclass'}) 

gam=300 %2.8045
sig2=0.2140  %0.2140

bay_modoutClass ({ Xtrain , Ytrain , 'c', gam , sig2 } , 'figure') ;
colorbar;