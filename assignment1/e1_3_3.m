load iris

algr='simplex';
%algr='gridsearch';

[ gam , sig2 , cost ] = tunelssvm ({ Xtrain , Ytrain , 'c', [] , [] ,'RBF_kernel'} , algr, 'crossvalidatelssvm',{10 , 'misclass'}) 