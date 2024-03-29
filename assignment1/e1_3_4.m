load iris

algr='simplex';
%algr='gridsearch';

[ gam , sig2 , cost ] = tunelssvm ({ Xtrain , Ytrain , 'c', [] , [] ,'RBF_kernel'} , algr, 'crossvalidatelssvm',{10 , 'misclass'}) 

[ alpha , b ] = trainlssvm ({ Xtrain , Ytrain , 'c', gam , sig2 , 'RBF_kernel'}) ;


[ Yest , Ylatent ] = simlssvm ({ Xtrain , Ytrain ,  'c', gam , sig2 , 'RBF_kernel'}, {alpha , b} , Xtest ) ;

roc(Ylatent, Ytest);
