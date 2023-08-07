clc;
clear;

%%

X = (-3:0.1:3)';
Y = sinc(X)+0.1.*randn(length(X),1);

Xtrain = X(1:2:end);
Ytrain = Y(1:2:end);
Xtest = X(2:2:end);
Ytest = Y(2:2:end);


%%
load gamsig2

sig2 = sig2list_sim(1);
gam = gamlist_sim(1);

crit_L1list=[];
crit_L2list=[];
crit_L3list=[];
gamlist=[];
siglist=[];
mselist=[]

%for i=1:10

crit_L1 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 } , 1) ;
crit_L2 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 } , 2) ;
crit_L3 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 } , 3) ;

[~ , sig2 ] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 } , 3) ;
[~ , gam ] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 } , 2) ;
[~ , alpha , b ] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 } , 1) ;


YtestEst = simlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'}, {alpha,b},Xtest);
sig2e = bay_errorbar ({ Xtrain , Ytrain , 'f', gam , sig2 } , 'figure') ;
%{
mse=immse(YtestEst,Ytest);
mselist=[mselist mse];

end
%}
%%

%sig2 = 0.01; gam = 1000; % Select pre-tuned hyperparameters
sig=0.15; gam=3; %0.0538=>0.0876

[alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'});
Ypred = simlssvm({Xtrain, Ytrain, 'f', gam, sig2, 'RBF_kernel'}, ...
    {alpha, b}, Xtest);
mse1 = mean((Ytest - Ypred).^2);


criterion_L1 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},1)
criterion_L2 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},2)
criterion_L3 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},3)


[~,sig2] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},3);
[~,gam] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},2);
[~,alpha,b] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},1);



% Compute error bars
sig2e = bay_errorbar({Xtrain,Ytrain,'f',gam,sig2},'figure');

Ypred = simlssvm({Xtrain, Ytrain, 'f', gam, sig2, 'RBF_kernel'}, ...
    {alpha, b}, Xtest);
mse = mean((Ytest - Ypred).^2);
disp(mse1) %0.0187
disp(mse) %0.0125







