clc;
clear;

load ripley.mat

%{
figure;
hold on;
plot(Xtest(1:500,1),Xtest(1:500,2),'bo');
plot(Xtrain(1:125,1),Xtrain(1:125,2),'bo');
plot(Xtest(501:1000,1),Xtest(501:1000,2),'ro');
plot(Xtrain(126:250,1),Xtrain(126:250,2),'ro');
pbaspect([1 1 1])
xlabel('property 1')
ylabel('property 2')
hold off;
%}

Xtrain = Xtrain;
Ytrain = Ytrain;
Xtest = Xtest;
Ytest = Ytest;

%algr='simplex';
algr='gridsearch';
kernel='lin_kernel';

[ gam , kpar , cost ] = tunelssvm ({ Xtrain , Ytrain , 'c', [] , [] ,kernel} , algr, 'crossvalidatelssvm',{10 , 'misclass'}) ;

[ alpha , b ] = trainlssvm ({ Xtrain , Ytrain , 'c', gam , kpar , kernel}) ;

[ Yest , Ylatent ] = simlssvm ({ Xtrain , Ytrain ,  'c', gam , kpar , kernel}, {alpha , b} , Xtest ) ;

err = sum(Yest~=Ytest);

fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Ytest)*100)

roc(Ylatent, Ytest);

