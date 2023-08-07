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
gam = 10;
sig2 = 0.3;
type = 'function estimation';
[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});


%[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel','original'});
%[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'});

Xt = 3.*randn(10,1);
 
Yt = simlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xt);

figure

plotlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
%plotlssvm({X,Y,type,gam,sig2,'RBF_kernel'},{alpha,b});

%hold on; plot(min(X):.1:max(X),sinc(min(X):.1:max(X)),'r-.');

%%

sig2list = [0.01 1 100];
gamlist=[10 1000 1000000];

for sig2=sig2list
    for gam=gamlist

        [alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'});

        YtestEst = simlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'}, {alpha,b},Xtest);
        
        figure
        hold on;
        plotlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},{alpha,b});
        hold on;
        plot(Xtest,Ytest,'.', 'MarkerSize', 15);
        plot(Xtest,YtestEst,'r-+', 'LineWidth', 2);

        legend('resulting funtion','Ytest','YtestEst');
        title(sprintf('gam =%g sig2 =%g',gam, sig2));
        hold off
        mse=immse(YtestEst,Ytest);
        fprintf('gam =%g sig2 =%g mse=%g\n',gam, sig2, mse);
    end
end

%%
algr='simplex';
%algr='gridsearch';

gamlist_sim=[];
sig2list_sim=[];
gamlist_grid=[];
sig2list_grid=[];

for i=1:100 
[ gam , kpar , cost ] = tunelssvm ({ Xtrain , Ytrain , 'f', [] , [] ,'RBF_kernel'} , algr, 'crossvalidatelssvm',{10 , 'mse'}) ;

sig2=kpar;
gamlist_sim=[gamlist_sim gam];
sig2list_sim=[sig2list_sim sig2];
end

algr='gridsearch';

for i=1:100 
[ gam , kpar , cost ] = tunelssvm ({ Xtrain , Ytrain , 'f', [] , [] ,'RBF_kernel'} , algr, 'crossvalidatelssvm',{10 , 'mse'}) ;

sig2=kpar;
gamlist_grid=[gamlist_grid gam];
sig2list_grid=[sig2list_grid sig2];
end

save("gamsig2.mat","gamlist_sim","sig2list_sim","gamlist_grid","sig2list_grid");

%%
load gamsig2



figure

histogram(gamlist_grid,[0:10:2000],'FaceColor','b')
hold on
histogram(gamlist_sim,[0:10:2000],'FaceColor','y')
axis([0 800 0 60])

load gamsig2

figure

histogram(sig2list_grid,[0:0.02:0.8],'FaceColor','b')
hold on
histogram(sig2list_sim,[0:0.02:0.8],'FaceColor','y')





