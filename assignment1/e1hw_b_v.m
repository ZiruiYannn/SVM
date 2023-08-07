
clear;

load breast.mat

%{
figure;
hold on;
for i=1:size(labels_test(1:60,:), 1)
    if labels_test(i,1)==1
        parallelcoords(testset(i,:),'color','r')
    else
        parallelcoords(testset(i,:),'color','b')
    end
end

axis([1,30, 1,3000])
xlabel('properties')
ylabel('value')
hold off;

%}


Xtrain = [trainset(:,1:4) trainset(:,13:14) trainset(:,21:24)];
Ytrain = labels_train;
Xtest = [testset(:,1:4) testset(:,13:14) testset(:,21:24)];
Ytest = labels_test;

algr='simplex';
%salgr='gridsearch';
kernel='RBF_kernel';

[ gam , kpar , cost ] = tunelssvm ({ Xtrain , Ytrain , 'c', [] , [] ,kernel} , algr, 'crossvalidatelssvm',{10 , 'misclass'}) ;

[ alpha , b ] = trainlssvm ({ Xtrain , Ytrain , 'c', gam , kpar , kernel}) ;

[ Yest , Ylatent ] = simlssvm ({ Xtrain , Ytrain ,  'c', gam , kpar , kernel}, {alpha , b} , Xtest ) ;

err = sum(Yest~=Ytest);

fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Ytest)*100)

roc(Ylatent, Ytest);