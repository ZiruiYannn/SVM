load iris.mat

gam=3
sig2=3;
kpar=sig2;
[ alpha , b ] = trainlssvm ({ Xtrain , Ytrain , 'c', gam ,  kpar,'RBF_kernel'});

figure;
plotlssvm ({ Xtrain , Ytrain , 'c', gam , kpar , 'RBF_kernel'} , { alpha ,b }) ;

Yest = simlssvm ({ Xtrain , Ytrain , 'c' , gam , kpar , 'RBF_kernel'} , {alpha , b } , Xtest ) ;

count=0;

for i = 1: size(Ytest,1)
    if Yest(i,1) == Ytest(i,1)
        count=count+1;
    end
end

rate=count/(size(Ytest,1))
print rate;
