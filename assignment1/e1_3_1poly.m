load iris.mat

gam=1;
t=1;
degree=12;
kpar=[t; degree];
[ alpha , b ] = trainlssvm ({ Xtrain , Ytrain , 'c', gam ,  kpar,'poly_kernel'});

figure;
plotlssvm ({ Xtrain , Ytrain , 'c', gam , kpar , 'poly_kernel'} , { alpha ,b }) ;

Yest = simlssvm ({ Xtrain , Ytrain , 'c' , gam , kpar , 'poly_kernel'} , {alpha , b } , Xtest ) ;

count=0;

for i = 1: size(Ytest,1)
    if Yest(i,1) == Ytest(i,1)
        count=count+1;
    end
end

rate=count/(size(Ytest,1))
print rate;
