clc; clear;

%%
load santafe.mat

%%
order = 10;
X = windowize (Z , 1:( order + 1) ) ;
Y = X (: , end ) ;
X = X (: , 1: order ) ;

gam = 10;
sig2 = 10;
[ alpha , b ] = trainlssvm ({ X , Y , 'f', gam , sig2 }) ;
 Xs = Z ( end - order +1: end , 1) ;

nb = 200;
prediction = predict ({ X , Y , 'f', gam , sig2 } , Xs , nb ) ;

figure ;
hold on;
plot ( Ztest , 'k') ;
plot ( prediction , 'r') ;
hold off;

%%

mean_Z = mean(Z);
std_Z=std(Z);
Z_processed = Z-mean_Z;

mselist=[];
costlist=[];

for order=10:300
    X = windowize (Z_processed , 1:( order + 1) ) ;
    Y = X (: , end ) ;
    X = X (: , 1: order ) ;
    Xtrain = X(1:600,:);
    Xtest=X(601:end, :);
    Ytrain = Y(1:600,:);
    Ytest=Y(601:end, :);

    algr='simplex';


    [ gam , sig2 , cost ] = tunelssvm ({ X , Y , 'f', [] , [] ,'RBF_kernel'} , algr, 'crossvalidatelssvm',{10 , 'mae'}) 
    costlist=[costlist, cost];

    [ alpha , b ] = trainlssvm ({ X , Y , 'f', gam , sig2 }) ;
    Yest = simlssvm ({ X , Y ,  'f', gam , sig2 , 'RBF_kernel'}, {alpha , b} , X ) ;
       mse=immse(Yest,Y);
       mselist=[mselist mse];
    
end

[value, local]=min(mselist)

%%
order = 50;

mean_Z = mean(Z);
std_Z=std(Z);
Z = Z-mean_Z;
X = windowize (Z , 1:( order + 1) ) ;
Y = X (: , end ) ;
X = X (: , 1: order ) ;

gam = 10;
sig2 = 10;
[ alpha , b ] = trainlssvm ({ X , Y , 'f', gam , sig2 }) ;
 Xs = Z ( end - order +1: end , 1) ;

nb = 200;
prediction = predict ({ X , Y , 'f', gam , sig2 } , Xs , nb ) + mean_Z;

figure ;
hold on;
plot ( Ztest , 'k') ;
plot ( prediction , 'r') ;
hold off;

mse=immse(Ztest, prediction)
%%

figure
plot(Z)
figure
plot(Ztest)