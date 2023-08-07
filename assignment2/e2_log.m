clc; clear;

%%
load logmap.mat

%%
order = 10;
X = windowize (Z , 1:( order + 1) ) ;
Y = X (: , end ) ;
X = X (: , 1: order ) ;

gam = 10;
sig2 = 10;
[ alpha , b ] = trainlssvm ({ X , Y , 'f', gam , sig2 }) ;
 Xs = Z ( end - order +1: end , 1) ;

nb = 50;
prediction = predict ({ X , Y , 'f', gam , sig2 } , Xs , nb ) ;

figure ;
hold on;
plot ( Ztest , 'k') ;
plot ( prediction , 'r') ;
hold off;

%%

Ztest_smooth=smoothdata(Ztest,'gaussian',2);
figure
plot(Ztest_withoutnoise,'y')
hold on
plot(Ztest_smooth,'r')
plot(Ztest,'b')

%%

mean_Z = mean(Z);
Z_processed = smoothdata(Z - mean_Z,'gaussian',2);
Z_processed = Z;

mselist=[];
costlist=[];

for order=2:30
    X = windowize (Z_processed , 1:( order + 1) ) ;
    Y = X (: , end ) ;
    X = X (: , 1: order ) ;

    algr='simplex';

    [ gam , sig2 , cost ] = tunelssvm ({ X , Y , 'f', [] , [] ,'RBF_kernel'} , algr, 'crossvalidatelssvm',{10 , 'mse'}) 
    costlist=[costlist, cost];
    disp(['gam : ', gam, '   sig2 : ', sig2]),
        
    [alpha,b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel'});
    
    Ztest_processed = smoothdata(Ztest - mean_Z,'gaussian',2);
    Xtest = windowize (Ztest_processed , 1:( order + 1) ) ;
    Ytest = Xtest (: , end ) ;
    Xtest = Xtest (: , 1: order ) ;
    [Yest, Zt] = simlssvm({X,Y,'f',gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);

    mse=immse(Yest+mean_Z,Ztest((order+1):end));
    mselist=[mselist,mse];
    fprintf('\n mse = %.2f% cost = %.2f% \n', mse, cost);
end

%%

mean_Z = mean(Z);
Z_processed = smoothdata(Z - mean_Z,'gaussian',2);

order=4;

X = windowize (Z_processed , 1:( order + 1) ) ;
Y = X (: , end ) ;
X = X (: , 1: order ) ;

algr='simplex';

[ gam , sig2 , cost ] = tunelssvm ({ X , Y , 'f', [] , [] ,'RBF_kernel'} , algr, 'crossvalidatelssvm',{10 , 'mse'}) 

Xs = Z ( end - order +1: end , 1) ;

nb = 50;
prediction = predict ({ X , Y , 'f', gam , sig2 } , Xs , nb ) ;

figure ;
hold on;
plot ( Ztest , 'k') ;
plot ( prediction , 'r') ;
hold off;

[alpha,b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel'});
Ztest_processed = smoothdata(Ztest - mean_Z,'gaussian',2);
Xtest = windowize (Ztest_processed , 1:( order + 1) ) ;
Ytest = Xtest (: , end ) ;
Xtest = Xtest (: , 1: order ) ;
[Yest, Zt] = simlssvm({X,Y,'f',gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);

Zest = Yest + mean_Z;

figure
hold on
plot(Ztest((order+1):end),'b')
plot(Zest,'r')
hold off

%%

mean_Z = mean(Z);
std_Z=std(Z);
Z_processed = Z-mean_Z;

mselist=[];
costlist=[];

for order=2:40
    X = windowize (Z_processed , 1:( order + 1) ) ;
    Y = X (: , end ) ;
    X = X (: , 1: order ) ;

    Xtrain = X(1:60,:);
    Ytrain = Y(1:60);
    Xtest = X(61:end,:);
    Ytest = Y(61:end);

    algr='gridsearch';


    [ gam , sig2 , cost ] = tunelssvm ({ Xtrain , Ytrain , 'f', [] , [] ,'RBF_kernel'} , algr, 'crossvalidatelssvm',{10 , 'mae'}) 
    costlist=[costlist, cost];

    [ alpha , b ] = trainlssvm ({ Xtrain , Ytrain , 'f', gam , sig2 }) ;

    Xs = Xtrain ( end - order +1: end , 1) ;

nb = length(Ytest);
prediction = predict ({ Xtrain , Ytrain , 'f', gam , sig2 } , Xs , nb ) ;

    %Yest = simlssvm ({ Xtrain, Ytrain ,  'f', gam , sig2 , 'RBF_kernel'}, {alpha , b} , Xtest ) ;
       mse=immse(prediction,Ytest);
       %mse=immse(Yest,Ytest);
       mselist=[mselist mse];
    
end

[value, local]=min(mselist)

%%
order = 24;
X = windowize (Z , 1:( order + 1) ) ;
Y = X (: , end ) ;
X = X (: , 1: order ) ;

gam = 10;
sig2 = 10;
[ alpha , b ] = trainlssvm ({ X , Y , 'f', gam , sig2 }) ;
 Xs = Z ( end - order +1: end , 1) ;

nb = 50;
prediction = predict ({ X , Y , 'f', gam , sig2 } , Xs , nb ) ;

figure ;
hold on;
plot ( Ztest , 'k') ;
plot ( prediction , 'r') ;
hold off;

%%
mean_Z = mean(Z);
ZZ = Z - mean_Z;

mse=zeros(39,1);

for order=2:40
    fprintf('order=%g',order)

    X = windowize (ZZ , 1:( order + 1) ) ;
    Y = X (: , end ) ;
    X = X (: , 1: order ) ;
    
    for i=1:10
        if i==1
            Xtrain=X(15:end,:);
            Ytrain=Y(15:end,1)
        elseif i==10
            Xtrain=X(1:126,:);
            Ytrain=Y(1:126,1)
        else
            Xtrain=X([1:14*(i-1) ((i*14)+1):end],:);
            Ytrain=Y([1:14*(i-1) ((i*14)+1):end],1);
        end
        Xtest=X((1+(i-1)*14):(i*14),:);
        Ytest=Y((1+(i-1)*14):(i*14),1);


        algr='simplex';
        [ gam, sig2 ,cost ] = tunelssvm ({ Xtrain , Ytrain , 'f', [] , [] ,'RBF_kernel'} , algr, 'crossvalidatelssvm',{10 , 'mse'}) 
        
        [ alpha , b ] = trainlssvm ({ Xtrain , Ytrain , 'f', gam , sig2 , 'RBF_kernel'}) ;

        Yest = simlssvm ({ Xtrain , Ytrain ,  'f', gam , sig2 , 'RBF_kernel'}, {alpha , b} , Xtest ) ;
        mse(order)=mse(order) + immse(Yest,Ytest);
    end

end

[value, local]=min(mse);
