clear;clc;

%%
X = (-6:0.2:6)';
Y = sinc(X) + 0.1.*rand(size(X));

% adding outliers
out = [15 17 19];
Y(out) = 0.7+0.3*rand(size(out));
out = [41 44 46];
Y(out) = 1.5+0.2*rand(size(out));

model = initlssvm(X ,Y, 'f', [], [], 'RBF_kernel');
costFun = 'crossvalidatelssvm';
model = tunelssvm ( model , 'simplex', costFun , {10 , 'mse';}) ;
figure
plotlssvm ( model ) ;



%%

model = initlssvm (X , Y , 'f', [] , [] , 'RBF_kernel') ;
costFun = 'rcrossvalidatelssvm';
wFun = 'wmyriad';
model = tunelssvm ( model , 'simplex', costFun , {10 , 'mse';} , wFun ) ;
model = robustlssvm ( model ) ;
figure
plotlssvm ( model ) ;



