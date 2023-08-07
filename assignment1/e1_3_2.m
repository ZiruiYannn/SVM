load iris
type='c'; 

disp('RBF kernel')
gamlist = [0.01, 0.1, 1, 10, 100, 1000, 10000]; sig2list=[0.001, 0.01, 0.1, 1, 10, 100];

errlist=[];
perflist=zeros(size(gamlist,2),size(sig2list,2));

for i=1:size(gamlist,2)
    gam=gamlist(i);
    for j=1:size(sig2list,2)
        sig2=sig2list(j);
        disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
        [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});
    
        % Plot the decision boundary of a 2-d LS-SVM classifier
        plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
    
        % Obtain the output of the trained classifier
        [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
        err = sum(Yht~=Ytest); errlist=[errlist; err];
        fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Ytest)*100)
        %perf = rsplitvalidate ({ Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'} , 0.80 , 'misclass') ;
        %perf=crossvalidate ({ Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'} , 10 , 'misclass');
        perf=leaveoneout  ({ Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'} ,  'misclass');
        
        perflist(i,j)=perf;
        fprintf('\n performance: %.2f% \n',perf)
        disp('Press any key to continue...'), pause,         
    end
end
