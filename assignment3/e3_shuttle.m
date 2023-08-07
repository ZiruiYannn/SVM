data = load('shuttle.dat','-ascii'); function_type = 'c';  data = data(1:100,:);

X = data(:,1:end-1);
Y = data(:,end);

Y(Y == 1) = 1;
Y(Y ~= 1) = -1;

testX = [];
testY = [];

for i=1:9
    subplot(3,3,i)
    for j=1:length(Y)
        if Y(j) == 1
            plot(X(j,i),Y(j),'ro')
            hold on
        else
            plot(X(j,i),Y(j),'b+')
            hold on
        end
    end
    axis('auto')
    title(sprintf('attribute %g',i))
    hold off
end