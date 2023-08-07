data = load('california.dat','-ascii'); function_type = 'f'; data = data(1:100,:);

X = data(:,1:end-1);
Y = data(:,end);


for i=1:8
    subplot(3,3,i)
    plot(X(:,i),Y(:),'.')
    axis('auto')
    title(sprintf('attribute %g',i))
    hold off
end