clc;
clear;

X1 = randn(50,2) +1;
X2 = randn(51,2) -1;

Y1 = ones(50,1);
Y2 = -ones(51,1);

disp(X1);

figure;
hold on;
plot(X1(:,1),X1(:,2),'ro');
plot(X2(:,1),X2(:,2),'bo');
hold off;

figure;
hold on;
plot(X1(:,1),X1(:,2),'ro');
plot(X2(:,1),X2(:,2),'bo');
plot([4,-4],[-4,4]);
hold off;