% x = [1.1, 1.35, 1.25, 1.05];
% 
% H = [ 0.2 -0.2 -0.12 0.02;...
%       -0.2 1.4 0.02 0.0;...
%       -0.12 0.02 1 -0.4;...
%        0.02 0 -0.4 0.2];
% 
% r = 1.0;
% Aeq = ones(1,4);
% beq = 1;
% A = x;
% b = [r];
% 
% ret = quadprog(H,[],A,b,Aeq,beq);

    
[rs, ret] = solve();

ps = cell2mat(ret)';

x = [1.1, 1.35, 1.25, 1.05]'; 

H = [ 0.2 -0.2 -0.12 0.02;...
      -0.2 1.4 0.02 0.0;...
      -0.12 0.02 1 -0.4;...
       0.02 0 -0.4 0.2];
   
ms = ps * x;

vs = diag(ps * H * ps');

plot(rs,ms);
hold on;
plot(rs,vs);
title('return expectaion and variance vs. min expected return');
legend('expectation','variance');
xlabel('min. expected return');
ylabel('return value');
hold off;

%% additional plot for part b

plot(rs,ms);
hold on;
plot(rs,vs);
title('return expectaion and variance / portfolio fraction vs. min expected return');
xlabel('min. expected return');
ylabel('return value / portolio fraction');
plot(rs,ps(:,1));
plot(rs,ps(:,2));
plot(rs,ps(:,3));
plot(rs,ps(:,4));
legend('expectation','variance', 'IBM', 'Google', 'Apple', 'Intel');
hold off;






      
