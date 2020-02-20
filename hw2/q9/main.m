
%% part a
H = diag(ones(10,1))
f = []
A = []
b= []
Aeq = [ones(1,10); 10:-1:1]
beq = [0;1]
f = quadprog(H,f,A,b,Aeq,beq)

plot(1:10,f)
title('force vs t')

v = zeros(11,1)
for i=2:1:11
    v(i) = v(i-1) + f(i-1)
end

plot(0:1:10,v)
title('velocity vs t')

x = zeros(11,1)
for i=1:1:11
    if i > 1
        x(i) = x(i-1) + v(i)
    else
        x(i) = v(i)
    end
end

plot(0:1:10,x)
title('displacement vs t')

%% part b
H = diag(ones(10,1))
f = []
A = []
b= []
Aeq = [ones(1,10); 10:-1:1; 5 4 3 2 1 0 0 0 0 0]
beq = [0;1;0]
f = quadprog(H,f,A,b,Aeq,beq)

plot(1:10,f)
title('force vs t')

v = zeros(11,1)
for i=2:1:11
    v(i) = v(i-1) + f(i-1)
end

plot(0:1:10,v)
title('velocity vs t')

x = zeros(11,1)
for i=1:1:11
    if i > 1
        x(i) = x(i-1) + v(i)
    else
        x(i) = v(i)
    end
end

plot(0:1:10,x)
title('displacement vs t')