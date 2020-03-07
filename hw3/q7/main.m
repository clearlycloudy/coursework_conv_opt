
%% part a
f = [zeros(10,1); ones(10,1)]
A = [eye(10) -eye(10); -eye(10) -eye(10)]
b= [zeros(20,1)]

Aeq = [ones(1,10) zeros(1,10); 10:-1:1 zeros(1,10)]
beq = [0;1]
[xs,fval,exitflag,output,lambda] = linprog(f,A,b,Aeq,beq)

force = xs(1:10)
plot(1:10,force)
title('force vs t')

v = zeros(11,1)
for i=2:1:11
    v(i) = v(i-1) + force(i-1)
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

%% part c
f = [zeros(10,1); 1]
A = [eye(10) -ones(10,1); -eye(10) -ones(10,1)]
b= [zeros(20,1)]

Aeq = [ones(1,10) 0; 10:-1:1 0]
beq = [0;1]
[xs,fval,exitflag,output,lambda] = linprog(f,A,b,Aeq,beq)

force = xs(1:10)
plot(1:10,force)
title('force vs t')

v = zeros(11,1)
for i=2:1:11
    v(i) = v(i-1) + force(i-1)
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

