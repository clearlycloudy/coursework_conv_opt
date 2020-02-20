function [rs, s] = solve()

    x = [1.1, 1.35, 1.25, 1.05];

    H = [ 0.2 -0.2 -0.12 0.02;...
          -0.2 1.4 0.02 0.0;...
          -0.12 0.02 1 -0.4;...
           0.02 0 -0.4 0.2];

    Aeq = ones(1,4);
    beq = 1;
    
    A = -x;

    rs = 0.5:0.01:1.35
    
    lb = zeros(4,1);
    ub = ones(4,1);
    s = arrayfun(@(b) quadprog(H,[],A,-b,Aeq,beq,lb,ub),rs, 'UniformOutput',false);
    
end