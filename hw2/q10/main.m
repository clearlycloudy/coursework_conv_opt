%% part a
n = length(h);
m = 20;

% setup convolution matrix
e = zeros(m+2*n,n);
for i=1:10
    e(:,i) = ones(m+2*n,1) * h(i);    
end
A = full(spdiags(e, 0:9, m+n+1, m+2*n));

% row index for conv(f,g)[t=0]
t_0 = size(A,1)-1;

gs = [];
losses = [];
ds = [];
% solve for different D offset from t_0 using QP
for D=-(m+n-1):1
    fprintf('d: %d\n',-D);
    A2 = [ A(1:t_0+D-1,:);
           zeros(1,m+2*n);
           A(t_0+D+1:end,:) ];
    H = A2'*A2;

    Aeq = A(t_0+D,:);
    beq = 1;
    b = zeros(8,1);
    lb = [zeros(n,1); ones(m,1) * -Inf; zeros(n,1)];
    ub = [zeros(n,1); ones(m,1) * Inf; zeros(n,1)];
    f = [];
    X = quadprog(H,f,[],[],Aeq,beq,lb,ub);
    if length(X) ~= 0 % guard for feasible answer
        g = flip(X(n+1:n+1+m-1)); % flip due to convolution
        loss = X'*H*X;
        losses = [ losses; loss];
        gs = [gs; g'];
        ds = [ds; -D];
    end
end

% obtain the best answer for g
[~,idx] = sort(losses(:,1));
losses_sorted = losses(idx,:);
gs_sorted = gs(idx,:);
ds_sorted = ds(idx,:);
g_best = gs_sorted(1,:);
d_best = ds_sorted(1,:);
fprintf("D best: %d\n", d_best);

temp = conv(g_best,h);
assert(abs(temp(d_best+1)-1.0)<1e-15);
stem([0:length(temp)-1], temp);
title('conv(g,h)');

stem([0:length(g_best)-1], g_best);
title('deconvolution filter');

% verify on arbitrary data
samples = rand(10000,1);
output = conv(g_best, conv(h, samples));
[r,lags] = xcorr(samples,output);
[~,i]=max(r);
l = -lags(i);
assert(l==d_best);

%% part b
hist(y);
title('histogram of y');
z=conv(y,g_best,'same');
hist(z);
title('histogram of conv(y,g)');

% g_optimal

    0.0664
    0.1949
    0.4076
    0.6687
    0.9353
    1.0797
    0.8390
   -0.2200
    0.5814
    0.3494
    0.5823
    0.0069
    0.0187
    0.1924
    0.2728
    0.0987
   -0.0172
    0.0044
    0.1194
    0.0786