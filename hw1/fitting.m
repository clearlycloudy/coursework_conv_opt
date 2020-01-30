x2 = x.^2
x1 = x
x0 = ones(length(x),1)
X = [x2 x2 x0]
v=inv(X'*X)*X'*y
%solve: (X'*X) v = (X'*y)
v = (X'*X)\(X'*y)
v=pinv(X'*X)*(X'*y)

syms f(xx);
f(xx) = v(1)*xx^2 + v(2)*xx + v(3)

domain = linspace(-300,300,600)
fit = f(domain)

scatter(x,y,'+')
hold on;
scatter(domain,fit)

x4 = x.^4
x3 = x.^3
x2 = x.^2
A = [ sum(x4) sum(x3) sum(x2);...
      sum(x2) sum(x) length(x);...
      sum(x3) sum(x2) sum(x) ]
B = [sum(x2.*y); sum(y); sum(x.*y)]
v = A\B

v = linprog([],[],[],A,B)

syms f(xx);
f(xx) = v(1)*xx^2 + v(2)*xx + v(3)

domain = linspace(-300,300,600)
fit = f(domain)

scatter(x,y,'+')
hold on;
plot(domain,fit)
xlabel('x')
ylabel('y')
