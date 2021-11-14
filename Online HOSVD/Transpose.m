function [ P  ] = Transpose( n1,n3 )

n = n1*(n3+1);
A = zeros(n,2);
A(1:n,2) = 1:n;
for i = 1:n-n1
    it = fix((i-1)/n3) ;
        A(i,1) = i+it;
end
for i = 1:n1
        A(n-n1+i,1) = i*(n3+1);
end
A(n,1) = n;
P = sparse(A(:,1),A(:,2),1,n,n);

