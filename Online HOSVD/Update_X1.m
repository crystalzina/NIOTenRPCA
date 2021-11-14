function [ x,mode ] = Update_X1( x,mode,iter )

ff = 1;
bs = mode.basis_size;
us = mode.update_size;
X_all = mode.X_all;
n3 = size(X_all,3);


rank = mode.rank;
r1 = rank(1);
r2 = rank(2);
r3 = rank(3);

ndim = 3;

n1 = size(X_all,1);
n2 = size(X_all,2);

if(n3 == bs+us)
    mode.it = 0;
    X_all = mode.A;
%     DisplayVideo2(X_all,'a')
    sizeA = size(mode.A);
    mode.A = 0;
    n3 = sizeA(3);
%     X_all = mode.X_all;
    mode.U1 = mode.U1A ;
    mode.U2 = mode.U2A ;
    mode.U3 = mode.U3A;
    mode.D1 = mode.D1A;
    mode.D2 = mode.D2A;
    mode.D3 = mode.D3A;
    mode.V1 = mode.V1A;
    mode.V2 = mode.V2A;
    mode.V3 = mode.V3A; 
end

if (n3 == bs+us - 30 && iter == mode.maxIter )
    mode.it = 1;
    A = X_all(:,:,end-19:end);
    n4 = size(A,3);
    sizeD = [n1,n2,n4];
    X1 = Unfold( A,sizeD,1 );
    X2 = Unfold( A,sizeD,2 );
    X3 = Unfold( A,sizeD,3 );
    
    [U1,D1,V1] = lansvd(X1,r1,'L');
    [U2,D2,V2] = lansvd(X2,r2,'L');
    [U3,D3,V3] = lansvd(X3,r3,'L');
    
    mode.U1A = U1 ;
    mode.U2A = U2 ;
    mode.U3A = U3;
    mode.D1A = D1;
    mode.D2A = D2;
    mode.D3A = D3;
    mode.V1A = V1;
    mode.V2A = V2;
    mode.V3A = V3; 
    mode.A = A;
end

F = x;
Ten = X_all;

X_all(:,:,n3+1) =F;


U1 = mode.U1 ;
U2 = mode.U2 ;
U3 = mode.U3 ;
D1 = mode.D1 ;
D2 = mode.D2 ;
D3 = mode.D3 ;
V1 = mode.V1 ;
V2 = mode.V2 ;
V3 = mode.V3 ;


mt = 'true';
[U11,D11,V11] = addblock_svd_update( U1, D1, V1, F,ff, mt );
[P] = Transpose( n1,n3 );
[U22,D22,V22] = addblock_svd_update( U2, D2, V2, F', ff,mt );
V22 = P' * V22;
[U33,D33,V33] = addblock_svd_update2( U3, D3, V3, F(:)',ff, mt );
D33 = D33';

Ut{1} = U11(:,1:rank(1))';
Ut{2} = U22(:,1:rank(2))';
Ut{3} = U33(:,1:rank(3))';
U{1} = U11(:,1:rank(1));
U{2} = U22(:,1:rank(2));
U{3} = U33(:,1:rank(3));

sizeD1 = [n1,n2,n3+1];
Core =  my_ttm(X_all,Ut,1:ndim,sizeD1,rank,ndim);
X_all  = my_ttm(Core,U,1:ndim,rank,sizeD1,ndim);

x = X_all(:,:,end);
if iter ~= mode.maxIter
    mode.X_all = Ten;
else
    mode.U1 = U11 ;
    mode.U2 = U22;
    mode.U3 = U33;
    mode.D1 = D11;
    mode.D2 = D22;
    mode.D3 = D33;
    mode.V1 = V11;
    mode.V2 = V22;
    mode.V3 = V33;
    mode.X_all = X_all;
end



if mode.it == 1 && iter == mode.maxIter
    A = mode.A;
    n4 = size(A,3);
    A(:,:,n4+1) = F;
    U1A = mode.U1A ;
    U2A = mode.U2A ;
    U3A = mode.U3A ;
    D1A = mode.D1A ;
    D2A = mode.D2A ;
    D3A = mode.D3A ;
    V1A = mode.V1A ;
    V2A = mode.V2A ;
    V3A = mode.V3A ;
    mt = 'true';
    [U1A,D1A,V1A] = addblock_svd_update( U1A, D1A, V1A, F,ff, mt );
    [P] = Transpose( n1,n4 );
    [U2A,D2A,V2A] = addblock_svd_update( U2A, D2A, V2A, F', ff,mt );
    V2A = P' * V2A;
    [U3A,D3A,V3A] = addblock_svd_update2( U3A, D3A, V3A, F(:)',ff, mt );
    D3A = D3A';
    Ut{1} = U1A(:,1:rank(1))';
    Ut{2} = U2A(:,1:rank(2))';
    Ut{3} = U3A(:,1:rank(3))';
    U{1} = U1A(:,1:rank(1));
    U{2} = U2A(:,1:rank(2));
    U{3} = U3A(:,1:rank(3));
    
    sizeD1 = [n1,n2,n4+1];
    Core = my_ttm(A,Ut,1:ndim,sizeD1,rank,ndim);
    A  = my_ttm(Core,U,1:ndim,rank,sizeD1,ndim);
    mode.U1A = U1A ;
    mode.U2A = U2A ;
    mode.U3A = U3A;
    mode.D1A = D1A;
    mode.D2A = D2A;
    mode.D3A = D3A;
    mode.V1A = V1A;
    mode.V2A = V2A;
    mode.V3A = V3A;
    mode.A = A;
end

    
    
 
   

