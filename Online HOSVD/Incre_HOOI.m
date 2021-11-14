function [ x,model ] = Incre_HOOI(x,model,par,iter )
x=reshape(x,par.imgsize);
ff = 1;
bs = par.bs;
us = par.us;
X_all = model.X_all;
n3 = size(X_all,3);  %%%%% n3 denotes the 3rd dimension of BASE TENSOR

rank = par.rank;
r1 = rank(1);
r2 = rank(2);
r3 = rank(3);

ndim = 3;

n1 = size(X_all,1);
n2 = size(X_all,2);
%%%%%%%%%%%%%% perform Incre_HOOI %%%%%%%%%%%%%%%%%%%%%%%%%%
if (n3 == bs+us - 20 && iter == par.Oiter )
    model.it = 1;
    A = X_all(:,:,end-19:end);
    n4 = size(A,3);
    sizeD = [n1,n2,n4];
    X1 = Unfold( A,sizeD,1 );
    X2 = Unfold( A,sizeD,2 );
    X3 = Unfold( A,sizeD,3 );
    
    [U1,D1,V1] = lansvd(X1,r1,'L');
    [U2,D2,V2] = lansvd(X2,r2,'L');
    [U3,D3,V3] = lansvd(X3,r3,'L');        
    
    model.U1A = U1 ;
    model.U2A = U2 ;
    model.U3A = U3;
    model.D1A = D1;
    model.D2A = D2;
    model.D3A = D3;
    model.V1A = V1;
    model.V2A = V2;
    model.V3A = V3; 
    model.A = A;
end

%%%%%%
if(n3 == bs+us)
    model.it = 0;
    X_all = model.A;
%     DisplayVideo2(X_all,'a')
    sizeA = size(model.A);
    model.A = 0;
    n3 = sizeA(3);
%     X_all = mode.X_all;
    model.U1 = model.U1A ;
    model.U2 = model.U2A ;
    model.U3 = model.U3A;
    model.D1 = model.D1A;
    model.D2 = model.D2A;
    model.D3 = model.D3A;
    model.V1 = model.V1A;
    model.V2 = model.V2A;
    model.V3 = model.V3A; 
end
%% perform incremental SVD along each mode and get new Tensor
%decompsition for new tensor
F = x;
Ten = X_all;

X_all(:,:,n3+1) =F;


U1 = model.U1 ;
U2 = model.U2 ;
U3 = model.U3 ;
D1 = model.D1 ;
D2 = model.D2 ;
D3 = model.D3 ;
V1 = model.V1 ;
V2 = model.V2 ;
V3 = model.V3 ;

%%%%%%%
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
x = x(:);
if iter ~= par.Oiter
    model.X_all = Ten;
else
    model.U1 = U11(:,1:r1);
    model.U2 = U22(:,1:r2);
    model.U3 = U33(:,1:r3);
    model.D1 = D11(1:r1,1:r1);
    model.D2 = D22(1:r2,1:r2);
    model.D3 = D33(1:r3,1:r3);
    model.V1 = V11(:,1:r1);
    model.V2 = V22(:,1:r2);
    model.V3 = V33(:,1:r3);
    model.X_all = X_all;
end



if model.it == 1 && iter == par.Oiter
    A = model.A;
    size(A)
    n4 = size(A,3);
    A(:,:,n4+1) = F;
    U1A = model.U1A ;
    U2A = model.U2A ;
    U3A = model.U3A ;
    D1A = model.D1A ;
    D2A = model.D2A ;
    D3A = model.D3A ;
    V1A = model.V1A ;
    V2A = model.V2A ;
    V3A = model.V3A ;
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
    U{1}  = U1A(:,1:rank(1));
    U{2}  = U2A(:,1:rank(2));
    U{3}  = U3A(:,1:rank(3));
    
    sizeD1 = [n1,n2,n4+1];
    Core = my_ttm(A,Ut,1:ndim,sizeD1,rank,ndim);
    A  = my_ttm(Core,U,1:ndim,rank,sizeD1,ndim);
    model.U1A = U1A(:,1:r1) ;
    model.U2A = U2A(:,1:r2) ;
    model.U3A = U3A(:,1:r3);
    model.D1A = D1A(:,1:r1);
    model.D2A = D2A(:,1:r2);
    model.D3A = D3A(:,1:r3);
    model.V1A = V1A(:,1:r1);
    model.V2A = V2A(:,1:r2);
    model.V3A = V3A(:,1:r3);
    model.A = A;
end

    
    
 
   

