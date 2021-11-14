function [ x,model ] = Incre_HOOI_new(x,model,par,iter )
x  = reshape(x,par.imgsize);
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
%% perform incremental SVD along each mode and get new Tensor
%decompsition for new tensor
F = x;
X_all(:,:,n3+1) = F;


U1 = model.U1 ;
U2 = model.U2 ;
U3 = model.U3 ;
D1 = model.D1 ;
D2 = model.D2 ;
D3 = model.D3 ;
V1 = model.V1 ;
V2 = model.V2 ;
V3 = model.V3 ;

%%%%%Incremental Updating of Tensor Decomposition
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

if iter==par.Oiter
    if n3~=bs+us
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
    else
    A = X_all(:,:,end-bs+1:end);
    n4 = size(A,3);
    sizeD = [n1,n2,n4];
    X1 = Unfold( A,sizeD,1 );
    X2 = Unfold( A,sizeD,2 );
    X3 = Unfold( A,sizeD,3 );
    
    [U1,D1,V1] = lansvd(X1,r1,'L');
    [U2,D2,V2] = lansvd(X2,r2,'L');
    [U3,D3,V3] = lansvd(X3,r3,'L');        
    
    model.U1 = U1 ;
    model.U2 = U2 ;
    model.U3 = U3;
    model.D1 = D1;
    model.D2 = D2;
    model.D3 = D3;
    model.V1 = V1;
    model.V2 = V2;
    model.V3 = V3; 
    model.X_all = A;



    end
end

end