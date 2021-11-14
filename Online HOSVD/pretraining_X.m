function [ model] = pretraining_X( X )
n1 = size(X,1);
n2 = size(X,2);

X_p = X;
p = size(X,3);
sizeD = [n1,n2,p];
X1 = Unfold( X_p,sizeD,1 );
X2 = Unfold( X_p,sizeD,2 );
X3 = Unfold( X_p,sizeD,3 );
[U1,D1,V1] = svd(X1,'econ');
[U2,D2,V2] = svd(X2,'econ');
[U3,D3,V3] = svd(X3,'econ');



model.U1=U1;
model.D1=D1;
model.V1=V1;

model.U2=U2;
model.D2=D2;
model.V2=V2;

model.U3=U3;
model.D3=D3;
model.V3=V3;

model.it = 0;
model.X_all = X_p;
end

