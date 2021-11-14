function [model] = warmstart(X,par)
disp('Warm-start...');
r=1;
%% PCA and L1MF for warmstart
[U,V]=initi_OMoG(X,r);
L=U*V';
k = 2;
n = prod(par.imgsize);
L=reshape(L,[par.imgsize,size(L,2)]);
% calculate the SVD of flattend matrix along each mode
[ model  ] = pretraining_X(L(:,:,end-par.bs+1:end));
% calculate the model
noi=X-reshape(L,[n,size(L,3)]);
[~,label,~] = mogcluster(noi(:),k); 
Sigma =var(noi(:).*(label(:)~=1))*50*0.05; 
disp('Warm start is over. ');
%% output
model.b     = reshape(L(:,:,end),prod(par.imgsize),[]);
noi         = noi(:);
fprior      = noi.*(label==1)';
model.fprior= fprior(end-(par.numFpri-1)*n+1:end);
model.f     = fprior(end-n+1:end);
ee          = noi.*(label~=1)';
model.e     = ee(end-n+1:end);
model.Sigma = Sigma;


end
