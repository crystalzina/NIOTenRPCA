clear
clc
currentFolder = pwd;
addpath(genpath(currentFolder))

dname={'office','peopleInShade','Curtain'};
ratioset=[0.8,0.6,0.4,0.2];   
K  = 3; % dataset index
SR = 1; % sampling ratio index
load([currentFolder filesep dname{K}])
X=X(:,:,1:200); 
%% parameter setting 
par.lambda  = 5;       % imposed on the background
par.alpha   = 3e-3;    % imposed on the continuous foreground
par.mu1     = 5e-2;    % within ADMM fs=Df
par.mu2     = 1;       % x=b+e+f;
par.mu3     = 2;       % y=Ax    Shoppping Mall

h=size(X,1); 
w=size(X,2);
par.rank    = [ceil(h*0.85),ceil(w*0.85),1];  % tensor rank
par.bs      = 20;      % the number of prior frame for background
par.us      = 40;      
par.ro      = 1.15;   % within ADMM
par.Oiter   = 100;     % loop numbers of the algorithm
par.numFpri = 2;       % the frame number of online foreground volume
par.PriLen  = 20;

imgsize     = [h,w];
downratio   = ratioset(SR);
n           = prod(imgsize);
m           = ceil(n*downratio);
par.imgsize = imgsize;

%% warmstart
DataTrain = reshape(Train,[n,size(Train,3)]);
[nmodel] = warmstart(DataTrain,par);

%% running OMoGMF
model=nmodel;
model.N = par.PriLen*n;    
model.compsize=[m,n];
tic
for i=1:size(X,3)
    
    A =  PermuteWHT2(n, 1, downratio);
    par.A=A;    
    CompData=A*reshape(X(:,:,i),n,[]);
    [x,b,f,e,model]= OnlineRPCA(model,CompData,par);
    X0(:,i)=x;
    B(:,i) =b;
    F(:,i) =f;   
end
toc
Matric = QualityAss(X,reshape(X0,[imgsize,size(X,3)]))
