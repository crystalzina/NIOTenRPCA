function[x,b,f,e,model,iter]= OnlineRPCA(model,y,par)
%Online Tensor-based RPCA for background subtraction
contpar= 0.95;
sizeD  = [par.imgsize, par.numFpri]; 
m      = model.compsize(1);
n      = model.compsize(2);
C      = par.A;      %compressive operator
tol    = 1e-4;
alpha  = par.alpha;
lambda = par.lambda;
mu1    = par.mu1;
mu2    = par.mu2;
mu3    = par.mu3;
%% Initialization
b       = model.b;       %background 
f_prior = model.fprior;  %foreground priors
f       = zeros(size(model.f));       %foreground
F_ini   = [f_prior;f];
fs      = diff3(F_ini,sizeD);
e       = zeros( n,1);         
T1      = zeros( 3*par.numFpri*n,1); 
T2      = zeros( par.numFpri*n,1);
T3      = zeros( m,1);
Sigma   = model.Sigma;
 
Eny_x   = ( abs(psf2otf([+1; -1], sizeD)) ).^2  ;
Eny_y   = ( abs(psf2otf([+1, -1], sizeD)) ).^2  ;
Eny_z   = ( abs(psf2otf([+1, -1], circshift(sizeD,[0,-1])))).^2  ;
Eny_z   =  permute(Eny_z, [3, 1, 2]);
denom1  =  Eny_x + Eny_y + Eny_z;

%% main outerloop
for iter=1:par.Oiter
       
    %% update x0
    T22= T2(end-n+1:end);
    Pre = mu3*(C'*(y-T3)) + mu2*(b+f+e+T22);
    Pre = Pre/mu2;
    x   = Pre-mu3/(mu3+mu2)*(C'*(C*Pre));
    Cx  = C*x; 

   %% update tensor basis 
    if iter==1 || iter==floor(par.Oiter/2) || iter==par.Oiter
       [b_tau,model]=Incre_HOOI_new(b,model,par,iter );
    end
     
   %% update b     
    b = (lambda*reshape(b_tau,prod(par.imgsize),[])+ mu2*(x-e-f-T22))/(lambda+mu2);
        
   %% update f
    temp_f   = x-b-e;
    F        = cat(1,f_prior,temp_f)-T2;
    diffT_p1 = diffT3( fs-T1 ,sizeD);
    numer1   = reshape( mu1*diffT_p1 + mu2*F, sizeD);
    f_mul1   = real( ifftn( fftn(numer1) ./ (mu1*denom1 + mu2) ) );
    f_mul    = f_mul1(:);
    f        = f_mul(end-n+1:end);
    f_td     = f_mul(1:end-n); 
       
    %% update fs
    diff_x2 = diff3(f_mul,sizeD); 
    fs      = softThres( diff_x2 + T1, alpha/mu1 );   
      
    %% update e
    ra = 1/Sigma^2+mu2;
    e  = mu2*(x-b-f-T22)/ra; 

    %% update sigma
    if iter<=5
       Sigma=sqrt((norm(e,2)+model.N*model.Sigma^2)/(n+model.N));    
    end
    
    %% update T1~T3
    T1=T1-(fs-diff3(F,sizeD));
    mid_t2 = cat(1,f_prior-f_td,x-b-e-f);
    T2=T2-mid_t2;
    T3=T3-(y-Cx);
    
    %% the stopping criterion 
     nr1 = norm(fs - diff_x2, 'fro');
     nr2 = norm(mid_t2, 'fro');
     nr3 = norm(y - Cx, 'fro');

     if iter >1 && nr1 > contpar * nr1_pre
        mu1 = par.ro*mu1;
     end
     if iter>1 && nr2 > contpar * nr2_pre
        mu2 = par.ro*mu2;
     end
     if iter>1 && nr3 > contpar * nr3_pre
        mu3 = par.ro*mu3;
     end    

    nr1_pre = nr1;    
    nr2_pre = nr2;    
    nr3_pre = nr3; 
    stopCond = norm(y-Cx)/norm(y);

    if (iter> 50) &&  (stopCond< tol) 
        break
    end
 end

%% update model priors
 model.b      = b;
 model.fprior = cat(1,f_prior(n+1:end),f);
 model.f      = f;
 model.Sigma  = Sigma;
 model.x      = x;
end

function x = softThres(a, tau)

x = sign(a).* max( abs(a) - tau, 0);
end




