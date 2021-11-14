function [value] = QualityAss(ground_truth, estimated)    

%PSNR
Ori_H= ground_truth;
Denoi_HSI= estimated;
[M,N,B] = size(Ori_H);
T1 = reshape(Ori_H*255,M*N,B);
T2 = reshape(Denoi_HSI*255,M*N,B);
temp = reshape(sum((T1 -T2).^2),B,1)/(M*N);
PSNR = 20*log10(255)-10*log10(temp);
MPSNR = mean(PSNR);
value.psnr = MPSNR;

%SSIM
for i=1:B
[mssim, ~]=ssim_index(Ori_H(:,:,i)*255, Denoi_HSI(:,:,i)*255);
SSIM(i) = mssim;
end
MSSIM = mean(SSIM);
value.ssim = MSSIM;

% RMSE
sz_x = size(estimated);
if length(sz_x)==2
    n_bands=1;
else
    n_bands = sz_x(3);
end
n_samples = sz_x(1)*sz_x(2);

aux = sum(sum((estimated - ground_truth).^2, 1), 2)/n_samples;
rmse_total = sqrt(sum(aux, 3)/n_bands);
value.rmse=rmse_total;
end