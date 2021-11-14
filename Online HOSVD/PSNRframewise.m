function PSNR_framewise = PSNRframewise( I,  I_denoised )

n = size(I,3);
n1 = size(I_denoised,3);

if n ~= 1
I = reshape(I,[],n);
end

if n1 ~= 1
I_denoised = reshape(I_denoised,[],n1);
end

for i = 1:size(I,2)
PSNR_framewise(i)=20*log10(255 * sqrt(numel(I_denoised(:,i))) / norm(I_denoised(:,i)-I(:,i)));
end

end

