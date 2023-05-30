clc;
clear;
% Hand writing fuction : dft1,idft1,dft2,idft2,my_fftshift2,
%                        my_ifftshift2,my_mat2gray

% Load image  and add noise
% load clean image
img = imread('lena.jpg');
img = im2double(img);

% filter size
r = 30;
% add noise
% add_random_noise_points(img ,num of point , val of the noise point) 
noise_img_dots = add_random_noise_points(img,100,0); 
noise_img = add_noise(img,10);

figure();
subplot(1,3,1),imshow(img);title('original image')
subplot(1,3,2),imshow(noise_img);title('noisy image')
subplot(1,3,3),imshow(noise_img_dots);title('noisy image(dots)')
%% Fourier transform the original image
F_oriimg = dft2(img(:,:,1));
% center shift
F_oriimg_s = my_fftshift2(F_oriimg); 
% for displaying the power part by log scale
log_FToriimg = log(1+abs(F_oriimg_s)); 
% normalize the power to 0-1
log_FToriimg = my_mat2gray(log_FToriimg); 


figure();
subplot(4,2,1),imshow(log_FToriimg);title('log FT original image')

%% Implement Fourier transform on noisy images
% period noise
F_img = dft2(noise_img(:,:,1));
F_mag = my_fftshift2(F_img);
log_FTimage = log(1+abs(F_mag));
log_FTimage = my_mat2gray(log_FTimage);
% dots
F_img_d = dft2(noise_img_dots(:,:,1));
F_mag_d = my_fftshift2(F_img_d);
log_FTimage_d = log(1+abs(F_mag_d));
log_FTimage_d = my_mat2gray(log_FTimage_d);

subplot(4,2,2),imshow(log_FTimage);title('log FT noisy image')
subplot(4,2,5),imshow(log_FTimage_d);title('log FT noisy image(dots)')

%% Creat a filter
% get the size of the input image
[m, n, z] = size(img); 
% create a rectangular filter at center
filter = zeros(m,n);
filter(r:m-r,r:n-r) =1;

subplot(4,2,3),imshow(filter,[]);title('filter')
subplot(4,2,6),imshow(filter,[]);title('filter(dots)')

%% Filter out the denoised image
% FT image .* the filter
% period noise
fil_img = F_mag.*filter; 

log_fil_img = log(1+abs(fil_img));
log_fil_img = my_mat2gray(log_fil_img);
% dots
fil_img_d = F_mag_d.*filter; 

log_fil_img_d = log(1+abs(fil_img_d));
log_fil_img_d = my_mat2gray(log_fil_img_d);

subplot(4,2,4),imshow(log_fil_img);title('after filter FT image')
subplot(4,2,7),imshow(log_fil_img_d);title('after filter FT image (dots)')
%% Inverse Fourier transform
% unshift
fil_img = my_ifftshift2(fil_img); 
% display the denoised image

result = real(idft2(fil_img)); 
result = my_mat2gray(result);
result_rgb = three_channel_denoise(noise_img,r);
%dots
fil_img_d = my_ifftshift2(fil_img_d); 
result_d = real(idft2(fil_img_d)); 
result_d = my_mat2gray(result_d);
result_rgb_d = three_channel_denoise(noise_img_dots,r);

figure();
subplot(2,3,1),imshow(noise_img);title('noisy image')
subplot(2,3,2);imshow(result,[]);title('denoised image')
subplot(2,3,3);imshow(result_rgb);title('denoised RGBimage')
subplot(2,3,4),imshow(noise_img_dots);title('noisy image(dots)')
subplot(2,3,5);imshow(result_d,[]);title('denoised image(dots)')
subplot(2,3,6);imshow(result_rgb_d);title('denoised RGBimage(dots)')
%% function
% fft
function X = dft1(x)
    N = length(x);
    X = zeros(1, N);
    for k = 1:N
        for n = 1:N
            X(k) = X(k) + x(n)*exp(-1j*2*pi*(k-1)*(n-1)/N);
        end
    end
end
% ifft
function x = idft1(X)
    N = length(X);
    x = zeros(1, N);
    for n = 0:N-1
        for k = 0:N-1
            x(n+1) = x(n+1) + X(k+1)*exp(1j*2*pi*k*n/N);
        end
        x(n+1) = x(n+1) / N;
    end
end
% fft2
function X = dft2(x)
    [M, N] = size(x);
    X = zeros(M, N);
    for row = 1:M
        X(row, :) = dft1(x(row, :));
    end
    X = X.'; % Transpose for column-wise operation
    for col = 1:N
        X(col, :) = dft1(X(col, :));
    end
    X = X.'; % Transpose back to original orientation
end
% ifft2
function x = idft2(X)
    [M, N] = size(X);
    x = zeros(M, N);
    for row = 1:M
        x(row, :) = idft1(X(row, :));
    end
    x = x.'; % Transpose for column-wise operation
    for col = 1:N
        x(col, :) = idft1(x(col, :));
    end
    x = x.'; % Transpose back to original orientation
end
% fftshift
function Y = my_fftshift2(X)
    [M, N] = size(X);
    Y = zeros(M, N);
    Y(1:ceil(M/2), 1:ceil(N/2)) = X(floor(M/2)+1:end, floor(N/2)+1:end);
    Y(ceil(M/2)+1:end, 1:ceil(N/2)) = X(1:floor(M/2), floor(N/2)+1:end);
    Y(1:ceil(M/2), ceil(N/2)+1:end) = X(floor(M/2)+1:end, 1:floor(N/2));
    Y(ceil(M/2)+1:end, ceil(N/2)+1:end) = X(1:floor(M/2), 1:floor(N/2));
end
% ifftshift
function Y = my_ifftshift2(X)
    [M, N] = size(X);
    Y = zeros(M, N);
    Y(1:floor(M/2), 1:floor(N/2)) = X(ceil(M/2)+1:end, ceil(N/2)+1:end);
    Y(floor(M/2)+1:end, 1:floor(N/2)) = X(1:ceil(M/2), ceil(N/2)+1:end);
    Y(1:floor(M/2), floor(N/2)+1:end) = X(ceil(M/2)+1:end, 1:ceil(N/2));
    Y(floor(M/2)+1:end, floor(N/2)+1:end) = X(1:ceil(M/2), 1:ceil(N/2));
end
% mat2gray
function img = my_mat2gray(A)
    A = double(A); % Ensure A is a double precision matrix
    minA = min(A(:));
    maxA = max(A(:));
    rangeA = maxA - minA;
    
    % Normalize A to the range [0, 1]
    if rangeA > 0
        img = (A - minA) / rangeA;
    else
        img = zeros(size(A));
    end
end
% add impluse noise
function img = add_random_noise_points(img, num_points,val)
    [m, n, ~] = size(img);
    for i = 1:num_points
        % Generate random row and column indices
        row = randi([1, m], 1);
        col = randi([1, n], 1);
        
        % Set the corresponding pixel to black
        img(row, col, :) = val;
    end
end

function result = three_channel_denoise(noise_img,r)
    % get the size of the input image
    [m, n, z] = size(noise_img); 
    % create a rectangular filter at center
    filter = zeros(m,n);
    filter(r:m-r,r:n-r) =1;
    
    dim = length(size(noise_img));
    result = zeros(size(noise_img));
    % if input is rgb picture
    if dim == 3
        for i = 1:3
            % DFT
            F_img = dft2(noise_img(:,:,i));
            F_mag = my_fftshift2(F_img);
            % apply filter
            fil_img = F_mag.*filter; 
            % unshift
            fil_img = my_ifftshift2(fil_img); 
            % display the denoised image
            temp = real(idft2(fil_img)); 
            result(:,:,i) = my_mat2gray(temp);
        end

    elseif dim == 2
            F_img = dft2(noise_img);
            F_mag = my_fftshift2(F_img);
            % apply filter
            fil_img = F_mag.*filter; 
            % unshift
            fil_img = my_ifftshift2(fil_img); 
            % display the denoised image
            temp = real(idft2(fil_img)); 
            result = my_mat2gray(temp);
    else
        disp("Size error")
    end
end
