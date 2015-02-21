function [ yuv ] = rgb2yuv( rgb )
%RGB2YUV Convert RGB image to YUV
% Input
%   rgb: [0, 1], double
% Output
%   yuv: y[0, 1], u[-0.436, 0.436], v[-0.615, 0.615], double

rgb = double(rgb);
assert(max(rgb(:))<=1 && min(rgb(:))>=0, 'rgb must be [0, 1]');

r = rgb(:,:,1);
g = rgb(:,:,2);
b = rgb(:,:,3);

y = 0.299 * r + 0.587 * g + 0.114 * b;
u = -0.14713 * r - 0.28886 * g + 0.436 * b;
v = 0.615 * r - 0.51499 * g - 0.10001 * b;

yuv = cat(3, y, u, v);

end