% load all images and save it in single files
% im: RGB, [0, 1], double
% imGt: 0 or 1
% imYuv: YUV, Y[0, 1], U[-0.436, 0.436], V[-0.615, 0.615], double
% the actions are performed seperately to save memory

% parameters
opt.imPath = '../data/person_png/';
opt.imGtPath = '../data/profile_png/';

imNum = 5389;

for i = 1 : imNum
%   imName = sprintf('%05d', i);
%   % im
%   im = imread([opt.imPath, imName, '.png']);
%   im = im2double(im);
%   % imGt
%   imGt = imread([opt.imGtPath, imName, '.png']);
%   imGt(imGt > 128) = 255;
%   imGt(imGt <= 128) = 0;
%   imGt = im2double(imGt);
%   % imYuv
%   imYuv = rgb2yuv(im);
%   % save
%   imSavePath = [opt.imPath, imName, '.mat'];
%   imGtSavePath = [opt.imGtPath, imName, '.mat'];
%   imYuvSavePath = [opt.imPath, imName, '.mat'];
%   save(imSavePath, 'im', '-v7.3');
%   save(imGtSavePath, 'imGt', '-v7.3');
%   save(imYuvSavePath, 'imYuv', '-v7.3');
%   % disp
%   disp(i);
end