% load all images and save it in mat, globally normalized version
% im: RGB, [0, 1], double
% imGt: 0 or 1
% imYuv: YUV, Y[0, 1], U[-0.436, 0.436], V[-0.615, 0.615], double
% the actions are performed seperately to save memory

% parameters
opt.imPath = '../data/person_height_320/';
opt.imGtPath = '../data/profile_height_320/';
opt.imNormPath = '../data/person_height_320_norm/';
imNum = 5389;

for i = 4750 : imNum
%   imName = sprintf('%05d', i);
%   imYuvPath = [opt.imPath, imName, '.mat'];
%   load(imYuvPath);
%   imYuv = imPreProcess(imYuv);
%   imYuvNormPath = [opt.imNormPath, imName, '.mat'];
%   save(imYuvNormPath, 'imYuv', '-v7.3');
%   disp(i);
end