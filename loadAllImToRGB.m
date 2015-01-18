% load all images and save it in single files
% im: RGB, [0, 1], double
% imGt: 0 or 1
% imYuv: YUV, Y[0, 1], U[-0.436, 0.436], V[-0.615, 0.615], double
% the actions are performed seperately to save memory

% parameters
opt.imPath = '../data/person_png/';
opt.imGtPath = '../data/profile_png/';
opt.imRGBMatPath = '../data/person_rgb/';
opt.imRGBNormMatPath = '../data/person_rgb_norm/';

imNum = 5389;
for i = 1 : imNum
  imName = sprintf('%05d', i);
  % im
  im = imread([opt.imPath, imName, '.png']);
  im = im2double(im);
  % save
  save([opt.imRGBMatPath, imName, '.mat'], 'im', '-v7.3');
  % disp
  disp(i);
end

imNum = 5389;
for i = 1 : imNum
  imName = sprintf('%05d', i);
  % im
  im = imread([opt.imPath, imName, '.png']);
  im = im2double(im);
  im = imPreProcess(im);
  % save
  save([opt.imRGBNormMatPath, imName, '.mat'], 'im', '-v7.3');
  % disp
  disp(i);
end


