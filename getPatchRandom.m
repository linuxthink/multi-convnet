function [ patch, label ] = getPatchRandom( opt, patchNum )
% function [ patch, label ] = getPatchRandom( allIm, allImGt, opt, patchNum )
%RANDOMGETPATCH Randomly get patch and label from allIm and allImGt
% Input
%   allIm           : each image must contain exactly 3 channels
%   allImGt         : ground truth image of 1 and 0
%   opt.patchDim    : the size of patches
%   opt.indRange    : [min, max]
%   opt.channel     : 'all', 'grayscale', 'first', 'second', 'third'
%                      if image is YUV, then 'grayscale' is meanningless
%   opt.scale       : e.g. [1, 2, 4], coorespond to scale 1, 2, 3
%   opt.scaleNum    : length(opt.scale)
%   patchNum        : number of patches, i.e. the minibatch size
% Output
%   patch           : range [0, 1], in double, strict format
%                     height * width * channelNum * scaleNum * patchNum
%   label           : either 1 or 2, in double, 1-based indexing
%                     1 -> background
%                     2 -> people

%% init patch, label
if strcmp(opt.channel, 'all')
  channelNum = 3;
elseif strcmp(opt.channel, 'grayscale') || ...
       strcmp(opt.channel, 'first') || ...
       strcmp(opt.channel, 'second') || ...
       strcmp(opt.channel, 'third')
  channelNum = 1;
else
  error('opt.channel error');
end
opt.scaleNum = length(opt.scale);
patch = zeros(opt.patchDim, opt.patchDim, channelNum, opt.scaleNum, ...
  patchNum);
label = zeros(1, patchNum);

%% get patch radomly
if opt.indRange(1) == opt.indRange(2)
  % optimize for single retrieval
  imInd = opt.indRange(1);
  imName = sprintf('%05d', imInd);
  load([opt.imPath, imName, '.mat']); % load im or imYuv
  load([opt.imGtPath, imName, '.mat']); % load imGt
  for patchInd = 1:patchNum
    % get pixelInd
    if ~isfield(opt, 'trainSelection') || strcmp(opt.trainSelection, 'all')
      % get normal random index
      pixelInd = randi(size(im, 1) * size(im, 2));
    else
      % get random index based on imMask
      if strcmp(opt.trainSelection, 'watershed_center')
        imMask = im2double(imread([opt.imPath, imName, ...
        '_watershed_center.png']));
      elseif strcmp(opt.trainSelection, 'watershed_boundary')
        imMask = im2double(imread([opt.imPath, imName, ...
          '_watershed_boundary.png']));
      elseif strcmp(opt.trainSelection, 'watershed_boundary_center')
        imSegCenter = im2double(imread([opt.imPath, imName, ...
        '_watershed_center.png']));
        imSegBoundary = im2double(imread([opt.imPath, imName, ...
          '_watershed_boundary.png']));
        imMask = imSegCenter + imSegBoundary;
      else
        error('error selection');
      end
      validInd = find(imMask == 1);
      pixelInd = validInd(randi(length(validInd)));
    end
    [patch(:, :, :, :, patchInd), label(patchInd)] = ...
      getPatchFromIm(im, imGt, opt.patchDim, opt.channel, opt.scale, ...
      [pixelInd, pixelInd], 'both');
    % print
    if mod(patchInd, 1000) == 0
      fprintf('random patch %d of %d\n', patchInd, patchNum);
    end
  end
else
  % retrive from multiple images
  for patchInd = 1:patchNum
    % random im, imGt
    imInd = randi(opt.indRange);
    imName = sprintf('%05d', imInd);
    % im = allIm{imInd};
    % imGt = allImGt{imInd};
    load([opt.imPath, imName, '.mat']); % load im or imYuv
    load([opt.imGtPath, imName, '.mat']); % load imGt
    % get pixelInd
    if ~isfield(opt, 'trainSelection') || strcmp(opt.trainSelection, 'all')
      % get normal random index
      pixelInd = randi(size(im, 1) * size(im, 2));
    else
      % get random index based on imMask
      if strcmp(opt.trainSelection, 'watershed_center')
        imMask = im2double(imread([opt.imPath, imName, ...
        '_watershed_center.png']));
      elseif strcmp(opt.trainSelection, 'watershed_boundary')
        imMask = im2double(imread([opt.imPath, imName, ...
          '_watershed_boundary.png']));
      elseif strcmp(opt.trainSelection, 'watershed_boundary_center')
        imSegCenter = im2double(imread([opt.imPath, imName, ...
        '_watershed_center.png']));
        imSegBoundary = im2double(imread([opt.imPath, imName, ...
          '_watershed_boundary.png']));
        imMask = imSegCenter + imSegBoundary;
      else
        error('error selection');
      end
      validInd = find(imMask == 1);
      pixelInd = validInd(randi(length(validInd)));
    end
    % get patch and label
    [patch(:, :, :, :, patchInd), label(patchInd)] = ...
      getPatchFromIm(im, imGt, opt.patchDim, opt.channel, opt.scale, ...
      [pixelInd, pixelInd], 'both');
    % print
    if mod(patchInd, 1000) == 0
      fprintf('random patch %d of %d\n', patchInd, patchNum);
    end
  end
end

end

