function [ patch, label ] = ...
  getPatchFromIm(im, imGt, patchDim, channel, scale, range, mode)
%GETPATCHCOMPLETE Get (complete) patch from an image
% Input
%   im            : the must contain exactly 3 channels, in double
%   imGt          : binary groundtruth image, in double, from [0, 1]
%   patchDim      : the size of patches
%   channel       : 'all', 'grayscale', 'first', 'second', 'third'
%                   if image is YUV, then 'grayscale' is meanningless
%   scale         : e.g. [1, 2, 4]
%   range         : range of pixel index [startInd, endInd]
%   mode          : 'label' mode or 'both' mode
% Output
%   patch         : range [0, 1], in double
%                   height * width * channelNum * scaleNum * patchNum
%   label         : either 1 or 2, in double, 1-based indexing
%                   1 -> background
%                   2 -> people

%% parse input
height = size(im, 1);
width = size(im, 2);
if nargin < 6
  range = [1, height * width];
end
if nargin < 7
  mode = 'both';
end
assert(strcmp(mode, 'label') || strcmp(mode, 'both'));
startInd = range(1);
endInd = range(2);
patchNum = endInd - startInd + 1;
scaleNum = length(scale);

%% get pach label, 1-indexed
label = imGt;
label(label > 0.5) = 1;
label(label <= 0.5) = 0;
label = label + 1;
label = reshape(label, 1, []);
label = label(startInd : endInd);

if strcmp(mode, 'label')
  patch = [];
  return;
end

%% get ready image
assert(ndims(im) == 3, 'im dim error');
if strcmp(channel, 'all')
  assert(ndims(im) == 3, 'im dim error');
elseif strcmp(channel, 'grayscale')
  im = rgb2gray(im);
elseif strcmp(channel, 'first')
  im = im(:,:,1);
elseif strcmp(channel, 'second')
  im = im(:,:,2);
elseif strcmp(channel, 'third')
  im = im(:,:,3);
end

%% intialize patch
if strcmp(channel, 'all')
  channelNum = 3;
elseif strcmp(channel, 'grayscale') || ...
       strcmp(channel, 'first') || ...
       strcmp(channel, 'second') || ...
       strcmp(channel, 'third')
  channelNum = 1;
else
  error('channel error');
end
patch = zeros(patchDim, patchDim, channelNum, scaleNum, patchNum);

%% get patch for each scale
for scaleInd = 1 : scaleNum
  % scale image
  currentScale = scale(scaleInd);
  imScale = imresize(im, 1 / currentScale);
  height = size(imScale, 1);
  width = size(imScale, 2);
  
  % get patch
  patchInd = 0;
  for pixelInd = startInd : endInd
    patchInd = patchInd + 1;
    % get scaled row and col index
    [rowInd, colInd] = ind2sub(size(im), pixelInd); % in original im
    rowInd = round(rowInd / currentScale);
    colInd = round(colInd / currentScale);
    % get padding size
    upLeftHalf = floor((patchDim - 1) / 2);
    bottomRightHalf = ceil((patchDim - 1) / 2);
    upInd = rowInd - upLeftHalf;
    leftInd = colInd - upLeftHalf;
    bottomInd = rowInd + bottomRightHalf;
    rightInd = colInd + bottomRightHalf;
    % get patch
    targetPatch = imScale(max(upInd, 1) : min(bottomInd, height), ...
      max(leftInd, 1) : min(rightInd, width), :);
    % add padding
    upPad = max(0, 1 - upInd);
    bottomPad = max(0, bottomInd - height);
    leftPad = max(0, 1 - leftInd);
    rightPad = max(0, rightInd - width);
    targetPatch = padarray(targetPatch, [upPad, leftPad], 'pre');
    targetPatch = padarray(targetPatch, [bottomPad, rightPad], 'post');
    assert(isequal(size(targetPatch), [patchDim, patchDim]) || ...
      isequal(size(targetPatch), [patchDim, patchDim, 3]));
    % store patch
    patch(:, :, :, scaleInd, patchInd) = targetPatch;
  end
end

% standard normal
patch = standardNormal(patch);

end
