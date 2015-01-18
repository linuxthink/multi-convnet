function [outMap, index] = cnnPool(inMap, poolDim, type)
%cnnPool Pools the given convolved features
% Input
%  inMap     : imDim * imDim * inChannel * sampleNum
%  poolDim   : dimension of pooling region
%
% Output
%  outMap    : outDim * outDim * inChannel * sampleNum

[inDim, ~, inChannel, sampleNum] = size(inMap);
assert(mod(inDim, poolDim) == 0, 'mod(convolvedDim, poolDim) ~= 0');
outDim = inDim / poolDim;

if strcmp(type, 'max')
  % max pooling
  % to be added later
  % use MaxPooling.mexmaci64, Jonathan Masci
  [outMap, index] = maxPooling(inMap, [poolDim, poolDim]);
else
  % mean pooing
  outMap = zeros(outDim, outDim, inChannel, sampleNum);
  meanFilter = ones(poolDim, poolDim) / (poolDim * poolDim);
  for sampleIndex = 1:sampleNum
    for channelhIndex = 1:inChannel
      im = inMap(:, :, channelhIndex, sampleIndex);
      meanIm = conv2(im, meanFilter, 'valid');
      outMap(:, :, channelhIndex, sampleIndex) = ...
        meanIm(1:poolDim:end, 1:poolDim:end);
    end
  end
  index = -1;
end

end






