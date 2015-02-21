function [ in ] = standardNormal( in )
%STANDARDNORMAL Convert to 0 mean 1 variance while keeping dimensions
% Input
%   in:    height * width * ... * ...
%          (the first two dimension must contain the image)
% Output
%   out(in):   height * width * ... * ...
%              (the first two dimension must contain the image)

inSize = size(in);
in = reshape(in, size(in, 1), size(in, 2), []);

for i = 1 : size(in, 3)
  in(:,:,i) = standardNormalOne(in(:,:,i));
end
in = reshape(in, inSize);

end

function [ in ] = standardNormalOne( in )
  patchMean = mean(in(:));
  patchVar = var(in(:)) + 1e-5;
  in = (in - patchMean) / sqrt(patchVar);
end