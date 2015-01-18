function [ table ] = connectionMapToTable( map )
%CNNMAPTOTABLE Convert connectionMap to connectionTable
% Support 3-way connection table and 3-way connection map
% Input
%   map : [1, 0, 1; 0, 1, 0]
% Output
%   table: [inInd, outInd, kernelInd;
%           ...  , ...   , ...      ];

inDim = size(map, 1);
outDim = size(map, 2);

% check the integrety of the connection map
uniqueKernel = unique(map(:));
kernelNum = size(uniqueKernel, 1);
minKernel = min(map(:));
maxKernel = max(map(:));
assert(kernelNum == maxKernel - minKernel + 1, 'map error');
assert(minKernel == 0 || minKernel == 1, 'min kernel error');
table = [];

for outInd = 1:outDim
  for inInd = 1:inDim
    kernelInd = map(inInd, outInd);
    if (kernelInd ~= 0)
      table = [table; [inInd, outInd, kernelInd]];
    end
  end
end

end
