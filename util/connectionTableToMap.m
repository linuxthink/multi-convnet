function [ map ] = connectionTableToMap( dim, table )
%CNNTABLETOMAP Convert connectionTable to connectionMap
% Support 3-way connection table and 3-way connection map
% Input
%   dim: height * width
%   table: [inInd, outInd, kernelInd;
%           ...  , ...   , ...     ];
% Output
%   map: [1, 0, 3, 4;
%         1, 2, 0, 4;
%         0, 2, 3, 4];

connctionNum = size(table, 1);
map = zeros(dim);
for i = 1:connctionNum
  inInd = table(i, 1);
  outInd = table(i, 2);
  kernelInd = table(i, 3);
  map(inInd, outInd) = kernelInd;
end

end
