function [ stamp ] = getOptStamp( opt )
%GETOPTSTAMP Convert opt to a string as stamp
% Only keep the top level information
% Input
%   opt
% Output
%   trainIndRange

stamp = [];
stamp = ['[', int2str(opt.trainIndRange(1)), '_', ...
  int2str(opt.trainIndRange(2)), ']'];

end

