function [ stamp ] = getNetOptStamp( netOpt )
%GETNETOPTSTAMP Convert netOpt to a string as stamp
% Only keep the top level information
% Input
%   netOpt
% Output
%   stamp
%   e.g.: s_i3_c16-7_p2_c40-7_p2_c80-7_f80_f48_s2
%         m1-2-4_i3_c16-7_p2_c40-7_p2_c80-7_f80_f48_s2

stamp = [];

for i = 1:length(netOpt)
  switch netOpt{i}.type
    case 'input'
      scaleStr = strrep(strrep(strrep(int2str(netOpt{i}.scale), ' ', '-'), ...
        '--', '-'), '--', '-');
      stamp = [stamp, 'm', scaleStr, '_i', int2str(netOpt{i}.inChannel)];
    case 'convolution'
      stamp = [stamp, '_c', int2str(netOpt{i}.outChannel), '-', ...
        int2str(netOpt{i}.kernelDim)];
    case 'rectification'
      stamp = [stamp, '_r'];
    case 'pooling'
      stamp = [stamp, '_p', int2str(netOpt{i}.poolDim)];
    case 'full'
      stamp = [stamp, '_f', int2str(netOpt{i}.outChannel)];
    case 'full_concat'
      stamp = [stamp, '_fc', int2str(netOpt{i}.outChannel)];
    case 'softmax'
      stamp = [stamp, '_s', int2str(netOpt{i}.outChannel)];
    otherwise 
      error('layer type error');
  end
end

end