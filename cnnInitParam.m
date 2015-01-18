function [globalTheta] = cnnInitParam(netOpt)
%%CNNINITPARAM Init parameters (weights, kernels, bias) for convnet
% Input
%   netOpt: the option for the convnet
% Output
%   globalTheta: all thetas aligned
%   netOpt: the updated netOpt (this will be eliminated later)

globalTheta = [];
for i = 1:length(netOpt)
  switch netOpt{i}.type
    case 'convolution'
      % initialize kernels, bias
      kernel = 1e-1 * randn(netOpt{i}.kernelDim, ...
        netOpt{i}.kernelDim, netOpt{i}.kernelNum);
      bias = 1e-1 * randn(netOpt{i}.outChannel, 1);
      globalTheta = [globalTheta; kernel(:); bias(:)];
    case {'full', 'full_concat', 'softmax'}
      % initialize W, b, using Xaxier's scaling factor
      r  = sqrt(6) / ...
        sqrt(netOpt{i}.outChannel + netOpt{i}.inChannel + 1);
      W = rand(netOpt{i}.outChannel, netOpt{i}.inChannel) ...
          * 2 * r - r; 
      b = 1e-1 * randn(netOpt{i}.outChannel, 1);
      globalTheta = [globalTheta; W(:); b(:)];
  end
end

end

