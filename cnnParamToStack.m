function [stack] = cnnParamToStack(theta, netOpt)
%%CNNPARAMTOSTACK theta, netOpt -> stack               
% Input
%  theta      :  all paramters
%  netOpt     :  network Options
%
% Output
%  stack{1}, stack{2}, stack{3}, ...
%  containing all paramters

stack = cell(1, length(netOpt));
thetaIndex = 1;

for layerInd = 1:length(netOpt)
  switch netOpt{layerInd}.type
    case 'input'
      % do nothing
    case 'convolution'
      % kernel
      kernelSize = [netOpt{layerInd}.kernelDim, ...
        netOpt{layerInd}.kernelDim, netOpt{layerInd}.kernelNum];
      kernelLength = prod(kernelSize);
      stack{layerInd}.kernel = reshape( ...
        theta(thetaIndex : thetaIndex + kernelLength - 1), kernelSize);
      thetaIndex = thetaIndex + kernelLength;
      % bias
      biasLength = netOpt{layerInd}.outChannel;
      stack{layerInd}.bias = theta(thetaIndex : thetaIndex + biasLength - 1);
      thetaIndex = thetaIndex + biasLength;
    case 'rectification'
      % do nothing
    case 'pooling'
      % do nothing
    case {'full', 'full_concat', 'softmax'}
      % W
      WSize = [netOpt{layerInd}.outChannel, netOpt{layerInd}.inChannel];
      WLength = prod(WSize);
      stack{layerInd}.W = reshape( ...
        theta(thetaIndex : thetaIndex + WLength - 1), WSize);
      thetaIndex = thetaIndex + WLength;
      % b
      bLength = netOpt{layerInd}.outChannel;
      stack{layerInd}.b = theta(thetaIndex : thetaIndex + bLength - 1);
      thetaIndex = thetaIndex + bLength;
    otherwise 
      error('layer type error');
  end
end

assert(thetaIndex == length(theta) + 1, 'param2stack error');

end