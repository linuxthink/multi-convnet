function [theta] = cnnStackToParam(stack, netOpt)
%CNNSTACKTOPARAM Summary of this function goes here
% IMPORTANT only supports param stack and paramGrad stack!

theta = [];
for i = 1:length(stack)
  switch netOpt{i}.type
  case 'convolution'
    % initialize kernels, bias
    theta = [theta; stack{i}.kernel(:)];
    theta = [theta; stack{i}.bias(:)];    
  case {'full', 'full_concat', 'softmax'}
    theta = [theta; stack{i}.W(:)];
    theta = [theta; stack{i}.b(:)];
  end
end

end

