function [cost, grad, hX] = cnnCostFast(theta, data, label, netOpt, opt)
% Input       
%  theta      : all weights and bias
%  data       : strictly follows dimension
%               height * width * channelNum * scaleNum * patchNum
%  label      : usually 1-indexed
%  netOpt     : options for the network
%  opt        : weightdecay option
% Output
%  cost       : cross entropy cost
%  grad       : gradient with respect to theta (if pred==False)
%  hX         : list of predictions for each example (if pred==True)
% Notes for multicale network
% - outputs from each network are concatinated to from vector F
% - F is passed to fully connected and finally softmax, fully connected 
%   layer is optional
%  

%% set up
% parameters
layerNum = length(netOpt);
singleLayerNum = netOpt{1}.singleLayerNum;
concatLayerNum = netOpt{1}.concatLayerNum;
scaleNum = size(data, 4);
assert(scaleNum == opt.scaleNum, 'error data, netOpt scaleNum');
sampleNum = size(data, 5);
classNum = netOpt{end}.outSize;
% init stacks
stack = cnnParamToStack(theta, netOpt); % actual parameter stack
gradStack = stack; % because of same dimension
outStack = cell(singleLayerNum, scaleNum); % output for each layer
concatOutStack = cell(concatLayerNum, 1);
deltaStack = cell(singleLayerNum, scaleNum); % error term for each layer
concatDeltaStack = cell(concatLayerNum, 1);
maxIndex = cell(singleLayerNum, scaleNum);

%% Forward Propagation
% indifidual multi-scale nets
for s = 1:scaleNum
  for i = 1:singleLayerNum
    switch netOpt{i}.type
      case 'input'
        outStack{i,s} = reshape(data(:, :, :, s, :), ...
          [netOpt{i}.imageDim, netOpt{i}.imageDim, netOpt{i}.inChannel, ...
          sampleNum]);
      case 'convolution'
        outStack{i,s} = zeros([netOpt{i}.outSize, sampleNum]);
        for sampleInd = 1:sampleNum
          inMapBuff = outStack{i-1,s}(:, :, :, sampleInd);
          kernelBuff = stack{i}.kernel;
          outChannelBuff = netOpt{i}.outChannel;
          connectionTableBuff = netOpt{i}.connectionTable;
          outMapBuff = cnnConvolve(inMapBuff, kernelBuff, ...
            outChannelBuff, connectionTableBuff, 'valid', false, false, opt.threadNum);
          outStack{i,s}(:, :, :, sampleInd) = outMapBuff;
%           for tableInd = 1:size(netOpt{i}.connectionTable, 1)
%             inInd = netOpt{i}.connectionTable(tableInd, 1);
%             outInd = netOpt{i}.connectionTable(tableInd, 2);
%             kernelInd = netOpt{i}.connectionTable(tableInd, 3);
%             outStack{i,s}(:, :, outInd, sampleInd) = ...
%               outStack{i,s}(:, :, outInd, sampleInd) + ...
%               conv2(outStack{i-1,s}(:, :, inInd, sampleInd), ...
%               stack{i}.kernel(:, :, kernelInd), 'valid');
%           end
          % bias
          for outInd = 1 : netOpt{i}.outChannel
            outStack{i,s}(:, :, outInd, sampleInd) = ...
              outStack{i,s}(:, :, outInd, sampleInd) + stack{i}.bias(outInd);
          end
        end
        % activation
        if strcmp(netOpt{i}.activation, 'tanh') % tanh
          outStack{i,s} = tanh(outStack{i,s});
        else % sigmoid
          outStack{i,s} = 1 ./ (1 + exp(-outStack{i,s}));
        end
      case 'rectification'
        outStack{i,s} = abs(outStack{i-1,s});
      case 'pooling'
        [outStack{i,s}, maxIndex{i,s}] = cnnPool(outStack{i-1,s}, ...
          netOpt{i}.poolDim, netOpt{i}.poolType);
      case 'full'
        % reshape previous layer
        outStack{i-1,s} = reshape(outStack{i-1,s}, [], sampleNum);
        outStack{i,s} = [stack{i}.W, stack{i}.b] * ...
          [outStack{i-1,s}; ones(1, size(outStack{i-1,s}, 2))];
        if strcmp(netOpt{i}.activation, 'tanh')
          outStack{i,s} = tanh(outStack{i,s}); % tanh
        elseif strcmp(netOpt{i}.activation, 'sigmoid')
          outStack{i,s} = 1 ./ (1 + exp(-outStack{i,s})); % sigmoid
        end
    end % switch layer type
  end % layerInd
end % scaleInd

% create bridge - outStackBridge
singleLength = prod(netOpt{singleLayerNum}.outSize); % one sample one scale
sampleLength = singleLength * scaleNum; % total length of one sample
outStackBridge = zeros(sampleLength, sampleNum);
for sampleInd = 1 : sampleNum
  for scaleInd = 1 : scaleNum
    if length(netOpt{singleLayerNum}.outSize) == 1
      targetOutStack = reshape(...
        outStack{singleLayerNum, scaleInd}(:,sampleInd), [], 1);
    elseif length(netOpt{singleLayerNum}.outSize) == 2
      targetOutStack = reshape(...
        outStack{singleLayerNum, scaleInd}(:,:,sampleInd), [], 1);
    elseif length(netOpt{singleLayerNum}.outSize) == 3
      targetOutStack = reshape(...
        outStack{singleLayerNum, scaleInd}(:,:,:,sampleInd), [], 1);
    end
    outStackBridge((scaleInd-1)*singleLength+1:scaleInd*singleLength, ...
      sampleInd) = targetOutStack;
  end
end

% concatenated nets
for i = singleLayerNum+1:layerNum
  iConcat = i - singleLayerNum; % the layer index of concat layers
  if iConcat == 1
    concatOutStack{iConcat} = [stack{i}.W, stack{i}.b] * ...
    [outStackBridge; ones(1, size(outStackBridge, 2))];
  else
    concatOutStack{iConcat} = [stack{i}.W, stack{i}.b] * ...
      [concatOutStack{iConcat - 1}; ...
      ones(1, size(concatOutStack{iConcat - 1}, 2))];
  end
  % switch full_concat and softmax
  switch netOpt{i}.type
    case 'full_concat'
      if strcmp(netOpt{i}.activation, 'tanh')
        concatOutStack{iConcat} = tanh(concatOutStack{iConcat});
      elseif strcmp(netOpt{i}.activation, 'sigmoid')
        concatOutStack{iConcat} = 1 ./ (1 + exp(-concatOutStack{iConcat}));
      end
    case 'softmax'
      aOut = concatOutStack{iConcat};
      aOut = bsxfun(@minus, aOut, max(aOut, [], 1));
      aOutExp = exp(aOut);
      hX = bsxfun(@rdivide, aOutExp, sum(aOutExp));
      gtMatrix = 1 * (repmat(label, [classNum, 1]) == ...
                      repmat((1:classNum)', [1, sampleNum]));
      cost = - (1 / sampleNum) * gtMatrix(:)' * log(hX(:));
    otherwise
      error('layer error');
  end % switch layer type
end % layerInd
% weight decay
if opt.weightDecay
  for i = 1:length(stack)
    switch netOpt{i}.type
    case 'convolution'
      cost = cost + (opt.lambda/2) * sumsqr(stack{i}.kernel);
    case {'full', 'full_concat', 'softmax'}
      cost = cost + (opt.lambda/2) * sumsqr(stack{i}.W);
    end
  end
end

%% return if only make predictions
if opt.predictOnly
  cost = -1;
  grad = 0;
  return;
end

%% Backpropagate delta 
% 1) each layer calculate error term for previous layer
% 2) self refine needed if the layer contains activation
% concatLayers
for i = layerNum : -1 : singleLayerNum + 1
  iConcat = i - singleLayerNum;
  switch netOpt{i}.type
    case 'full_concat'
      % self refine
      if strcmp(netOpt{i}.activation, 'tanh')
        concatDeltaStack{iConcat} = concatDeltaStack{iConcat} .* ...
          (1 - concatOutStack{iConcat} .* concatOutStack{iConcat});
      elseif strcmp(netOpt{i}.activation, 'sigmoid')
        concatDeltaStack{iConcat} = concatDeltaStack{iConcat} .* ...
          (1 - concatOutStack{iConcat}) .* concatOutStack{iConcat};
      end
    case 'softmax'
      % get last delta stack
      concatDeltaStack{iConcat} = -(gtMatrix - hX);
    otherwise
      error('layer type error');
  end
  % calculate and reshape for previous
    if iConcat - 1 > 0
      concatDeltaStack{iConcat-1} = stack{i}.W' * concatDeltaStack{iConcat};
      concatDeltaStack{iConcat-1} = reshape(concatDeltaStack{iConcat-1}, ...
        [netOpt{i-1}.outSize, sampleNum]);
    else
      deltaStackBridge = stack{i}.W' * concatDeltaStack{iConcat};
      deltaStackBridge = reshape(deltaStackBridge, ...
        [sampleLength, sampleNum]);
    end
end

% transfer the deltaStackBridge
for scaleInd = 1 : scaleNum
  deltaStack{singleLayerNum, scaleInd} = ...
    zeros([netOpt{singleLayerNum}.outSize, sampleNum]);
  for sampleInd = 1 : sampleNum
    if length(netOpt{singleLayerNum}.outSize) == 1
      singleLayerOutSize = [netOpt{singleLayerNum}.outSize, 1];
    else
      singleLayerOutSize = netOpt{singleLayerNum}.outSize;
    end
    targetDeltaStack = reshape(deltaStackBridge(...
      (scaleInd-1)*singleLength+1:scaleInd*singleLength, sampleInd), ...
      singleLayerOutSize);
    if length(netOpt{singleLayerNum}.outSize) == 1
      deltaStack{singleLayerNum, scaleInd}(:, sampleInd) = ...
        targetDeltaStack;
    elseif length(netOpt{singleLayerNum}.outSize) == 2
      deltaStack{singleLayerNum, scaleInd}(:, :, sampleInd) = ...
        targetDeltaStack;
    elseif length(netOpt{singleLayerNum}.outSize) == 3
      deltaStack{singleLayerNum, scaleInd}(:, :, :, sampleInd) = ...
        targetDeltaStack;
    end
  end
end

% single layers
for s = 1:scaleNum
  for i = singleLayerNum : -1 : 2
    switch netOpt{i}.type
      case 'input'
        % do nothing
      case 'convolution'
        % self refine
        if strcmp(netOpt{i}.activation, 'tanh') % self refine, tanh
          currentOutStack = reshape(outStack{i,s}, size(deltaStack{i,s}));
          deltaStack{i,s} = deltaStack{i,s} .* ...
            (1 - currentOutStack .* currentOutStack);
        else % self refine, sigmoid
          currentOutStack = reshape(outStack{i,s}, size(deltaStack{i,s}));
          deltaStack{i,s} = deltaStack{i,s} .* (1 - currentOutStack) ...
            .* currentOutStack;
        end
        % calc previous delta if layer index > 2
        deltaStack{i-1,s} = zeros([netOpt{i-1}.outSize, sampleNum]);
        for sampleInd = 1:sampleNum
          inMapBuff = deltaStack{i,s}(:, :, :, sampleInd);
          kernelBuff = stack{i}.kernel;
          outChannelBuff = netOpt{i}.inChannel;
          connectionTableBuff = zeros(size(netOpt{i}.connectionTable));
          connectionTableBuff(:,1) = netOpt{i}.connectionTable(:,2);
          connectionTableBuff(:,2) = netOpt{i}.connectionTable(:,1);
          connectionTableBuff(:,3) = netOpt{i}.connectionTable(:,3);
          outMapBuff = cnnConvolve(inMapBuff, kernelBuff, ...
            outChannelBuff, connectionTableBuff, 'full', false, true, opt.threadNum);
          deltaStack{i-1,s}(:,:,:,sampleInd) = outMapBuff;
%           for tableInd = 1:size(netOpt{i}.connectionTable, 1)
%             inInd = netOpt{i}.connectionTable(tableInd, 1);
%             outInd = netOpt{i}.connectionTable(tableInd, 2);
%             kernelInd = netOpt{i}.connectionTable(tableInd, 3);
%             deltaStack{i-1,s}(:, :, inInd, sampleInd) = ...
%               deltaStack{i-1,s}(:, :, inInd, sampleInd) + ...
%               conv2(deltaStack{i,s}(:, :, outInd, sampleInd), ...
%               rot90(stack{i}.kernel(:, :, kernelInd), 2), 'full');
%           end
        end
      case 'rectification'
        rectifier = outStack{i-1,s};
        rectifier(rectifier >= 0) = 1;
        rectifier(rectifier < 0) = -1;
        deltaStack{i-1,s} = deltaStack{i,s} .* rectifier;
      case 'pooling'
        deltaStack{i-1,s} = zeros([netOpt{i-1}.outSize, sampleNum]);
        if strcmp(netOpt{i}.poolType, 'max')
  %         [~, maxIndex] = cnnPool(outStack{i-1,s}, netOpt{i}.poolDim, ...
  %           netOpt{i}.poolType);
          deltaStack{i-1,s}(maxIndex{i,s}) = deltaStack{i,s}(:);
        else % mean pooling
          outChannel = netOpt{i-1}.outSize(3);
          for sampleInd = 1:sampleNum
            for channelInd = 1:outChannel
              deltaStack{i-1,s}(:, :, channelInd, sampleInd) = ...
                (1 ./ netOpt{i}.poolDim ^ 2) ...
                * kron(deltaStack{i,s}(:, :, channelInd, sampleInd), ...
                ones(netOpt{i}.poolDim));
            end
          end
        end
      case 'full'
        if strcmp(netOpt{i}.activation, 'tanh')
          deltaStack{i,s} = deltaStack{i,s} .* (1 - outStack{i,s} ...
            .* outStack{i,s});
        elseif strcmp(netOpt{i}.activation, 'sigmoid')
          deltaStack{i,s} = deltaStack{i,s} .* (1 - outStack{i,s}) ...
            .* outStack{i,s};
        end
        deltaStack{i-1,s} = stack{i}.W' * deltaStack{i,s};
        deltaStack{i-1,s} = reshape(deltaStack{i-1,s}, ...
          [netOpt{i-1}.outSize, sampleNum]);
      otherwise
        error('layer type error');
    end
  end
end

%% Gradients
% use previous layer output and this layer gradient stack to calculate
% gradients for this layer's parameter
for i = layerNum:-1:2
  iConcat = i - singleLayerNum; % the layer index of concat layers
  switch netOpt{i}.type
    case 'input'
      % do nothing
    case 'convolution'
      gradStack{i}.kernel = zeros(size(gradStack{i}.kernel));
      gradStack{i}.bias = zeros(size(gradStack{i}.bias));
      for s = 1:scaleNum
        % kernel
        for sampleInd = 1:sampleNum
          inMapBuff = outStack{i-1,s}(:, :, :, sampleInd);
          kernelBuff = deltaStack{i,s}(:, :, :, sampleInd);
          outChannelBuff = netOpt{i}.kernelNum;
          connectionTableBuff = zeros(size(netOpt{i}.connectionTable));
          connectionTableBuff(:, 1) = netOpt{i}.connectionTable(:, 1);
          connectionTableBuff(:, 2) = netOpt{i}.connectionTable(:, 3);
          connectionTableBuff(:, 3) = netOpt{i}.connectionTable(:, 2);
          outMapBuff = cnnConvolve(inMapBuff, kernelBuff, ...
            outChannelBuff, connectionTableBuff, 'valid', true, false, opt.threadNum);
          gradStack{i}.kernel = outMapBuff + gradStack{i}.kernel; % accumulate!
%           for tableInd = 1:size(netOpt{i}.connectionTable, 1)
%             inInd = netOpt{i}.connectionTable(tableInd, 1);
%             outInd = netOpt{i}.connectionTable(tableInd, 2);
%             kernelInd = netOpt{i}.connectionTable(tableInd, 3);
%             gradStack{i}.kernel(:, :, kernelInd) = ...
%               gradStack{i}.kernel(:, :, kernelInd) + ...
%               conv2(rot90(outStack{i-1,s}(:, :, inInd, sampleInd), 2), ...
%               deltaStack{i,s}(:, :, outInd, sampleInd), 'valid');
%           end
        end
        % bias
        for outInd = 1:netOpt{i}.outChannel
          gradStack{i}.bias(outInd) = gradStack{i}.bias(outInd) +  ...
            sum(reshape(deltaStack{i,s}(:, :, outInd, :), [], 1));
        end
      end
      gradStack{i}.kernel = (1 / sampleNum) * gradStack{i}.kernel;
      gradStack{i}.bias = (1 / sampleNum) * gradStack{i}.bias;
    case 'rectification'
      gradStack{i} = [];
    case 'pooling'
      gradStack{i} = [];
    case 'full'
      gradStack{i}.W = zeros(size(gradStack{i}.W));
      gradStack{i}.b = zeros(size(gradStack{i}.b));
      for s = 1:scaleNum
        gradStack{i}.W = gradStack{i}.W + ...
          (1/sampleNum) * deltaStack{i,s} * outStack{i-1,s}';
        gradStack{i}.b = gradStack{i}.b + ...
          (1/sampleNum) * sum(deltaStack{i,s}, 2);
      end
    case {'full_concat', 'softmax'}
      if iConcat > 1
        gradStack{i}.W = (1/sampleNum) * concatDeltaStack{iConcat} * ...
          concatOutStack{iConcat-1}';
      else
        gradStack{i}.W = (1/sampleNum) * concatDeltaStack{iConcat} * ...
          outStackBridge';
      end
      gradStack{i}.b = (1/sampleNum) * sum(concatDeltaStack{iConcat}, 2);
    otherwise
      error('layer type error');
  end
end % gradients
% weight decay
if opt.weightDecay
  for i = 1:length(stack)
    switch netOpt{i}.type
    case 'convolution'
      gradStack{i}.kernel = gradStack{i}.kernel + ...
        opt.lambda * stack{i}.kernel;
    case {'full', 'full_concat', 'softmax'}
      gradStack{i}.W = gradStack{i}.W + opt.lambda * stack{i}.W;
    end
  end
end

%% unroll gradients
grad = cnnStackToParam(gradStack, netOpt);

end

