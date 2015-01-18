function [ netOpt ] = cnnInitNetOpt( myNet )
%INITNETOPT Init netOpt automatically
% Input
%   myNet: hand input parameter
% Output
%   netOpt: network options
% (Compatible with multi-scale network)

%% check all params, make assertions
% full_concat and softmax must be in the last stages

%% init netOpt
% get number of layers
layerNum = sum(strcmp(myNet, 'input')) + ...
           sum(strcmp(myNet, 'convolution')) + ...
           sum(strcmp(myNet, 'rectification')) + ...
           sum(strcmp(myNet, 'pooling')) + ...
           sum(strcmp(myNet, 'full')) + ...
           sum(strcmp(myNet, 'full_concat')) + ...
           sum(strcmp(myNet, 'softmax'));
singleLayerNum = sum(strcmp(myNet, 'input')) + ...
                 sum(strcmp(myNet, 'convolution')) + ...
                 sum(strcmp(myNet, 'rectification')) + ...
                 sum(strcmp(myNet, 'pooling')) + ...
                 sum(strcmp(myNet, 'full'));
concatLayerNum = sum(strcmp(myNet, 'full_concat')) + ...
                 sum(strcmp(myNet, 'softmax'));
netOpt = cell(1, layerNum);

myNetInd = 1;
i = 1; % layer index!
while myNetInd <= length(myNet)
  switch myNet{myNetInd}
    case 'input'
      % read from myNet
      netOpt{i} = struct( ...
        'type', 'input', ...
        'inChannel', myNet{myNetInd+1}, ... % 1: grayscale, 3: RGB or YUV
        'scale', myNet{myNetInd+2}, ... % scales
        'outChannel', [], ... % same as inChannel
        'sampleNum', [], ... % determined by dataset automatically
        'imageDim', myNet{myNetInd+3}, ...
        'layerNum', layerNum, ... % stores some network info
        'singleLayerNum', singleLayerNum, ... % stores some network info
        'concatLayerNum', concatLayerNum, ... % stores some network info
        'outSize', []);
      % init other options
      assert(i == 1, 'input must be 1');
      netOpt{i}.scaleNum = length(netOpt{i}.scale);
      currentOutSize = [netOpt{i}.imageDim, ...
        netOpt{i}.imageDim, netOpt{i}.inChannel];
      netOpt{i}.outChannel = netOpt{i}.inChannel;
      netOpt{i}.outSize = currentOutSize;
      % increment
      i = i + 1;
      myNetInd = myNetInd + 4;

    case 'convolution'
      % read from myNet
      netOpt{i} = struct( ...
        'type', 'convolution', ...
        'inChannel', [], ...  % same as previous layer
        'outChannel', myNet{myNetInd+1}, ...  % <-----
        'kernelDim', myNet{myNetInd+2}, ...   % <-----
        'activation', myNet{myNetInd+3}, ... % <-----
        'connectionMap', myNet{myNetInd+4}, ...
        'outSize', []);
      % init other options
      assert(strcmp(netOpt{i}.activation,'tanh') || ...
        strcmp(netOpt{i}.activation,'sigmoid'), 'activation error');
      netOpt{i}.inChannel = netOpt{i-1}.outChannel;  
      % connection map and table
      if isempty(netOpt{i}.connectionMap) % empty matrix -> full connected
        netOpt{i}.connectionMap = ...
          reshape(1 : netOpt{i}.inChannel * netOpt{i}.outChannel, ...
            [netOpt{i}.inChannel, netOpt{i}.outChannel]);
      elseif length(netOpt{i}.connectionMap) == 1
        netOpt{i}.connectionMap = makeRandomTable( netOpt{i}.inChannel, ...
          netOpt{i}.outChannel, netOpt{i}.connectionMap );
      end
      assert(size(netOpt{i}.connectionMap, 1) == netOpt{i}.inChannel && ...
        size(netOpt{i}.connectionMap, 2) == netOpt{i}.outChannel, ...
        'error connection map dimension');
      netOpt{i}.connectionTable = ...
        connectionMapToTable(netOpt{i}.connectionMap);
      uniqueMap = unique(netOpt{i}.connectionMap(:));
      uniqueMap(uniqueMap == 0) = [];
      netOpt{i}.kernelNum = size(uniqueMap, 1);
      % assert( ...
      %   max(netOpt{i}.connectionTable(:,1)) == netOpt{i}.inChannel && ...
      %   min(netOpt{i}.connectionTable(:,1)) == 1 && ...
      %   max(netOpt{i}.connectionTable(:,2)) == netOpt{i}.outChannel && ...
      %   min(netOpt{i}.connectionTable(:,2)) == 1, ...
      %   'error connection table dimension');
      % update outSize
      currentOutSize = [currentOutSize(1) - netOpt{i}.kernelDim + 1, ...
                        currentOutSize(1) - netOpt{i}.kernelDim + 1, ...
                        netOpt{i}.outChannel];
      netOpt{i}.outSize = currentOutSize;
      % increment
      i = i + 1;
      myNetInd = myNetInd + 5;

    case 'rectification'
      % read from myNet
      netOpt{i} = struct( ...
        'type', 'rectification', ...
        'inChannel', [], ...  % same as previous layer
        'outChannel', [], ... % same as inChannel
        'outSize', []);       % same as previous outSize
      % init other options
      netOpt{i}.inChannel = netOpt{i-1}.outChannel;
      netOpt{i}.outChannel = netOpt{i}.inChannel;
      netOpt{i}.outSize = currentOutSize;
      % increment
      i = i + 1;
      myNetInd = myNetInd + 1;

    case 'pooling'
      % read from myNet
      netOpt{i} = struct( ...
        'type', 'pooling', ...
        'inChannel', [], ...  % same as previous layer
        'outChannel', [], ... % same as previous layer
        'poolDim', myNet{myNetInd+1}, ... % <-----
        'poolType', myNet{myNetInd+2}, ... 
        'outSize', []);
      % init other options
      assert(strcmp(netOpt{i}.poolType,'max') || ...
        strcmp(netOpt{i}.poolType,'mean'), 'poolType error');
      netOpt{i}.inChannel = netOpt{i-1}.outChannel;
      netOpt{i}.outChannel = netOpt{i}.inChannel;
      % update outSize
      sizeBackup = currentOutSize;
      currentOutSize = [currentOutSize(1) / netOpt{i}.poolDim, ...
                        currentOutSize(2) / netOpt{i}.poolDim, ...
                        currentOutSize(3)];
      if prod(double(~mod(currentOutSize,1)))~=1 || ...
          currentOutSize(3)~=netOpt{i}.outChannel
        fprintf('[%d %d]->pool[%d %d]\n', sizeBackup(1), sizeBackup(2), ...
          netOpt{i}.poolDim,netOpt{i}.poolDim);
      end
      assert(prod(double(~mod(currentOutSize,1))) == 1, ...
        'pooling output dim not int');
      assert(currentOutSize(3) == netOpt{i}.outChannel, ...
        'pooling dim error');
      netOpt{i}.outSize = currentOutSize;
      % increment
      i = i + 1;
      myNetInd = myNetInd + 3;

    case 'full'
      % read from myNet
      netOpt{i} = struct( ...
        'type', 'full', ...
        'inChannel', [], ...  % same as previous layer
        'outChannel', myNet{myNetInd+1}, ... % <-----
        'activation', myNet{myNetInd+2}, ... % <-----
        'outSize', []);
      % init other options
      assert(strcmp(netOpt{i}.activation,'tanh') || ...
        strcmp(netOpt{i}.activation,'sigmoid') || ...
        strcmp(netOpt{i}.activation, 'equal'), 'activation error');
      assert(length(currentOutSize)==3 || length(currentOutSize)==1, ...
        'error full layer input');
      if length(currentOutSize) == 3 % needs flatten
        currentOutSize = prod(currentOutSize(:));
      end
      netOpt{i}.inChannel = currentOutSize(1);
      % update outSize
      currentOutSize = netOpt{i}.outChannel;
      netOpt{i}.outSize = currentOutSize;
      % increment
      i = i + 1;
      myNetInd = myNetInd + 3;

    case 'full_concat'
      % read from myNet
      netOpt{i} = struct( ...
        'type', 'full_concat', ...
        'inChannel', [], ...  % MAY need to * scaleNum
        'outChannel', myNet{myNetInd+1}, ... % <----- (flattened)
        'activation', myNet{myNetInd+2}, ... % <-----
        'outSize', []);
      % init other options
      if strcmp(netOpt{i-1}.type, 'full_concat')
        assert(length(currentOutSize)==1, 'error full_concat layer input');
        netOpt{i}.inChannel = currentOutSize;
      else
        assert(length(currentOutSize)==3 || length(currentOutSize)==1, ...
          'error full_concat layer input');
        % need flattening
        netOpt{i}.inChannel = prod(currentOutSize(:)) * netOpt{1}.scaleNum;
      end
      % update outSize
      currentOutSize = netOpt{i}.outChannel;
      netOpt{i}.outSize = currentOutSize;
      % increment
      i = i + 1;
      myNetInd = myNetInd + 3;

    case 'softmax'
      netOpt{i} = struct( ...
        'type', 'softmax', ...
        'inChannel', [], ... % need flatten, MAY need to * scaleNum
        'outChannel', myNet{myNetInd+1}, ...
        'outSize', []);
      % init other options
      if strcmp(netOpt{i-1}.type, 'full_concat')
        assert(length(currentOutSize)==1, 'error softmax layer input');
        netOpt{i}.inChannel = currentOutSize;
      else
        assert(length(currentOutSize)==3 || length(currentOutSize)==1, ...
          'error softmax layer input');
        % need flattening
        netOpt{i}.inChannel = prod(currentOutSize(:)) * netOpt{1}.scaleNum;
      end
      % update outSize
      currentOutSize = netOpt{i}.outChannel;
      netOpt{i}.outSize = currentOutSize;
      % increment
      i = i + 1;
      myNetInd = myNetInd + 2;

    otherwise
      error('layer type error');
  end
end

end

function [ connectionTable ] = ...
  makeRandomTable( inChannel, outChannel, percent )
%MAKERANDOMTABLE Make random connection table
% Input
%   inChannel: input dimension
%   outChannel: output dimension
%   percent: one ouput channel is determined by "percent" percent of
%     randomlly selected input channels
% Output
%   connectionTable: inChannel * outChannel

inSelectNum = min(round(inChannel * percent), inChannel);
if inSelectNum > inChannel
  disp('inSelectNum set to inChannel');
elseif inSelectNum < 1
  disp('inSelectNum set to 1');
end

connectionTable = zeros(inChannel, outChannel);
count = 1;
for outInd = 1 : outChannel
  connectionTable(sort(randperm(inChannel, inSelectNum)), outInd) = ...
    (count : count + inSelectNum - 1)';
  count = count + inSelectNum;
end

end




