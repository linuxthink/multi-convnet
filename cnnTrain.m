%% Convolutional Neural Network - Training
% Yixing Lao

clc; tic;
addpath('./util/');
clear netOpt;
clear myNet;

loadModel = true;
if loadModel % continue from previous training task
  modelPath = '../data/model/';
  modelName = ...
    '4scale_1mini_20140420231106_[1_5000]_m1-2-3-4_i3_c25-7_p2_c64-5_p2_c192-5_p2_f256_fc128_fc32_s2_iter800000.mat';
  load([modelPath, modelName]);
  opt.threadNum = 16;
  opt.imPath = '../data/person_yuv/';
  opt.imGtPath = '../data/profile/';
else % create new trainign task
  %% Options
  opt.debug = false;
  opt.plotConverge = false;
  % multi-thread option
  opt.threadNum = 16;
  % train parameters
  opt.modelPrefix = '4scale_1mini';
  opt.totalIterNum = 2000000;
  opt.trainIndRange = [1, 5000];
  opt.patchDim = 46;
  opt.channel = 'all';
  opt.scale = [1,2,3,4];
  opt.scaleNum = length(opt.scale);
  opt.outClass = 2;
  opt.backupInterval = 10000;
  % opt.trainSelection = 'watershed_center';
  % SGD paramters
  opt.initIterInd = 0; % default setting
  opt.minibatch = 1;
  opt.alpha = 0.001;
  opt.momentum = 0.0;
  opt.weightDecay = true;
  opt.lambda = 1e-6;
  % paths
  opt.imPath = '../data/person_yuv/';
  opt.imGtPath = '../data/profile/';
  opt.imDebugPath = '../data/debug/';
  opt.modelPath = '../data/model/';
  % for logging the training
  opt.completedIterNum = 0;
  opt.costCollector = [];
  opt.ratio2Collecor = [];
  opt.pixelAccuracyCollector = [];
  opt.overlapAccuracyCollector = [];

  %% Network parameters
%   debug net
%   myNet = {'input', 3, opt.scale, opt.patchDim, ...
%            'convolution', 1, 5, 'tanh', [], ...
%            'pooling', 6, 'mean', ...
%            'full_concat', 8, 'tanh'...
%            'softmax', opt.outClass};
  % 2-layer net debug
%   myNet = {'input', 3, opt.scale, opt.patchDim, ...
%            'convolution', 2, 7, 'tanh', [], ...
%            'pooling', 2, 'max', ...
%            'convolution', 2, 7, 'tanh', [], ...
%            'full', 2, 'tanh', ...
%            'full_concat', 4, 'tanh', ...
%            'softmax', opt.outClass};
           % 'full_concat', 16, 'tanh', ...
% 2 layer simplified proven network
% myNet = {'input', 3, opt.scale, opt.patchDim, ...
%          'convolution', 16, 7, 'tanh', ...
%          [1,2,3,4,5,6,7,8,9,10,0,0,0,0,0,0;
%           0,0,0,0,0,0,0,0,0,0,11,12,13,0,0,0;
%           0,0,0,0,0,0,0,0,0,0,0,0,0,14,15,16], ...
%          'pooling', 2, 'max', ...
%          'convolution', 20, 7, 'tanh', 0.5, ...
%          'pooling', 2, 'max', ...
%          'convolution', 40, 7, 'tanh', 0.5, ...
%          'full', 48, 'tanh', ...
%          'full', 32, 'tanh', ...
%          'softmax', opt.outClass};
  % 2 layer proven network
% myNet = {'input', 3, opt.scale, opt.patchDim, ...
%          'convolution', 16, 7, 'tanh', ...
%          [1,2,3,4,5,6,7,8,9,10,0,0,0,0,0,0;
%           0,0,0,0,0,0,0,0,0,0,11,12,13,0,0,0;
%           0,0,0,0,0,0,0,0,0,0,0,0,0,14,15,16], ...
%          'pooling', 2, 'max', ...
%          'convolution', 40, 7, 'tanh', 0.5, ...
%          'pooling', 2, 'max', ...
%          'convolution', 80, 7, 'tanh', 0.5, ...
%          'full', 48, 'tanh', ...
%          'full', 32, 'tanh', ...
%          'softmax', opt.outClass};
  % 3-layer full net - crazy design -debug
%   myNet = {'input', 3, opt.scale, opt.patchDim, ...
%            'convolution', 25, 7, 'tanh', ...
%            [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,0,0,0,0,0,0,0,0,0;
%             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,17,18,19,20,0,0,0,0,0;
%             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,21,22,23,24,25], ...
%            'pooling', 2, 'max', ...
%            'convolution', 2, 5, 'tanh', 0.6, ...
%            'pooling', 2, 'max', ...
%            'convolution', 2, 5, 'tanh', 0.5, ...
%            'pooling', 2, 'max', ...
%            'full', 2, 'tanh', ...
%            'full_concat', 2, 'tanh', ...
%            'full_concat', 2, 'tanh', ...
%            'softmax', opt.outClass};
  % 3-layer full net - crazy design
  myNet = {'input', 3, opt.scale, opt.patchDim, ...
           'convolution', 25, 7, 'tanh', ...
           [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,0,0,0,0,0,0,0,0,0;
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,17,18,19,20,0,0,0,0,0;
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,21,22,23,24,25], ...
           'pooling', 2, 'max', ...
           'convolution', 64, 5, 'tanh', 0.6, ...
           'pooling', 2, 'max', ...
           'convolution', 192, 5, 'tanh', 0.5, ...
           'pooling', 2, 'max', ...
           'full', 256, 'tanh', ...
           'full_concat', 128, 'tanh', ...
           'full_concat', 32, 'tanh', ...
           'softmax', opt.outClass};
  netOpt = cnnInitNetOpt(myNet);

  %% Initialize Net
  [theta] = cnnInitParam(netOpt);
  fprintf('theta length: %d\n', length(theta));
  stack = cnnParamToStack(theta, netOpt); 
  newTheta = cnnStackToParam(stack, netOpt);
  assert(isequal(theta, newTheta), 'error stack param conversion');

  %% Gradient Check
  count = 0;
  if opt.debug
    % subset of training sample
    opt.predictOnly = false;
    opt.indRange = opt.trainIndRange;
    [ dataTrainDebug, labelTrainDebug ] = getPatchRandom(opt, 2);
    % [ dataTrainDebug, labelTrainDebug ] = getPatchRandom(allImYuv, allImGt, ...
    %	opt, 2);
    % gradient
    [cost, grad, hX] = cnnCost(theta, dataTrainDebug, labelTrainDebug, ...
      netOpt, opt);
    % numerical gradient
    numGrad = getNumericalGradient ...
      (@(x) cnnCost(x, dataTrainDebug, labelTrainDebug, netOpt, opt), theta);
    % compare two gradients
    diff = norm(numGrad - grad) / norm(numGrad + grad);
    disp([numGrad, grad, numGrad - grad]);
    disp(diff);
    plot(1:length(grad), abs(numGrad - grad));
    assert(diff < 1e-8, 'gradients difference too large, or try mean pooling!');
  end
end % if load model

%% Training
if ~opt.debug
  opt.predictOnly = false;
  [bestTheta, opt] = ...
    cnnMinFuncSGD(@(x,y,z) cnnCostFast(x, y, z, netOpt, opt), theta, opt, netOpt);
end

%%
fprintf('total time: %s\n', time2str(toc));




