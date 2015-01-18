%% Convolutional Neural Network - Testing
% Yixing Lao

%% set up
clc; tic;
addpath('./util/');
clear netOpt;
clear myNet;

%% test parameters
modelPath = '../data/model/';
modelName = ...
  '20140401025254_m1-2_i3_c16-7_p2_c20-7_p2_c40-7_f48_f32_s2_iter50.mat';
load([modelPath, modelName]); % load: theta, netOpt, opt
pureModelName = modelName(1:end-4);
load([modelPath, modelName]); % load: theta, netOpt, opt
opt.testRange = [5010, 5010];
opt.testSubset = 200; %200;
opt.predictOnly = true;
opt.threadNum = 10;
opt.imPath = '../data/person_yuv/';
opt.imGtPath = '../data/profile/';
%% Testing
for imInd = opt.testRange(1) : opt.testRange(2)
  % get image
  imName = sprintf('%05d', imInd);
  fprintf('testing %d\n', imInd);
  load([opt.imPath, imName, '.mat']); % im or imYuv
  load([opt.imGtPath, imName, '.mat']); % imGt
  % init
  dataTestNum = size(im, 1) * size(im, 2);
  subsetNum = ceil(dataTestNum / opt.testSubset);
  labelPredictCollect = zeros(1, dataTestNum);
  labelTestCollect = zeros(1, dataTestNum);
  hXCollect = zeros(opt.outClass, dataTestNum);
  testTime = tic;
  % loop through all testSet
  for subsetInd = 1:subsetNum
    % get test data / label
    startInd = (subsetInd - 1) * opt.testSubset + 1;
    endInd = min(subsetInd * opt.testSubset , dataTestNum);
    [dataTest, labelTest] = getPatchFromIm(im, imGt, ...
      opt.patchDim, opt.channel, opt.scale, [startInd, endInd], 'both');
    % testing
    [~, ~, hX] = cnnCostFast(theta, dataTest, labelTest, netOpt, opt);
    [~, labelPredict] = max(hX);
    labelPredictCollect(startInd : endInd) = labelPredict;
    labelTestCollect(startInd : endInd) = labelTest;
    hXCollect(:, startInd : endInd) = hX;
    % print for amusement
    elapseTime = toc(testTime);
    remainTime = elapseTime / subsetInd * subsetNum - elapseTime;
    [pixelAccuracy, overlapAccuracy] = ...
      getAccuracy(labelPredictCollect(1 : endInd), ...
      labelTestCollect(1 : endInd));
    fprintf('tested %d of %d, remain %s, pixelAcc %f, overlapAcc %f\n', ...
      subsetInd, subsetNum, time2str(remainTime), pixelAccuracy, ...
      overlapAccuracy);
  end
  % get accuracy
  [pixelAccuracy, overlapAccuracy] = ...
    getAccuracy(labelPredictCollect, labelTestCollect);
  fprintf('pixel accuracy is %f\n', pixelAccuracy);
  fprintf('overlap accuracy is %f\n', overlapAccuracy);
  % save reconstructed image
  imPredict = reshape(labelPredictCollect - 1, size(im, 1), ...
    size(im, 2));
  imPredictSoft = reshape(hXCollect(2, :), size(im, 1), size(im, 2));
  imDebug = [im(:,:,1), imGt, imPredict, imPredictSoft];
  imwrite(imDebug, [opt.imDebugPath, pureModelName, '_img', imName, ...
    '_', num2str(pixelAccuracy), '_', num2str(overlapAccuracy), '.png'], 'png');
end

fprintf('total testing time: %s\n', time2str(toc));
