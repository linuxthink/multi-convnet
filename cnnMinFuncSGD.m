function [optTheta, opt] = cnnMinFuncSGD(funObj, theta, opt, netOpt)
% Runs stochastic gradient descent with momentum
% Input
%  funObj      : function handle which accepts as input theta,
%                data, labels and returns cost and gradient w.r.t
%                to theta.
%  theta       : unrolled parameter vector
%  allIm       : image in h * w * channel * imageNum, double
%                can be YUV or RGB
%  allImGt     : image groundtruth in h * w * channel * imageNum, double
%  opt         : struct to store specific options for optimization
%                - alpha*      : initial learning rate
%                - minibatch*  : size of minibatch
%                - momentum    : momentum constant
%  netOpt      :
% Output
%  optTheta    : optimal theta
%  costCollector : cost over iterations

tic;

% if opt.plotConverge
%   figure();
% end

%% setup
% init velocity
velocity = zeros(size(theta));
% for getPatchRandom
opt.indRange = opt.trainIndRange;
% check netOpt validity
assert(length(opt.costCollector) == opt.completedIterNum, 'error init');
assert(size(opt.ratio2Collecor, 1) == opt.completedIterNum, 'error init');
assert(length(opt.pixelAccuracyCollector) == opt.completedIterNum, 'error init');
assert(length(opt.overlapAccuracyCollector) == opt.completedIterNum, 'error init');
% do backup if init is 0
if opt.completedIterNum == 0
  save(getBackupPath(opt, netOpt), 'theta', 'opt', 'netOpt');
end

%% SGD loop
initCompletedIterNum = opt.completedIterNum;
for iterInd = initCompletedIterNum + 1 : opt.totalIterNum
  % get next minibatch
  % [dataMinibatch, labelMinibatch] = getPatchRandom(allIm, allImGt, opt, opt.minibatch);
  [dataMinibatch, labelMinibatch] = getPatchRandom(opt, opt.minibatch);
  
  % train minibatch
  [cost, grad, hX] = funObj(theta, dataMinibatch, labelMinibatch);
  
  % evaluate minibatch
  [~, labelPredict] = max(hX);
  ratio2 = [(sum(labelPredict(:)) - length(labelPredict(:))) / ...
            length(labelPredict(:)), ...
            (sum(labelMinibatch(:)) - length(labelMinibatch(:))) / ...
            length(labelMinibatch(:))];
  [pixelAccuracy, overlapAccuracy] = getAccuracy(labelPredict, labelMinibatch);
  
  % adjust velocity and theta
  velocity = opt.momentum * velocity + opt.alpha * grad;
  theta = theta - velocity;
  
  % store training results
  opt.completedIterNum = opt.completedIterNum + 1;
  opt.ratio2Collecor = [opt.ratio2Collecor; ratio2];
  opt.costCollector = [opt.costCollector; cost];
  opt.pixelAccuracyCollector = [opt.pixelAccuracyCollector; pixelAccuracy];
  opt.overlapAccuracyCollector = [opt.overlapAccuracyCollector; overlapAccuracy];
  assert(opt.completedIterNum == iterInd, 'error iterInd');
  assert(length(opt.costCollector) == opt.completedIterNum, 'error length');
  assert(size(opt.ratio2Collecor, 1) == opt.completedIterNum, 'error length');
  assert(length(opt.pixelAccuracyCollector) == opt.completedIterNum, 'error length');
  assert(length(opt.overlapAccuracyCollector) == opt.completedIterNum, 'error length');
  
  % report time
  elapseTime = toc;
  remainTime = elapseTime / (opt.completedIterNum - initCompletedIterNum) * ...
    (opt.totalIterNum - opt.completedIterNum);
  fprintf('iter %d of %d, remainTime %s, cost %f, last500 %f %f\n', ...
    iterInd, opt.totalIterNum, time2str(remainTime), cost, ...
    mean(opt.pixelAccuracyCollector(...
    end-min(500, length(opt.pixelAccuracyCollector))+1:end)), ...
    mean(opt.overlapAccuracyCollector(...
    end-min(500, length(opt.overlapAccuracyCollector))+1:end)));
  
  % backp training results
  if ~mod(iterInd, opt.backupInterval) || iterInd == opt.totalIterNum
    disp(['saving at iteration ', int2str(iterInd)]);
    save(getBackupPath(opt, netOpt), 'theta', 'opt', 'netOpt');
  end
  
  % plot converge graph
  if opt.plotConverge
    plotOpt(opt);
    drawnow;
    pause(0);
  end
end

optTheta = theta;

end

function [backupPath] = getBackupPath(opt, netOpt)
  backupPath = opt.modelPath;
  backupPath = [backupPath, opt.modelPrefix];
  backupPath = [backupPath, '_', getTimeStamp()];
  backupPath = [backupPath, '_', getOptStamp(opt)];
  backupPath = [backupPath, '_', getNetOptStamp(netOpt)];
  backupPath = [backupPath, '_iter', int2str(opt.completedIterNum)];
  backupPath = [backupPath, '.mat'];
end


