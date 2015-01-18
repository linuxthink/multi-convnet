function plotOpt(opt, range)
%PLOTOPT Plot current training status
iterInd = opt.completedIterNum;
if nargin < 2
  range = [1,iterInd];
end

subplot(3, 1, 1);
plot(range(1):range(2), opt.costCollector(range(1):range(2)));
legend('Cost', 'Location', 'NorthWest');
subplot(3, 1, 2);
plot(range(1):range(2), opt.ratio2Collecor(range(1):range(2),2), 'b', ...
     range(1):range(2), opt.ratio2Collecor(range(1):range(2),1), 'r'); % predict
legend('2 Ratio Gt','2 Ratio Pred', 'Location', 'NorthWest');
subplot(3, 1, 3);
plot(range(1):range(2), opt.pixelAccuracyCollector(range(1):range(2)), 'b', ...
     range(1):range(2), opt.overlapAccuracyCollector(range(1):range(2)), 'r'); % predict
legend('Pixel Acc','Overlap Acc', 'Location', 'NorthWest');

end

