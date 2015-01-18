function [ pixelAccuracy, overlapAccuracy ] = ...
  getAccuracy( labelPredict, labelTest )
%GETACCURACY Get accuracy given prediction and ground truth
% Input
%   labelPredict: must be 1 or 2
%   labelTest: must be 1 or 2

% check input
assert(length(find(labelPredict == 1)) + ...
       length(find(labelPredict == 2)) == ...
       length(labelPredict) && ...
       length(find(labelTest == 1)) + ...
       length(find(labelTest == 2)) == ...
       length(labelTest), 'error label');

% pixel accuracy
pixelAccuracy = mean(labelPredict == labelTest);

% intersect / union accuracy
labelTest = labelTest - 1;
labelPredict = labelPredict - 1;
intersectSet = labelPredict + labelTest;
intersectSet(intersectSet == 1) = 0;
intersectSet(intersectSet == 2) = 1;
unionSet = labelPredict + labelTest;
unionSet(unionSet == 2) = 1;

if sum(sum(labelPredict)) == 0 && sum(sum(labelTest)) == 0
  overlapAccuracy = 1;
else
  if sum(sum(unionSet)) == 0
    overlapAccuracy = 0;
  else
    overlapAccuracy = sum(sum(intersectSet)) / sum(sum(unionSet));
  end
end

end

