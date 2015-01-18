function numgrad = getNumericalGradient(J, theta)
% theta: a vector of parameters
% J    : a function that outputs a real-number. 
%        calling y = J(theta) will return the function value at theta. 

if size(theta, 1) < size(theta, 2)
    theta = theta';
end

% parameters
epsilon = 0.00001;
numTheta = length(theta);
fprintf('my theta length: %d\n', numTheta);

% implementation 1
tic;
numgrad = zeros(size(theta));
for i = 1 : numTheta
  % back up theta(i)
  thetaI = theta(i);
  % obtain plus result
  theta(i) = theta(i) + epsilon;
  plusResult = J(theta);
  % obtain minus result
  theta(i) = thetaI - epsilon;
  minusResult = J(theta);
  % recover theta
  theta(i) = thetaI;
  % get numerial gradient
  numgrad(i) = (plusResult - minusResult) / (2 * epsilon);
  if mod(i, 100) == 0
    fprintf('done with %d\n', i);
  end;
end
fprintf('numgrad time: %f\n', toc);

% implementation 2 - takes up too much space
% tic;
% plusMatrix = repmat(theta, [1, len]) + eye(len) * epsilon;
% minusMatrix = repmat(theta, [1, len]) - eye(len) * epsilon;
% plusCell = num2cell(plusMatrix, 1);
% minusCell = num2cell(minusMatrix, 1);
% plusResult = cellfun(J, plusCell);
% minusResult = cellfun(J, minusCell);
% numgrad = (plusResult - minusResult)' / (2 * epsilon);
% fprintf('%f\n', toc);

end





