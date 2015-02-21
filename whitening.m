function patch_w = whitening(patch,acc,epsilon,display)
%% whitening.m do whitening for the input patches using ZCAwhitening
% INPUT:
% patch: patches selected from the raw images (stored in first two dimensions).
% acc: to what degree we do the whitening, acc should be in the interval of
%         (0,1)
% epsilon:
% display: true or false
% OUTPUT:
% patch_w: the whitened patches, also stored in columns 
if nargin < 4
  display = false;
end
patch = squeeze(patch);

% transform the size of the patch into columns
[m,n,p] = size(patch);
tmp=zeros(m*n,p);
for i = 1:p
    im = patch(:,:,i);
    im = reshape(im,m*n,1);
    tmp(:,i) = im;
end
patch = tmp;

% visualization
if display
  figure('name','Raw patches');
  randsel = randi(size(patch,2),100,1); % A random selection of samples for visualization
  display_network(patch(:,randsel));
end

% zero-mean
avg = mean(patch, 1);
patch = patch - repmat(avg, size(patch, 1), 1);

% get patchRot
sigma = patch * patch' / size(patch, 2);
[U, S, V] = svd(sigma);
patchRot = U' * patch;

% check patchRot
covar = patchRot * patchRot' / size(patchRot, 2);
if display
  figure('name','Visualisation of covariance matrix');
  imagesc(covar);
end

% get k, the number of eigenvectors to keep
SD = diag(S);
k = length(SD((cumsum(SD) / sum(SD)) <= acc));

% PCA
patchRot = U(:,1:k)' * patch;
patchHat = U(:,1:k) * patchRot;
% visualization
if display
  figure('name',['PCA processed patches ',sprintf('(%d / %d dimensions)', k, size(patch, 1)),'']);
  display_network(patchHat(:,randsel));
  figure('name','Raw patches');
  display_network(patch(:,randsel));
end

% PCAwhitening
patchPCAWhite = diag(1./sqrt(diag(S) + epsilon)) * U' * patch;

% check PCAwhitening
covar = patchPCAWhite * patchPCAWhite' / size(patchPCAWhite, 2);
if display
  figure('name','Visualisation of covariance matrix');
  imagesc(covar);
end

% ZCAwhitening
patchZCAWhite = U * patchPCAWhite;
% visualization
if display
  figure('name','ZCA whitened patches');
  display_network(patchZCAWhite(:,randsel));
  figure('name','Raw patches');
  display_network(patch(:,randsel));
end

patch_w = patchZCAWhite;

% transform the size of patch in to 2D
tmp = zeros(m,n,p);
for i = 1:p
    im = patch_w(:,i);
    im = reshape(im,m,n);
    tmp(:,:,i) = im;
end
patch_w = tmp;
end


