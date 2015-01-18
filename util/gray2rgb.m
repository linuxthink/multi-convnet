function [ imRGB ] = gray2rgb( imGray )
%GRAY2RGB Convert Grayscale Image to RGB
% If RGB, returns itslef; if gray, returns RGB

if ndims(imGray) == 3
    imRGB = imGray;
else
    imRGB = zeros(size(imGray, 1), size(imGray, 2), 3);
    imRGB(:,:,1) = imGray;
    imRGB(:,:,2) = imGray;
    imRGB(:,:,3) = imGray;
end

end

