function [locs] = getLocalExtrema(DoGPyramid, DoGLevels, PrincipalCurvature, th_contrast, th_r)
%%Gets local extrema (keypoints) from a difference-of-gaussian pyramid
% INPUTS
%   DoGPyramid - the difference-of-guassians filter convolved with an image, as in createDoGPyramid()
%   DoGLevels - the blur levels involved in the DoGPyramid, as in createDoGPyramid()
%   PrincipalCurvature - a matrix the same size as DoGPyramid whose entries give the curvature ratio at each point in DoGPyramid, as in computePrincipalCurvature()
%   th_contrast - the minimum DoGPyramid value, used as a cutoff
%   th_r - the maximum PrincipalCurvature value, used as a cutoff
% OUTPUS
%   locs - an Nx3 matrix whose rows are coordinates of the keypoints as [x,y,level]

D_abs = abs(DoGPyramid);

isExtremum = false(size(D_abs));
for layer=1:size(D_abs,3)
   DAtLayer = D_abs(:,:,layer);
   localLayerMax = ordfilt2(DAtLayer, 9, ones(3,3));
   isExtremum(:,:,layer) = (DAtLayer==localLayerMax);
end
isExtremum(:,:,2:end) = isExtremum(:,:,2:end) & (D_abs(:,:,2:end) > D_abs(:,:,1:(end-1)));
isExtremum(:,:,1:(end-1)) = isExtremum(:,:,1:(end-1)) & (D_abs(:,:,1:(end-1)) > D_abs(:,:,2:end));

rOK = PrincipalCurvature<th_r;
contrastOK = D_abs>th_contrast;

keypoints = isExtremum & rOK & contrastOK;
idx_flat = find(keypoints);
[y,x,d] = ind2sub(size(D_abs), idx_flat);
locs = [x,y,d];

end

