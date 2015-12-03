function [locs, GaussianPyramid] = DoGdetector(im, sigma0, k, levels, th_contrast, th_r)
%%Gets local extrema (keypoints) from a difference-of-gaussian pyramid
% INPUTS
%   im - a grayscale image with range 0 to 1
%   sigma0 - the standard deviation of the blur at level 0
%   k - the multiplicative factor of sigma at each level, where sigma=sigma_0 k^l
%   levels - the levels of the pyramid where the blur at each level is
%   th_contrast - the minimum DoGPyramid value, used as a cutoff
%   th_r - the maximum allowed curvature ratio, used as a cutoff
% OUTPUS
%   locs - an Nx3 matrix whose rows are coordinates of the keypoints as [x,y,level]
%   GuassianPyramid - a matrix of grayscale images of size (size(im),numel(levels))

[GaussianPyramid] = createGaussianPyramid(im, sigma0, k, levels);
[DoGPyramid, DoGLevels] = createDoGPyramid(GaussianPyramid, levels);
[PrincipalCurvature] = computePrincipalCurvature(DoGPyramid);
[locs] = getLocalExtrema(DoGPyramid, DoGLevels, PrincipalCurvature, th_contrast, th_r);

end
