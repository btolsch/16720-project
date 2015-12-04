function [DoGPyramid, DoGLevels] = createDoGPyramid(GaussianPyramid, levels)
%%Creates a difference-of-gaussian pyramid from a guassian pyramid
% INPUTS
%   GaussianPyramid - the image convolved with guassian filters at different levels, as in createGaussianPyramid()
%   levels - the blur levels involved in the GaussianPyramid
% OUTPUS
%   DoGPyramid - the pyramid we create
%   DoGLevels - the blur levels involved in the DoGPyramid (which will be all but the first element of "levels")

DoGPyramid = GaussianPyramid(:,:,2:end) - GaussianPyramid(:,:,1:(end-1));
DoGLevels = levels(2:end);

end
