function [PrincipalCurvature] = computePrincipalCurvature(DoGPyramid)
%%Computes R at each point in the DoG, where R is the curvature ratio
% INPUTS
%   DoGPyramid - the difference-of-guassians filter convolved with an image, as in createDoGPyramid()
% OUTPUS
%   PrincipalCurvature - a matrix the same size as DoGPyramid whose entries give R at each point in DoGPyramid

[nr, nc, nl] = size(DoGPyramid);
PrincipalCurvature = zeros([nr, nc, nl]);
for i = 1:nl
    [Dx,Dy] = gradient(DoGPyramid(:,:,i));  %use imgradientxy instead?
    [Dxx,Dxy] = gradient(Dx);
    [Dyx,Dyy] = gradient(Dy);
    Det_H = Dxx.*Dyy - Dxy.*Dyx;
    Tr_H = Dxx + Dyy;
    R = Tr_H .* Tr_H ./ Det_H;
    R( isnan(R) | isinf(R) | R<0 ) = 0;
    PrincipalCurvature(:,:,i) = R;
end

end
