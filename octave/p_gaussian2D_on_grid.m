function [p] = p_gaussian2D_on_grid(imsize, mu, var)
    %evaluates the probability at each position for some 2D gaussian distribution
    %INPUTS
        %imsize - 1x2 matrix determining the range of positions for which we desire to find p_xi
        %mu, var - the mean (size 1x2) and the covariance (size 2x2)
    %OUTPUTS
        %p - image of size imsize giving the probability of at each position
    [gridX, gridY] = meshgrid(1:imsize(2), 1:imsize(1));
    mu = resize(mu, [1,2]);
    var = resize(var, [2,2]);
    p_flat = mvnpdf([gridX(:), gridY(:)], mu, var);
    p = reshape(p_flat, imsize);
end
