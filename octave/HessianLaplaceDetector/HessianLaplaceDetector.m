function [keypoints] = HarrisLaplaceDetect(image, k, levels, corn_thresh, DOG_thresh, maxCorners = 0)


    %translated from http://code.opencv.org/attachments/609/HarrisLaplace.patch
    
    
    %%sigma_ = sigma0*k^levels(i);
    %%h = fspecial('gaussian',floor(3*sigma_*2)+1,sigma_);
    %%%num_octaves_old, num_layers_old
    
    [GaussianPyramid] = createGaussianPyramid(im2double(image), sigma0, k, levels);
    [DoGPyramid, DoGLevels] = createDoGPyramid(GaussianPyramid, levels);
    numLayers = numel(levels);
    
    
    % Find Harris corners on each layer
    isKeypoint = false(size(DoGPyramid));
    
    for layerNo = 2:numLayers
        sigmaI = sigma0 * k ^ levels(layerNo);
    
    
    for octave = 1:num_octaves_old
        for layer = 2:num_layers_old
            if (octave == 0)
                layer = num_layers_old;
            end

            sigmaI = 2 ^ (layer/num_layers_old);
            sigmaD = sigmaI * 0.7;
            curr_layer = GaussianPyramid(:, :, octave*num_layers_old+layer);

            % Calculate second moment matrix
            Lx = imfilter(curr_layer, [-1, 0, 1])
            Ly = imfilter(curr_layer, [-1; 0; 1])
            gI = fspecial('gaussian', floor(3*sigmaI*2)+1, sigmaI);
            mu_xx = imfilter(Lx.*Lx, gI);  %should have "BORDER_REPLICATE" equivalent
            mu_yy = imfilter(Ly.*Ly, gI);  %should have "BORDER_REPLICATE" equivalent
            mu_xy = imfilter(Lx.*Ly, gI);  %should have "BORDER_REPLICATE" equivalent

            % Calculates cornerness in each pixel of the image
            dets = mu_xx.*mu_yy - mu_xy.*mu_xy;
            trs = mu_xx + mu_yy;
            cornern_mat = dets - (0.04 * tr .* tr); %aka "cornerness"
            
            % Find max cornerness value and rejects all corners that are lower than a threshold
            threshVal = max(cornern_mat(:)) * corn_thresh;
            cornern_mat(cornern_mat<threshVal) = 0;
            corn_dilate = dilate(cornern_mat,  strel("cube", 3));

            imgsize = size(curr_layer);

            % Verify for each of the initial points whether the DoG attains a maximum at the scale of the point
            prevDOG = DoGPyramid(:, :, octave*num_layers_old+layer-1);
            curDOG = DoGPyramid(:, :, octave*num_layers_old+layer);
            succDOG = DoGPyramid(:, :, octave*num_layers_old+layer+1);
            kp_size = 3*sigmaI*2;
            [coordsX, coordsY] = meshgrid(
                (1:size(corn_dilate,2))+1/2,
                (1:size(corn_dilate,1))+1/2);
            isKeypoint(:, :, octave*num_layers_old+layer) = curDOG~=0 & curDOG==corn_dilate
                            & curDOG>prevDOG & curDOG>succDOG & curDOG>=DOG_thresh
                            & (coordsX-kp_size/2)>0 & (coordsY-kp_size/2)>0
                            & (coordsX+kp_size/2)<size(image,1) & (coordsY+kp_size/2)<size(image,2);
                        %those boundaries are sketchy junk
        end
    end
    
    
    %get keypoints
    idx_flat = find(isKeypoint);
    [y,x,d] = ind2sub(size(cornern_mat), idx_flat);
    keypoints = [x,y,d];
        %% Original implementation has the points as [coordsX, coordsY], with octave and cornern_mat information as well
    
    
    % Sort keypoints in decreasing cornerness order
    [responsesSorted, sortOrder] = sort(cornern_mat(isKeypoint), 1, 'descend');
        %%%%THIS IS EXPECTING A 3D concern_mat, which I don't think is the case
    keypoints = keypoints(sortOrder, :);

    %combine keypoints that are too close together
    for i = 2:size(keypoints,1)
        float max_diff = pow(2, keypoints[i].octave + 1/2);
        if (responsesSorted(i)==responsesSorted(i-1) && norm(keypoints(i,:)-keypoints(i-1,:)) <= max_diff)
            keypoints(i,1:2) = (keypoints(i,1:2) + keypoints(i-1,1:2)) / 2;
            %%%%%%%delete keypoints(i-1,:)
            %%%and continue at i (since keypoint from i is now at i-1)
        end
    end

    % Select strongest keypoints
    if (maxCorners>0 & maxCorners<size(keypoints,1))
        keypoints = keypoints(1:maxCorners, :);
end















function [locs, GaussianPyramid] = HessianLaplaceDetector(im, sigma0, k, levels, th_contrast, th_r)
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



%{
    1) Run Harris Detector at scales sigma0, 1.2*sigma0, 1.2*1.2*sigma0, ...
        cornerness = det(mu) - alpha*trace^2(mu)  %%this is our detector
        mu(x,sigmaI,sigmaD) = [mu11,mu12;mu21,mu22]
            = sigmaD*sigmaD*g(sigmaI)*[Lx^2(x,sigmaD), LxLy(x,sigmaD); LxLy(x,sigmaD), Ly^2(x,sigmaD)]
          (mu(x,sigmaI,sigmaD) is AKA second moment matrix)
        sigmaI/D = integration/differentiation scale
        La = derivitive in the a direction
        The eigenvalues of scaleAdaptedSecondMomentMatrix represent two principal signal changes in the neighborhood of a point
        ((Figure 2 useful for checking results at this point))
    2) extract interest points as local maxima at each level; reject if small cornerness
    3) For each interest point, given that it's scale is L:
        -Get LoG at scale L, L+1, L-1.
            |LoG(x,sigmaN)| = sigmaN^2 * |Lxx(x,sigmaN)+Lyy(x,sigmaN)|
        -reject if LoG not maximum at L or if LoG below threshold

%}
