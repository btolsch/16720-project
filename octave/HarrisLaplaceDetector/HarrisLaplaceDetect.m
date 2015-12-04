function [keypoints] = HarrisLaplaceDetect(image, sigma0, k, levels, corn_thresh, DOG_thresh, maxCorners = 0)
    %translated from http://code.opencv.org/attachments/609/HarrisLaplace.patch
    
    [GaussianPyramid] = createGaussianPyramid(im2double(image), sigma0, k, levels);
    [DoGPyramid, DoGLevels] = createDoGPyramid(GaussianPyramid, levels);
    numLayers = numel(levels);
    numDoGLayers = numLayers-1;
    
    % Find Harris corners on each layer
    isKeypoint = false(size(DoGPyramid));
    cornerness = zeros(size(DoGPyramid));
    for layerNo = 2:(numDoGLayers-1)
        sigmaI = sigma0 * k ^ levels(layerNo);
        sigmaD = sigmaI * 0.7;
        curr_layer = GaussianPyramid(:, :, layerNo-1);
        
        % Calculate second moment matrix
        Lx = imfilter(curr_layer, [-1, 0, 1]);
        Ly = imfilter(curr_layer, [-1; 0; 1]);
        gI = fspecial('gaussian', floor(3*sigmaI*2)+1, sigmaI);
        mu_xx = imfilter(Lx.*Lx, gI);
        mu_yy = imfilter(Ly.*Ly, gI);
        mu_xy = imfilter(Lx.*Ly, gI);
        
        % Calculates cornerness in each pixel of the image
        dets = mu_xx.*mu_yy - mu_xy.*mu_xy;
        trs = mu_xx + mu_yy;
        cornern_mat = dets - (0.04 * trs .* trs);
        
        % Find max cornerness value and rejects all corners that are lower than a threshold
        threshVal = max(cornern_mat(:)) * corn_thresh;
        cornern_mat(cornern_mat<threshVal) = 0;
        corn_dilate = imdilate(cornern_mat,  ones(3,3));  %strel("cube", 3));
        cornerness(:, :, layerNo) = cornern_mat;
        
        % Verify for each of the initial points whether the DoG attains a maximum at the scale of the point
        prevDOG = DoGPyramid(:, :, layerNo-1);
        curDOG = DoGPyramid(:, :, layerNo);
        succDOG = DoGPyramid(:, :, layerNo+1);
        kp_size = 3*sigmaI*2;
        [coordsX, coordsY] = meshgrid(
            (1:size(corn_dilate,2))+1/2,
            (1:size(corn_dilate,1))+1/2);
        isKeypoint(:, :, layerNo) = (curDOG~=0 & curDOG==corn_dilate
                        & curDOG>prevDOG & curDOG>succDOG & curDOG>=DOG_thresh
                        & (coordsX-kp_size/2)>1 & (coordsY-kp_size/2)>1
                        & (coordsX+kp_size/2)<size(image,1) & (coordsY+kp_size/2)<size(image,2));
    end
    
    %get keypoints
    idx_flat = find(isKeypoint);
    [y,x,d] = ind2sub(size(cornerness), idx_flat);
    keypoints = [x,y,d];
    
    % Sort keypoints in decreasing cornerness order
    [responsesSorted, sortOrder] = sort(cornerness(isKeypoint), 1, 'descend');
    keypoints = keypoints(sortOrder, :);

    %combine keypoints that are too close together
    for i = 2:size(keypoints,1)
        float max_diff = pow(2, keypoints[i].octave + 1/2);
        if (responsesSorted(i)==responsesSorted(i-1) && norm(keypoints(i,:)-keypoints(i-1,:)) <= max_diff)
            keypoints(i,1:2) = (keypoints(i,1:2) + keypoints(i-1,1:2)) / 2;
            keypoints([i-1],:) = [];
            i = i - 1;
        end
    end

    % Select strongest keypoints
    if (maxCorners>0 & maxCorners<size(keypoints,1))
        keypoints = keypoints(1:maxCorners, :);
end
