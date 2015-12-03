function [keypoints, descriptors] = getAllKeypointsAndDescriptors(imgNames, boxes, sigma0, k, levels, theta_c, theta_r, dispProgress=true)
    %INPUTS
        %imgNames - cell array where each cell holds the full path to an image
        %boxes - boxes(i,:) gives [x1,y1,x2,y2,~], a bounding box indicating the important part of the ith image
        %sigma0 - the standard deviation of the blur at level 0
        %k - the multiplicative factor of sigma at each level, where sigma=sigma_0 k^l
        %levels - the levels of the pyramid where the blur at each level is
        %th_contrast - the minimum DoGPyramid value, used as a cutoff
        %th_r - the maximum allowed curvature ratio, used as a cutoff
        %dispProgress - whether to display progress to stdout
    %OUTPUTS
        %keypoints - cell array where each cell corresponds to the numKeypoints x 3 matrix of keypoints where each keypoint is of the form [x,y,scale]
        %descriptors - cell array where each cell corresponds to the numKeypoints x numFeatures feature matrix for an image
    
    numImgs = numel(imgNames);
    keypoints = cell(numImgs, 1);
    descriptors = cell(numImgs, 1);
    for imNo = 1:numImgs
        if (dispProgress)
            disp(['On image ',num2str(imNo),'/',num2str(numImgs)]);
            fflush(stdout);
        end
        im = imread([imgNames{imNo}]);
        box = boxes(imNo,:);
        patch = im(box(2):box(4), box(1):box(3), :);
        [pts, ~] = DoGdetector(patch, sigma0, k, levels, theta_c, theta_r);
        keypoints{imNo} = pts;
        descriptors{imNo} = shapeContexts(pts);
    end
end
