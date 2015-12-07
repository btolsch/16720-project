pkg load image;       %for im2double
pkg load statistics;  %for pdist2

%cd "C:\\Users\\Joe\\Documents\\JDocs\\School\\CMU\\2015 Fall\\16-720 Computer Vision\\Final Project\\git\\octave";

addpath(genpath('HW2_DoGDetector'));
addpath(genpath('shapeContexts'));


%helpers (not all actually used)

function plotImAndPts(im,pts)
    imshow(im2double(im)); hold on;
    hold on;
    plot(pts(:,1),pts(:,2),'*', 'MarkerSize', 6, 'LineWidth', 2);
    hold off;
end
function plotImAndPts_normalized(im,pts)
    imshow(im2double(im),[]); hold on;
    hold on;
    plot(pts(:,1),pts(:,2),'*', 'MarkerSize', 6, 'LineWidth', 2);
    hold off;
end
function visualizeShapeContext(fd, nbinsLogR=5, nbinsTheta=12)
    imshow(reshape(fd,[nbinsLogR,nbinsTheta]),[]);
end
function visualizeColoredKeypoints(im,keypoints,clusters)
    colors =  'ymcrgbwymcrgbwymcrgbwymcrgbwkkkk';
    markers = '*******+++++++xxxxxxxooooooo*+xo';
    imshow(im2double(im));
    hold on;
    for i = 1:max(clusters)
        pts = keypoints(clusters==i,:);
        plot(pts(:,1)',pts(:,2)', markers(i), 'color', colors(i), 'MarkerSize', 6, 'LineWidth', 2);
    end
    hold off;
end


%test Harris-Laplace Interest Point Detectors (ABANDONED and not done)
    cd "HarrisLaplaceDetector";
    
    image = imread('../../train-210/IFWN-sequence-035.avi-00096-R200.png');
    sigma0 = 1;
    k = 1.2;
    levels = [-1,0,1,2,3,4];
    corn_thresh = 0.03;
    DOG_thresh = 12;
    [keypoints] = HarrisLaplaceDetect(image, sigma0, k, levels, corn_thresh, DOG_thresh);
    
    cd "..";

%test feature descriptor
    cd "shapeContexts";
    testShapeContexts;
    cd "..";

%real stuff
    resultsDir = '../results/';
    dataDir = '../train-210/';
    
    %train
    numClusters = 10;
    sigma0 = 1;
    k = 1.2;
    levels = [-1,0,1,2,3,4];
    theta_c = 0.03;
    theta_r = 12;
    load([resultsDir,'train210.mat'], 'imgnamesTrain', 'aTrain', 'LTrain', 'boxesTrain');
    imgnamesTrain = imgnamesTrain(1:10);
    aTrain = aTrain(1:10);
    LTrain = LTrain(1:10,:,:);
    boxesTrain = boxesTrain(1:10,:);
    imgnamesTrain_full = cellfun(@(x) [dataDir,x], imgnamesTrain, 'UniformOutput', false);
    [allKeypoints, allDescriptors] = getAllKeypointsAndDescriptors(imgnamesTrain_full, boxesTrain, sigma0, k, levels, theta_c, theta_r);
    [codebook, allMemberships] = clusterAllDescriptors(allDescriptors, numClusters);
    [p_a, p_xi_mu, p_xi_var] = train_p_xi_and_p_a(aTrain, LTrain);
    [p_xiMinusKeypoint_mu, p_xiMinusKeypoint_var] = train_p_xiMinusKeypoint(aTrain, LTrain, allKeypoints, allMemberships);
    
    %prepare p_xi_given_evidence for all a, jointNo pairs
        %NEVER RAN TO COMPLETION
    imNo = 2;
    im = imread([imgnamesTrain_full{imNo}]);
    box = boxesTrain(imNo,:);
    keypoints = allKeypoints{imNo};
    descriptors = allDescriptors{imNo};
    im = im(box(2):box(4), box(1):box(3), :);
    keypoints(:,1:2) = keypoints(:,1:2) - box(1:2);
    imsize = [size(im,1), size(im,2)];
    p_cj = p_cj_for_keypoints(descriptors, codebook);
    all_p_xi = all_p_xi_given_evidence(keypoints, p_cj, imsize, p_xiMinusKeypoint_mu, p_xiMinusKeypoint_var);
    save([resultsDir,'im2TestData.mat'], 'im', 'box', 'keypoints', 'descriptors', 'imsize', 'p_cj', 'all_p_xi');
    
    %test best_L_and_a
        %NEVER RAN TO COMPLETION
    [L,a,pr] = best_L_and_a(keypoints, p_cj, imsize, p_xi_mu, p_xi_var, p_a, p_xiMinusKeypoint_mu, p_xiMinusKeypoint_var);
    
    %FOR POSTER: show how bad the clustering is
    visualizeColoredKeypoints(im,keypoints,allMemberships{imNo});
    
    %FOR POSTER: get p_xi_given_evidence for single keypoint for single joint
    jointNo = 1; a = 2;
    p_x1_given_a2 = p_xi_given_evidence(jointNo, a, keypoints(500,:), p_cj, imsize, p_xiMinusKeypoint_mu, p_xiMinusKeypoint_var);
    plotImAndPts_normalized(p_x1_given_a2, keypoints(500,:));
    
    %FOR POSTER: show p_xi_given_evidence for single joint
    jointNo = 1; a = 2;
    p_x1_given_a1 = p_xi_given_evidence(jointNo, a, keypoints, p_cj, imsize, p_xiMinusKeypoint_mu, p_xiMinusKeypoint_var);
    imshow(p_x1_given_a1,[]);
    
    %FOR POSTER:  show p_xi_given_x0 * (beta + p_xi_given_evidence)
    jointNo = 1; a = 2;
    x0 = double(reshape(LTrain(imNo,1,:),[1,2]) - box(1:2));
    p_xi_given_appearance = p_xi_given_evidence(jointNo, a, keypoints, p_cj, imsize, p_xiMinusKeypoint_mu, p_xiMinusKeypoint_var);
    mu_given_x0 = reshape(p_xi_mu(a,jointNo,:),[1,2]) + x0;
    var_given_x0 = p_xi_var(a,jointNo,:,:);
    p_xi_given_x0 = p_gaussian2D_on_grid(imsize, mu_given_x0, var_given_x0);
    piece_pr_img = p_xi_given_x0 .* (beta + p_xi_given_appearance(:,:,jointNo));
    imshow(piece_pr_img,[]);
    