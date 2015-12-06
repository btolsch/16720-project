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
    colors = 'ymcrgbwkkkkkkkkkkkkkkkkkkkkkkkk';
    hold on;
    imshow(im);
    for c = 1:max(clusters)
        pts = keypoints(clusters==c,:);
        plot(pts(:,1)',pts(:,2)', '*', 'color', colors(c), 'MarkerSize', 6, 'LineWidth', 2);
    end
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
    kldsjkfljdsflsdjf
    
    %test best_L_and_a
    imNo = 2;
    keypoints_test = allKeypoints{imNo};
    descriptors_test = allDescriptors{imNo};
    imsize = [boxesTrain(imNo,4)-boxesTrain(imNo,2)+1, boxesTrain(imNo,3)-boxesTrain(imNo,1)+1];
    p_cj = p_cj_for_keypoints(descriptors_test, codebook);
    [L,a,pr] = best_L_and_a(keypoints_test, p_cj, imsize, p_xi_mu, p_xi_var, p_a, p_xiMinusKeypoint_mu, p_xiMinusKeypoint_var);
    
    %FOR POSTER: show how bad the clustering is
    imNo = 2;
    im = imread([imgnamesTrain_full{imNo}]);
    box = boxesTrain(imNo,:);
    patch = im(box(2):box(4), box(1):box(3), :);
    visualizeColoredKeypoints(patch,allKeypoints{imNo},allMemberships{imNo});
    
    %FOR POSTER: get p_xi_given_evidence for single keypoint for single joint
    imNo = 2;
    keypoints_test = allKeypoints{imNo};
    descriptors_test = allDescriptors{imNo};
    keypoints_test = keypoints_test(500,:);
    descriptors_test = descriptors_test(500,:);
    keypoints_test(:,1:2) = keypoints_test(:,1:2) - boxesTrain(imNo,1:2);
    imsize = [boxesTrain(imNo,4)-boxesTrain(imNo,2)+1, boxesTrain(imNo,3)-boxesTrain(imNo,1)+1];
    p_cj = p_cj_for_keypoints(descriptors_test, codebook);
    p_x1_given_a1 = p_xi_given_evidence(1, 1, keypoints_test, p_cj, imsize, p_xiMinusKeypoint_mu, p_xiMinusKeypoint_var);
    plotImAndPts_normalized(p_x1_given_a1, keypoints_test);
    
    %FOR POSTER: show p_xi_given_evidence for single joint
    imNo = 2;
    keypoints_test = allKeypoints{imNo};
    descriptors_test = allDescriptors{imNo};
    keypoints_test(:,1:2) = keypoints_test(:,1:2) - boxesTrain(imNo,1:2);
    imsize = [boxesTrain(imNo,4)-boxesTrain(imNo,2)+1, boxesTrain(imNo,3)-boxesTrain(imNo,1)+1];
    p_cj = p_cj_for_keypoints(descriptors_test, codebook);
    p_x1_given_a1 = p_xi_given_evidence(1, 1, keypoints_test, p_cj, imsize, p_xiMinusKeypoint_mu, p_xiMinusKeypoint_var);
    imdisp(p_x1_given_a1,[]);
    
    %FOR POSTER:  show p_xi_given_x0 * (beta + p_xi_given_evidence)
    
        M = size(p_xi_mu,2);
    imsize = size(p_xi_given_appearance);
    L = zeros(M+1, 2);
    L(1,:) = x0;
    pr = 1;
    for jointNo = 1:M
        mu_given_x0 = p_xi_mu(a,jointNo,:) + x0;
        var_given_x0 = p_xi_var(a,jointNo,:,:);
        p_xi_given_x0 = p_gaussian2D_on_grid(imsize, mu_given_x0, var_given_x0);
        piece_pr_img = p_xi_given_x0 .* (beta + p_xi_given_appearance(:,:,jointNo));
        [piece_pr, idx_flat] = max(piece_pr_img(:));
        pr = pr * piece_pr;
        L(jointNo+1,:) = ind2sub(size(piece_pr_img), idx_flat);
    end

    