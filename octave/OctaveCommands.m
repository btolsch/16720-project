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
function visualizeShapeContext(fd, nbinsLogR=5, nbinsTheta=12)
    imshow(reshape(fd,[nbinsLogR,nbinsTheta]),[]);
end
function visualizeColoredKeypoints(im,keypoints,clusters)
    colors = 'rgbmckkkkkkkkkkkkkkkkkkkkkkk';
    hold on;
    imshow(im);
    for c = 1:max(clusters)
        pts = keypoints(clusters==c,:);
        plot(pts(:,1)',pts(:,2)', '*', 'color', colors(c), 'MarkerSize', 6, 'LineWidth', 2);
    end
end


function plotImAndLocs2(im,locs1,locs2)
    close();
    image(im);
    hold on;
    plot(locs1(:,1),locs1(:,2),'go');
    plot(locs2(:,1),locs2(:,2),'b.');
    for i = 1:length(locs1)
        p1 = locs1(i,:);
        p2 = locs2(i,:); 
        line([p1(1),p2(1)],[p1(2),p2(2)], 'Color','r','LineWidth',1);
    end
end

function [filenames] = get_all_filenames_with_extension(path, ext)
    fileList = dir(path);
    extOk = false(size(fileList));
    for fileNo = 1:size(fileList,1)
        [_,_,fileext] = fileparts(fileList(fileNo).name);
        extOk(fileNo) = strcmp(fileext, ext);
    end
    filenames = arrayfun(@(x) [x.name], fileList(extOk), 'UniformOutput', false);
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
    
    load([resultsDir,'train210.mat'], 'imgnamesTrain', 'aTrain', 'LTrain', 'boxesTrain');
    imgnamesTrain = imgnamesTrain(1:10);
    aTrain = aTrain(1:10);
    LTrain = LTrain(1:10,:,:);
    boxesTrain = boxesTrain(1:10,:);
    
    numClusters = 10;
    sigma0 = 1;
    k = 1.2;
    levels = [-1,0,1,2,3,4];
    theta_c = 0.03;
    theta_r = 12;
    
    imgnamesTrain_full = cellfun(@(x) [dataDir,x], imgnamesTrain, 'UniformOutput', false);
    [allKeypoints, allDescriptors] = getAllKeypointsAndDescriptors(imgnamesTrain_full, boxesTrain, sigma0, k, levels, theta_c, theta_r);
    [codebook, allMemberships] = clusterAllDescriptors(allDescriptors, numClusters);
    [p_a, p_xi_mu, p_xi_var] = train_p_xi_and_p_a(aTrain, LTrain);
    [p_xiMinusKeypoint_mu, p_xiMinusKeypoint_var] = train_p_xiMinusKeypoint(aTrain, LTrain, allKeypoints, allMemberships);
    
    %test best_L_and_a
    %%still need to find p_xi_given_appearance by calling p_xi_given_evidence() for each joint
    [L,a,pr] = best_L_and_a(p_xi_given_appearance, p_xi_mu,p_xi_var, p_a)
    
    %test p_xi_given_evidence
    imNo = 2;
    keypoints_test = allKeypoints{imNo};
    descriptors_test = allDescriptors{imNo};
    keypoints_test(:,1:2) = keypoints_test(:,1:2) - boxesTrain(imNo,1:2);
    imsize = [boxesTrain(imNo,4)-boxesTrain(imNo,2)+1, boxesTrain(imNo,3)-boxesTrain(imNo,1)+1];
    p_x1_given_a1 = p_xi_given_evidence(1, 1, keypoints_test, descriptors_test, imsize, codebook, p_xiMinusKeypoint_mu, p_xiMinusKeypoint_var);
    imdisp(p_x1_given_a1,[]);
    
    %test clustering
    imNo = 2;
    im = imread([imgnamesTrain_full{imNo}]);
    box = boxesTrain(imNo,:);
    patch = im(box(2):box(4), box(1):box(3), :);
    visualizeColoredKeypoints(patch,allKeypoints{imNo},allMemberships{imNo});
