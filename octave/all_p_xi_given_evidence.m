function [all_p_xi] = all_p_xi_given_evidence(keypoints, p_cj, imsize, p_xiMinusKeypoint_mu, p_xiMinusKeypoint_var, dispProgress=true)
    %INPUTS
        %keypoints - Nx3 matrix of keypoints in [x,y,~] form
        %p_cj - NxK matrix with the probability of each keypoint's descriptor belonging to each cluster
        %imsize - 1x2 matrix determining the range of positions for which we desire to find p_xi
        %p_xiMinusKeypoint_mu - p_xiMinusKeypoint_mu(jointNo,a,cj,:) gives the 1x2 matrix with the mean of xi-keypoint
        %p_xiMinusKeypoint_var - p_xiMinusKeypoint_var(jointNo,a,cj,:,:) gives the 2x2 matrix with the covariance of xi-keypoint
        %dispProgress - whether to display progress to stdout
    %OUTPUTS
        %all_p_xi - AxM cell array giving the results of p_xi_given_appearance() for each articulation state and joint number
    M = size(p_xiMinusKeypoint_mu, 1);
    A = size(p_xiMinusKeypoint_mu, 2);
    all_p_xi = cell(A, M);
    for a = 1:A
        for jointNo = 1:M
            all_p_xi(a, jointNo) = p_xi_given_evidence(jointNo, a, keypoints, p_cj, imsize, p_xiMinusKeypoint_mu, p_xiMinusKeypoint_var, dispProgress);
        end
    end
end
