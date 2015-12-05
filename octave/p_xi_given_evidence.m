function [p_xi] = p_xi_given_evidence(jointNo, a, keypoints, descriptors, imsize, codebook, p_xiMinusKeypoint_mu, p_xiMinusKeypoint_var, dispProgress=true)
    %finds SUM_keypointDescriptorPair[p(x_jointNo | a, keypoint, descriptor)]
    %  where  p(x_jointNo | a, keypoint, descriptor) = SUM_cj[p(x_jointNo | a, cj, keypoint) * p(cj | descriptor)]
    %INPUTS
        %jointNo - the joint number whose pdf we wish to find
        %a - the articulation state
        %keypoints - Nx3 matrix of keypoints in [x,y,~] form
        %descriptors - NxF matrix of descriptors corresponding to each keypoint
        %imsize - 1x2 matrix determining the range of positions for which we desire to find p_xi
        %codebook - KxF matrix giving the centers of each cluster of descriptors
        %p_xiMinusKeypoint_mu - p_xiMinusKeypoint_mu(jointNo,a,cj,:) gives the 1x2 matrix with the mean of xi-keypoint
        %p_xiMinusKeypoint_var - p_xiMinusKeypoint_var(jointNo,a,cj,:,:) gives the 2x2 matrix with the covariance of xi-keypoint
        %dispProgress - whether to display progress to stdout
    %OUTPUTS
        %p_xi - image of size [H, W, numJoints] giving the probability of x_jointNo landing on each pixel
    N = size(keypoints,1);
    K = size(codebook,1);
    p_cj = p_cj_for_keypoints(descriptors, codebook);
    p_xi = zeros(imsize,'double');
    for kpNo = 1:N
        if (dispProgress)
            disp(['On keypoint ',num2str(kpNo),'/',num2str(N)]);
            fflush(stdout);
        end
        for cj = 1:K
            %find p(x_jointNo | a, cj, keypoint)
            mu = keypoints(kpNo, 1:2) + reshape(p_xiMinusKeypoint_mu(jointNo,a,cj,:), [1,2]);
            var = reshape(p_xiMinusKeypoint_var(jointNo,a,cj,:,:), [2,2]);
            if (~any(isnan(mu)))
                p_xi_given_cj = p_gaussian2D_on_grid(imsize, mu, var);
                p_xi = p_xi + p_xi_given_cj * p_cj(kpNo,cj);
            end
        end
    end
end


function [p_cj] = p_cj_for_keypoints(descriptors, codebook)
    %finds p(cj | descriptor) for each descriptor
    %INPUTS
        %descriptors - NxF matrix of descriptors
        %codebook - KxF matrix of cluster centers
    %OUTPUTS
        %p_cj - NxK matrix with the probability of each descriptor belonging to each cluster
    dists = pdist2(descriptors, codebook, 'euclidean');
    p_cj = dists ./ sum(dists,2);
end
