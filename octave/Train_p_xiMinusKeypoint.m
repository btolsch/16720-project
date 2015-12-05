function [p_xiMinusKeypoint_mu, p_xiMinusKeypoint_var] = train_p_xiMinusKeypoint(aTrain, LTrain, allKeypoints, allMemberships)
    %p_xiMinusKeypoint_mu is an MxAxKx2 matrix where p_xiMinusKeypoint_mu(i,a,cj,:) gives the mean value of p(x_i-keypoint | a,cj)
    %p_xiMinusKeypoint_var is an MxAxKx2x2 vector where p_xiMinusKeypoint_var(i,a,cj,:) gives the covariance of p(x_i-keypoint | a,cj)

    N = size(aTrain, 1);
    M = size(LTrain, 2) - 1;
    A = max(aTrain);
    K = max(cellfun("max", allMemberships));
    
    %accumulate sum(X) and sum(X^2), where X is the offset between joints and keypoints
    n = zeros(A,K);
    s = zeros(M,A,K,2);
    s2_11 = zeros(M,A,K);
    s2_12 = zeros(M,A,K);
    s2_22 = zeros(M,A,K);
    for patchNo = 1:N
        a = aTrain(patchNo);
        L = reshape(LTrain(patchNo, 2:end, :), [M,2]);
        keypoints = allKeypoints{patchNo};
        memberships = allMemberships{patchNo};
        for kpNo = 1:size(keypoints,1)
            keypoint = keypoints(kpNo,1:2);
            cj = memberships(kpNo);
            jointOffsets = reshape(L-keypoint, [M,1,1,2]);
            n(a,cj) = n(a,cj) + 1;
            s(:,a,cj,:) = s(:,a,cj,:) + jointOffsets;
            s2_11(:,a,cj) = s2_11(:,a,cj) + (jointOffsets(:,1) .* jointOffsets(:,1));
            s2_12(:,a,cj) = s2_12(:,a,cj) + (jointOffsets(:,1) .* jointOffsets(:,2));
            s2_22(:,a,cj) = s2_22(:,a,cj) + (jointOffsets(:,2) .* jointOffsets(:,2));
        end
    end
    
    %get mean and covariance
    p_xiMinusKeypoint_mu = zeros(M,A,K,2);
    p_xiMinusKeypoint_var = zeros(M,A,K,2,2);
    for a = 1:A
        for cj = 1:K
            mu = s(:,a,cj,:) / n(a,cj);
            p_xiMinusKeypoint_mu(:,a,cj,:) = mu;
            p_xiMinusKeypoint_var(:,a,cj,1,1) = (s2_11(:,a,cj) / n(a,cj)) - (mu(:,1) .* mu(:,1));
            p_xiMinusKeypoint_var(:,a,cj,1,2) = (s2_12(:,a,cj) / n(a,cj)) - (mu(:,1) .* mu(:,2));
            p_xiMinusKeypoint_var(:,a,cj,2,1) = p_xiMinusKeypoint_var(:,a,cj,1,2);
            p_xiMinusKeypoint_var(:,a,cj,2,2) = (s2_22(:,a,cj) / n(a,cj)) - (mu(:,2) .* mu(:,2));
        end
    end
end
