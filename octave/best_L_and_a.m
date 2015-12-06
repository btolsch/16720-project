function [L,a,pr] = best_L_and_a(keypoints, p_cj, imsize, p_xi_mu, p_xi_var, p_a, p_xiMinusKeypoint_mu, p_xiMinusKeypoint_var, dispProgress=true)
    %INPUTS
        %keypoints - Nx3 matrix of keypoints in [x,y,~] form
        %p_cj - NxK matrix with the probability of each keypoint's descriptor belonging to each cluster
        %imsize - 1x2 matrix determining the range of positions for which we desire to find p_xi
        %p_xi_mu - AxMx2 vector giving the average xi-x0 for each joint and articulation state
        %p_xi_var - AxMx2x2 vector giving the covariance matrix of xi-x0 for each joint and articulation state
        %p_a - numArtic x 1 vector with the probability of each articulation state
        %p_xiMinusKeypoint_mu - p_xiMinusKeypoint_mu(jointNo,a,cj,:) gives the 1x2 matrix with the mean of xi-keypoint
        %p_xiMinusKeypoint_var - p_xiMinusKeypoint_var(jointNo,a,cj,:,:) gives the 2x2 matrix with the covariance of xi-keypoint
        %dispProgress - whether to display progress to stdout
    %OUTPUTS
        %L - (M+1)x2 matrix holding best set of part positions
        %a - scalar giving the best articulation state
        %pr - scalar giving p(L | a, keypoints, descriptors)
    A = size(p_xi_mu, 1);
    M = size(p_xi_mu, 2);
    L = zeros(M+1, 2);
    a = 1;
    pr = 0;
    for a_cur = 1:A
        p_xi_given_appearance = zeros([imsize(1), imsize(2), M]);
        for jointNo = 1:M
            p_xi_given_appearance(:,:,jointNo) = p_xi_given_evidence(jointNo, a, keypoints, p_cj, imsize, p_xiMinusKeypoint_mu, p_xiMinusKeypoint_var);
        end
        [L_cur,pr_cur] = best_L_given_a(a_cur, p_xi_given_appearance, p_xi_mu, p_xi_var);
        pr_cur = pr_cur * p_a(a_cur);
        if (pr_cur >= pr)
            L = L_cur;
            a = a_cur;
            pr = pr_cur;
        end
    end
end


function [L,pr] = best_L_given_a(a, p_xi_given_appearance, p_xi_mu, p_xi_var)
    [L,pr] = best_L_given_a_naive(a, p_xi_given_appearance, p_xi_mu,p_xi_var);
end


function [L,pr] = best_L_given_a_naive(a, p_xi_given_appearance, p_xi_mu, p_xi_var)
    M = size(p_xi_given_appearance, 3);
    imsize = size(p_xi_given_appearance);
    L = zeros(M+1, 2);
    pr = 0;
    for y = 1:imsize(1)
        for x = 1:imsize(2)
            [L_cur, pr_cur] = best_L_given_x0_and_a([x,y], a, p_xi_given_appearance, p_xi_mu, p_xi_var)
            if (pr_cur >= pr)
                L = L_cur;
                pr = pr_cur;
            end
        end
    end
end


function [L,pr] = best_L_given_x0_and_a(x0, a, p_xi_given_appearance, p_xi_mu, p_xi_var)
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
end
