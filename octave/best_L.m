function [L,a,pr] = best_L_and_a(p_xi_given_appearance, p_xi_mu,p_xi_var, p_a)
    %INPUTS
        %p_xi_given_appearance - image of size [H, W, numJoints] giving the probability of x_jointNo landing on each pixel
        %p_xi_mu - AxMx2 vector giving the average xi-x0 for each joint and articulation state
        %p_xi_var - AxMx2x2 vector giving the covariance matrix of xi-x0 for each joint and articulation state
        %p_a - numArtic x 1 vector with the probability of each articulation state
    %OUTPUTS
        %L - (M+1)x2 matrix holding best set of part positions
        %a - scalar giving the best articulation state
        %pr - scalar giving p(L | a, p_xi_given_appearance)
    [L,a,pr] = best_L_and_a_naive(p_xi_mu,p_xi_var, p_xi_given_appearance, p_a);
end


function [L,a,pr] = best_L_and_a_naive(p_xi_mu,p_xi_var, p_xi_given_appearance, p_a)
    imsize = size(p_xi_given_appearance);
    L = zeros(M+1, 2);
    a = 1;
    pr = 0;
    for y = 1:imsize(1)
        for x = 1:imsize(2)
            [L_cur, a_cur, pr_cur] = best_L_and_a_given_x0([x,y], p_xi_mu,p_xi_var, p_xi_given_appearance, p_a);
            if (pr_cur >= pr)
                L = L_cur;
                a = a_cur;
                pr = pr_cur;
            end
        end
    end
end


function [L,a,pr] = best_L_and_a_given_x0(x0, p_xi_mu,p_xi_var, p_xi_given_appearance, p_a)
    A = numel(p_a);
    L = zeros(M+1, 2);
    a = 1;
    pr = 0;
    for a_cur = 1:A
        [L_cur,pr_cur] = best_L_given_x0_and_a(x0,a_cur, p_xi_mu,p_xi_var, p_xi_given_appearance)
        pr_cur = pr_cur * p_a(a_cur);
        if (pr_cur >= pr)
            L = L_cur;
            a = a_cur;
            pr = pr_cur;
        end
    end
end


function [L,pr] = best_L_given_x0_and_a(x0,a, p_xi_mu,p_xi_var, p_xi_given_appearance)
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
