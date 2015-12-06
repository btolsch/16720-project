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
