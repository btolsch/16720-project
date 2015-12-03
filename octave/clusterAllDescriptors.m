function [clusters, memberships] = clusterAllDescriptors(descriptors, numClusters)
    %INPUTS
        %descriptors - cell array where each cell corresponds to the numKeypoints x numFeatures feature matrix for an image
        %numClusters - gives the number of clusters to use for kmeans clustering
    %OUTPUTS
        %clusters - numClusters x numFeatures matrix giving the centers of the clusters
        %memberships - cell array where each cell corresponds to a numKeypoints x 1 matrix giving the cluster memberships for an image
    
    numKeypoints = cellfun(@(x) [size(x,1)], descriptors, 'UniformOutput', true);
    %flatten
    descriptors_flat = zeros(sum(numKeypoints), size(descriptors{1},2));
    curIdx = 1;
    for imNo = 1:numel(descriptors)
        endIdx = curIdx + numKeypoints(imNo) - 1;
        descriptors_flat(curIdx:endIdx,:) = descriptors{imNo};
        curIdx = endIdx + 1;
    end
    %cluster
    [memberships_flat, clusters] = kmeans(descriptors_flat, numClusters, 'EmptyAction', 'singleton');
    %unflatten
    memberships = cell(size(descriptors));
    curIdx = 1;
    for imNo = 1:numel(descriptors)
        endIdx = curIdx + numKeypoints(imNo) - 1;
        memberships{imNo} = memberships_flat(curIdx:endIdx,:);
        curIdx = endIdx + 1;
    end
end
