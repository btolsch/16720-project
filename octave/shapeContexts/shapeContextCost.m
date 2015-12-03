function C = shapeContextCost(features1, features2)
    %Inputs:
     %features1, features2 - NxNf matrices of features (N=numPts, Nf=numFeatures)
    %Outputs:
     %C - NxN matrix such that C(i,j) gives the matching cost between features1(i,:) and features2(j,:)
    C = pdist2(features1,features2,'chisq');
end
