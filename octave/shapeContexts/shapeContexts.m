function fds = shapeContexts(pts, nbinsLogR=5, nbinsTheta=12)
    %Inputs:
     %pts - Nx2 matrix of feature points
     %nbinsLogR, nbinsTheta - give the number of bins to use
    %Outputs:
     %fds - Nx(nbinsLogR*nbinsTheta) array of feature descriptors
        
    N = size(pts,1);
    Nf = nbinsLogR*nbinsTheta;
    scale = median(median(sqrt(pdist2(pts,pts,'euclidean'))));
    fds = zeros(N,Nf);
    for p=1:N
        fds(p,:) = shapeContext(pts, pts(p,:), scale, nbinsLogR, nbinsTheta);
    end
end
