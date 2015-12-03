function fd = shapeContext(pts, origin, scale='auto', nbinsLogR=5, nbinsTheta=12)
    %Inputs:
     %pts - Nx2 matrix of feature points
     %origin - [x0,y0] used for shifting all points
     %scale - either 'auto', or a scale factor.  If auto, the median distance is used
     %nbinsLogR, nbinsTheta - give the number of bins to use
    %Outputs:
     %fd - 1x(nbinsLogR*nbinsTheta) array holding the feature descriptors
    
    N = size(pts,1);
    Nf = nbinsLogR*nbinsTheta;
    %shift and scale
    ptsShifted = bsxfun(@minus, pts, origin);
    distsSq = dot(ptsShifted',ptsShifted')';
    ptsShifted = ptsShifted(distsSq>0,:);
    distsSq = distsSq(distsSq>0);
    if (scale=='auto')
        scale = median(median(sqrt(pdist2(pts,pts,'euclidean'))));
    end
    distsSq = distsSq * (1/(scale*scale));
    %find and normalize logR and theta
    logR = log2(distsSq) / 2;
    logR = (logR-min(logR)) / (max(logR)-min(logR)+min(abs(logR))/1000);
    theta = atan2(ptsShifted(:,2), ptsShifted(:,1));
    theta = theta/(2*pi) + 1/2;
    %combine logR and theta together to let us bin them easily
%    logR_I = floor(logR * nbinsLogR + nbinsLogR/2 + 1);
    logR_I = floor(logR * nbinsLogR + 1);
%    logR_I = min(max(1,logR_I),nbinsLogR);
    theta_I = floor(theta * nbinsTheta + 1);
    combined = (theta_I-1)*nbinsLogR + logR_I;
    h = hist(combined,1:(Nf));
    fd = h / sum(h);
end
