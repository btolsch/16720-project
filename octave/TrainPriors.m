%could be called by loading variables: load train210.mat
function [p_a, p_xi_mu, p_xi_var] = TrainPriors(aTrain, LTrain)

numArtic = max(aTrain) - min(aTrain) + 1;
aCounts = zeros(numArtic, 1);
N = size(aTrain, 1);
M = size(LTrain, 2) - 1;

%convert positions to offsets relative to center
LOffsets = zeros(N, M, 2);
for i=1:N
	LOffsets(i, :, :) = LTrain(i, 2:end, :) - LTrain(i, 1, :);
end

%compute p(a) distribution
for i=1:N
	j = aTrain(i);
	aCounts(j) += 1;
end
p_a = aCounts / N;

%compute p(x^i | x^0, a)
%start by computing the mean for each 'a' and 'i' where i is an index into L except 0
p_xi_mu = zeros(numArtic, M, 2);
muCounts = zeros(numArtic, M);
for i=1:M
	for j=1:N
		k = aTrain(j);
		p_xi_mu(k, i, :) += LOffsets(j, i, :);
		muCounts(k, i) += 1;
	end
end
p_xi_mu = p_xi_mu ./ muCounts;

%compute the covariance
p_xi_var = zeros(numArtic, M, 2, 2);
for i=1:M
	for j=1:N
		k = aTrain(j);
		p_xi_var(k, i, 1, 1) += (LOffsets(j, i, 1) - p_xi_mu(k, i, 1))*(LOffsets(j, i, 1) - p_xi_mu(k, i, 1));
		p_xi_var(k, i, 1, 2) += (LOffsets(j, i, 1) - p_xi_mu(k, i, 1))*(LOffsets(j, i, 2) - p_xi_mu(k, i, 2));
		p_xi_var(k, i, 2, 1) += (LOffsets(j, i, 2) - p_xi_mu(k, i, 2))*(LOffsets(j, i, 1) - p_xi_mu(k, i, 1));
		p_xi_var(k, i, 2, 2) += (LOffsets(j, i, 2) - p_xi_mu(k, i, 2))*(LOffsets(j, i, 2) - p_xi_mu(k, i, 2));
	end
end
p_xi_var = p_xi_var ./ (muCounts - 1);
