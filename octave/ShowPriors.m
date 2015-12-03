function ShowPriors(mu, sigma, a)

N = size(mu, 2);
t = linspace(0, 2*pi);
markers = {'-r', '-b', '-r', '-b', '-g', '-g', '-c', '-m'};
figure
title(sprintf('Prior Distributions (a=%d)', a));
hold on
for i=1:N
	[V, D] = eig(reshape(sigma(a, i, :, :), 2, 2));
	lambda = sqrt(D);
	width = lambda(1, 1);
	height = lambda(2, 2);
	angle = atan2(V(2, 1), V(1, 1));
	plot(mu(a, i, 1) + width*cos(t), -mu(a, i, 2) - height*sin(t+angle), markers{i}, 'LineWidth', 1.5);
end
axis equal
hold off
