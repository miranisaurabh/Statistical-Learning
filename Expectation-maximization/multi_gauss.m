function [prob] = multi_gaussian(x,mu, sigma)

inv_sigma = inv(sigma);
var = diag(inv_sigma)';

power_nD = ((x - mu).^2).*var;
power = sum(power_nD,2);
dim = size(sigma,2);
prob = exp(-0.5*power)/(sqrt(((2*pi)^dim)*det(sigma)));
end

