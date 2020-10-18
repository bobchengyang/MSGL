function [ff_CMN] = ff_CMN(fu,q,u)
%FF_CMN Summary of this function goes here
%   Detailed explanation goes here

sum_fu = sum(fu(:,1));

a = q * ( u - sum_fu );

b = ( 1 - q ) * sum_fu;

ff_CMN = a * fu(:,1) ./ ( a * fu(:,1) + b * fu(:,2) );

end

