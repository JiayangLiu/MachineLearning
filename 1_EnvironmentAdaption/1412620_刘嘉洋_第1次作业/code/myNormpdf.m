%   @Coded by LIUJiayang, 1412620
function [ result ] = myNormpdf( X,MU,SIGMA )
%   Y = myNormpdf(X,MU,SIGMA) returns the pdf of the normal distribution
%   mean MU, standard deviation SIGMA, evaluated at the values in X.
    result = 1/(SIGMA*sqrt(2*pi))*exp(-(X-MU).^2/(2*SIGMA^2));
end
