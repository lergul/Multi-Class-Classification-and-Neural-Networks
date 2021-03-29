function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization

% Initialize some useful values
m = length(y); % number of training examples

%%% Inıtialize values for cost and gradient 
J = 0;
grad = zeros(size(theta));

%%% Initial values
gd = zeros(size(theta));
hx= sigmoid(X*theta);
reg= 0;

for i = 2:size(theta)
    reg = reg + theta(i)^2;
    gd(i) = lambda/m*theta(i);
end

J = (-y'*log(hx)-(1-y)'*log((1-hx)))/m+reg*lambda/2/m; %cost

grad = X'*(hx-y)/m+gd;

grad=grad(:); %gradient




end
