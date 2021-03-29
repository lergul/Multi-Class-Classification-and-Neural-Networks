function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);


X= [ones(m,1) X]; % adding ones column to matrix X for bias 

hx = sigmoid(X*Theta1');
hx = [ones(m,1) hx]; %% adding ones column to matrix hx for bias 
hx = sigmoid(hx*Theta2');

[training p] = max(hx, [], 2);


end
