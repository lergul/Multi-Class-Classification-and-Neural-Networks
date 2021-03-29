function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i


% Some useful variables
m = size(X, 1); %5000
n = size(X, 2); %400

% Theta values set zeros matrix
all_theta = zeros(num_labels, n + 1); 

% Add ones to the X data matrix
X = [ones(m, 1) X];

%     % Set Initial theta
     initial_theta = zeros(n + 1, 1);
    
     % Set options for fminunc
     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost

    
 for i=1 : num_labels

    [theta]= fmincg (@(t)(lrCostFunction(t, X, (y == i), lambda)),initial_theta, options);
    
    all_theta(i,:)=theta; 

 end
 

end
