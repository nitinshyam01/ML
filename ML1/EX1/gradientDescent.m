function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
theta0=0;
theta1=0;
for i=1:m
theta0=theta0+((X(i,1)*theta(1,1)+X(i,2)*theta(2,1))-y(i,1));
theta1=theta1+((X(i,1)*theta(1,1)+X(i,2)*theta(2,1))-y(i,1))*X(i,2);
end
theta0=theta0/m;
theta1=theta1/m;
Delta=[theta0;theta1];

theta=theta-alpha*Delta;






    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
