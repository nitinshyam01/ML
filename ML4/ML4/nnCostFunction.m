 function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
temp1=Theta1;
temp1(:,1)=zeros(hidden_layer_size,1);
temp2=Theta2;
temp2(:,1)=zeros(num_labels,1);
y_multi=zeros(length(y),num_labels);
for i=1:num_labels
y_multi(:,i)=y==i;
end
X=[ones(m,1),X];
g_theta1=sigmoid(X*(Theta1'));
g_theta1=[ones(m,1),g_theta1];
g_theta2=sigmoid(g_theta1*(Theta2'));

J=J-sum(sum((y_multi).*log(g_theta2)+(1-y_multi).*log(1-g_theta2)),2)/m;
J=J+(lambda/(2*m))*(sum(sum(temp1.*temp1),2)+sum(sum(temp2.*temp2),2));
Theta2_grad=(((g_theta2-y_multi)'*g_theta1)/m)+lambda*temp2/m;
Theta1_gradt=(((Theta2'*(g_theta2-y_multi)').*(g_theta1)'.*(1-(g_theta1)'))*X/m);
Theta1_grad=Theta1_gradt(2:end,:)+lambda*temp1/m;

%epsilon=1e-4;
%for i=1:length(nn_params);
%epsilonplus=nn_params;
%5epsilonplus(i)=epsilonplus(i)+epsilon;
%epsilonminus=nn_params;
%epsilonminus(i)=epsilonminus(i)-epsilon;
%grad_approx(i)=(J(epsilonplus)-J(epsilonminus))/(2*epsilon)










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
