function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z. z can be a matrix, vector or
%   scalar

g = 1./(1+exp(-z));
end
