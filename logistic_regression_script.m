%Logistic Regression
%
% Perform logistic regression; predict output (2-class variable) using
% unconstrained optimization routine (fminunc)
%
% Functions used: sigmoid.m, costFunction.m
%
% Code based on ml-class.org Ex.2
% 

%to do:
%generate confusion matrix
%display sensitivity and specificity


%Input must contain feature columns followed by dependent variable column at end
data = load('class_function_01.txt');

%logistic regression parameters
alpha = 0.01;
num_iters = 1000;

%percentage of data to use for training
train_perc = .77;

%extract columns to use
X = data(:,1:end-1);
y = data(:,end);

%split into training and test sets:
test_rows = round(size(X,1)*(1-train_perc)); %number of rows to use in test set
X_test = X(1:test_rows,:); y_test = y(1:test_rows,:);%this is the test set
X = X(test_rows+1:end,:); y = y(test_rows+1:end,:);%this is the training set

%Add intercept term to X
X = [ones(size(X,1), 1) X];
X_test = [ones(size(X_test,1), 1) X_test];

% Initialize fitting parameters
initial_theta = zeros(size(X,2), 1);

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

p_train = double(sigmoid(X*theta) >= 0.5);
fprintf('Training set accuracy: %f\n', mean(double(p_train == y)) * 100);

p_test = double(sigmoid(X_test*theta) >= 0.5);
fprintf('Test set accuracy: %f\n', mean(double(p_test == y_test)) * 100);

%confusion matrix
cm = confusionmat(y_test,p_test);
sens = cm(1,1) / (cm(1,1) + cm(1,2)); %ability to identify positive class
spec = cm(2,2) / (cm(2,2) + cm(2,1)); %ability to identify negative class

















