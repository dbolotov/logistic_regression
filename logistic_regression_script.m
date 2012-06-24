%Logistic Regression

%predicts output using fminunc


%Input must contain feature columns followed by dependent variable column at end
data = load('class_function_01.txt');

%logistic regression parameters
alpha = 0.01;
num_iters = 1000;

%percentage of data to use for training
train_perc = .85;

%extract columns to use
X = data(:,1:end-1);
X_orig = X;
y = data(:,end);

%split into training and test sets:
test_rows = round(size(X,1)*(1-train_perc)); %number of rows to use in test set
X_test = X(1:test_rows,:); y_test = y(1:test_rows,:);%this is the test set
X = X(test_rows+1:end,:); y = y(test_rows+1:end,:);%this is the training set

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

%Add intercept term to X
X = [ones(size(X,1), 1) X];
X_test = [ones(size(X_test,1), 1) X_test];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

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

p_train = (sigmoid(X*theta) >= 0.5);
fprintf('Training set accuracy: %f\n', mean(double(p_train == y)) * 100);

p_test = (sigmoid(X_test*theta) >= 0.5);
fprintf('Test set accuracy: %f\n', mean(double(p_test == y_test)) * 100);





















