function [theta,y_hat_train,y_hat_test,cm,sens,spec] = logistic_regression(dataset,train_frac,thresh,metrics_flag)

% LOGISTIC_REGRESSION performs classification for a 2-class dependent
% variable using unconstrained optimization routine (fminunc)
%
% [theta,y_hat_train,y_hat_test,cm,sens,spec] = LOGISTIC_REGRESSION(dataset,train_frac,thresh,metrics_flag)
%
% Input:
% dataset: .txt format, without header
% train_frac: fraction of dataset to use for training (0 to 1)
% thresh: threshold for classifying hypothesis output
% verb_flag: display regression results (0 = don't display, 1 = do display)
%
% Functions used: sigmoid.m, costFunction.m, misclassError.m, confMatrix.m
%
% Code based on ml-class.org Ex.2


%Input must contain feature columns followed by dependent variable column at end
data = load(dataset);

%extract columns to use
X = data(:,1:end-1);
y = data(:,end);

%split into training and test sets:
test_rows = round(size(X,1)*(1-train_frac)); %number of rows to use in test set
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
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

%prediction accuracy
y_hat_train = double(sigmoid(X*theta) >= thresh);
y_hat_test = double(sigmoid(X_test*theta) >= thresh);

%confusion matrix, sensitivity, specificity
cm = confMatrix(y_test,y_hat_test);
sens = cm(1,1) / (cm(1,1) + cm(1,2)); %ability to identify positive class
spec= cm(2,2) / (cm(2,2) + cm(2,1)); %ability to identify negative class

testError = misclassError(y_test,sigmoid(X_test*theta),thresh); %0/1 misclassification error on test set

if metrics_flag == 1 %Display metrics
    fprintf('Theta:\n'); disp(theta);

    fprintf('Training set accuracy: %f\n', mean(double(y_hat_train == y)) * 100);
    fprintf('Test set accuracy: %f\n', mean(double(y_hat_test == y_test)) * 100);

    fprintf('Confusion Matrix:\n');disp(cm)
    fprintf('Sensitivity: %g\n',sens);
    fprintf('Specificity: %g\n',spec);

    fprintf('Misclassification error on test set: %g\n',testError);
end

end














