function [testError] = misclassError(y,y_hat,thresh)
%MISCLASSERROR Compute 0/1 misclassification error for a 2-class problem
% misclassError [out] = MISCLASSERROR(in) computes the 0/1
% misclassification error given predicted and actual output.
%
% Input:
% y = actual class
% y_hat = predicted class
% thresh = probability threshold
%

m = size(y,1);

testError = (1/m) * sum(double(y_hat>=thresh & y==0 | y_hat<=thresh & y==1 ));

end

