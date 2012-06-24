function g = confMatrix(T,P)
%CONFMATRIX Compute confusion matrix for 2-class problem
% [cm] = CONFMATRIX(T,P) computes the 2x2 confusion matrix using true class 
% (T) and predicted class (P) values
%
%T = true conversion value
%P = predicted conversion value
%T and P must be of same size, and can only take on values of 0 or 1

g = zeros(2); 

for p = 1:length(T)
	if T(p) == 0 && P(p) == 0
		g(1,1) = g(1,1)+1;
	end
	
	if T(p) == 1 && P(p) == 1
		g(2,2) = g(2,2)+1;
	end
	
	if T(p) == 1 && P(p) == 0
		g(1,2) = g(1,2)+1;
	end
	
	if T(p) == 0 && P(p) == 1
		g(2,1) = g(2,1)+1;
	end	
	
end

g = g'; 

end