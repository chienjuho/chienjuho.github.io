function [ w iterations ] = perceptron_learn( data_in )
%perceptron_learn Run PLA on the input data
%   Inputs: data_in: Assumed to be a matrix with each row representing an
%                    (x,y) pair, with the x vector augmented with an
%                    initial 1, and the label (y) in the last column
%   Outputs: w: A weight vector (should linearly separate the data if it is
%               linearly separable)
%            iterations: The number of iterations the algorithm ran for
w=zeros(1,11);
iterations=0;
x=data_in(:,1:11);
y=data_in(:,12);
for i=1:100
    y=w*x';
    if sign(data_in(:,12))~=y
        w=w+(y*x);
        iterations=iterations+1;
    end
end
