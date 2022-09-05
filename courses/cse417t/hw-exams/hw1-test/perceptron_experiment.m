function [ num_iters bounds] = perceptron_experiment ( N, d, num_samples )
%perceptron_experiment Code for running the perceptron experiment in HW1
%   Inputs: N is the number of training examples
%           d is the dimensionality of each example (before adding the 1)
%           num_samples is the number of times to repeat the experiment
%   Outputs: num_iters is the # of iterations PLA takes for each sample
%            bounds is the theoretical bound on the # of iterations
%              for each sample
%      (both the outputs should be num_samples long)
num_iters=[];
bounds=[];
for i=1:num_samples
    x1=ones(100,1);
    x2=2*rand(N,d)-1;
    x=[x1 x2];
    wStar1=0;
    wStar2=rand(1,10);
    wStar=[wStar1 wStar2];
    y=x*wStar';
    data_in=[x y];
    [ w iterations ] = perceptron_learn( data_in );
    num_iters=cat(1,num_iters,iterations);
    p=min((y.*(x*wStar')).^2);
    R=max(norm(x.^2));
    bounds=[bounds R.*wStar.^2/p];
end
end
