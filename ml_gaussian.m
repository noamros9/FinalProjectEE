%{
In this algorithm, we try to implement a first try for a neuronal decoder, based
on the firing rates of recorded neurons in the Thalamus.
Generally speaking, we will generate a gaussian distribution for every bin
of every neuron for each target (i.e. a gaussian distribution for the time
interval [-0.5, -0.4] in the target "imagine_a" in cl 4).
After creating all of the aforementioned distributions, we will take new data
(recorded neuronal activity), create the same disributions and then compare
them (we will use Kullback–Leibler divergence as a measurment).
Lastly, we will choose the target that has the minimum aggregated
difference.
Note: we will use frequencies for consistency and compatability
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Defining The parameters in the problem.

data = sct.fr_hz_per_trial_per_cl_per_target;

% targets is how many interpretations options we have
num_targets = length(data);

% M is the number of channels
M = length(data(:,1));

% N is the number of bins - how much time intervals we have
N = length(data{1});


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%calculating the corresponding gaussian distributions, using 75% of the
%data (the rest is for testing and validation)
gaussians = create_gaussians(data, M, N, num_targets);

%Activating the estimation:
choice = ML_G_IID(gaussians, sample, M, N, num_targets, g_title);

%The function disp desplays to the command window the choice
disp(choice)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%The function ML_G_IID calculate the ML estimation for a specific sample

function choice = ML_G_IID(gaussians, sample, M, N, num_targets, g_title)
    decision_values = zeros(1,num_targets);
    for alpha = 1:num_targets
        for i=1:M
            for j = 1:N 
                decision_values(alpha) = descision_values(alpha) + ...
                log(gaussians{i, alpha, j}.sigma) + ...
                ((sample(i, j) -  gaussians{i, alpha, j}.mu)^2)/...
                (2 * gaussians{i, alpha, j}.sigma);
                
            end
        end
    end
    
    [min_val,decision] = min(decision_values);
    choice = g_title{decision,2};
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function gaussians = create_gaussians(data_cell, M, N, num_targets)
    % a cell of dimensions num_of_cl X num_of_targets X num_of_bins
    gaussians = cell(M, num_targets, N);

    %create the corresponding gaussian distributions
    for i = 1:M
        for j = 1:num_targets
            for k = 1:N
                max_samp = ceil(length(data_cell{i,j}(:,k)*0.75));
                freqs = data_cell{i, j}(1: max_samp, k);
                gaussians(i,j,k) = {fitdist(freqs, 'Normal')};
            end
        end
    end
    
end
