%{
In this code, we try to implement a first try for a neuronal decoder, based
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

% a cell of dimensions num_of_cl X num_of_targets X num_of_bins
distributions = cell(length(sct.fr_hz_per_trial_per_cl_per_target(:,1)),...
length(sct.fr_hz_per_trial_per_cl_per_target), ...
length(sct.fr_hz_per_trial_per_cl_per_target{1}));

%create the corresponding gaussian distributions
for i = 1:length(sct.fr_hz_per_trial_per_cl_per_target(:,1))
    for j = 1:length(sct.fr_hz_per_trial_per_cl_per_target)
        for k = 1:length(sct.fr_hz_per_trial_per_cl_per_target{1})
            freqs = sct.fr_hz_per_trial_per_cl_per_target{i, j}(:, k);
            distributions(i,j,k) = {fitdist(freqs, 'Normal')};
        end
    end
end

%now, let's compare the new data to the distributions we created




