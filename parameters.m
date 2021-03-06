%this file will contain the settings and parameters
%for the different kinds of algorithms we may want 
%to run

%%%%%%%%%%%%%%%%%%%%%%%%%% Section 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\
%Determine the action need to be decoded:
starter = ""; %can be only im- for imagine, au- for audio, and "" for Speech
%Determin the percent of data that will be used for training.
training_precent_from_data = 0.75;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%feature selection:
% 1.max_bin = choose the bin with the max frequency
% 3.max_bin_prev_pred = choose all the 3 relevant bins (max, avg(max+prev), avg(max+fol))

feature_selection = 'max_bin';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%algo: The algorithm might run in 4 different forms:
% 1.standard - we pass the baseline and the bin(s) chosen
% 2.diff - we pass the diffrence between thee baseline and 
% the bin(s) chosen. It leads to one less feature
% 3.z_score - we pass the zscore of the bin(s), in relation
% to the baseline
% 4.pca = choose the bin which was the cause for the 
% neuron to be declared significant


algo = 'pca';

p_value_threshold = 0.05/3;

%pca paramters - which bins to check and how many principal components to take into consideration
pca_bins = 11:30;
num_pca_components = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%% Section 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%% Section 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch algo
	case 'standard'
		switch feature_selection
			case 'max_bin'
				start_bin = 13;
				end_bin = 17;
			case 'max_bin_prev_pred'
				start_bin = 14;
				end_bin = 17;
		end
	case 'diff'
		switch feature_selection
			case 'max_bin'
				start_bin = 13;
				end_bin = 15;
			case 'max_bin_prev_pred'
				start_bin = 16;
				end_bin = 20;
		end
	case 'z_score'
		switch feature_selection
			case 'max_bin'
				start_bin = 18;
				end_bin = 20;
			case 'max_bin_prev_pred'
				start_bin = 16;
				end_bin = 20;
		end   
	case 'pca'
		start_bin = 13;
		end_bin = 17;
end
%%%%%%%%%%%%%%%%%%%%%%%%%% Section 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save parameters.mat;

