%this file will contain the settings and parameters
%for the different kinds of algorithms we may want 
%to run

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%feature selection:
%max_bin = choose the bin with the max frequency
%sig_bin = choose the bin which was the cause for the 
%neuron to be declared significant
%all_bin = choose all the 3 relevant bins (max, avg(max+prev), avg(max+fol))

feature_selection = 'all_bin';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%algo: The algorithm might run in 3 different forms:
%standard - we pass the baseline and the bin(s) chosen
%diff - we pass the diffrence between thee baseline and 
%the bin(s) chosen. It leads to one less feature
%z_score - we pass the zscore of the bin(s), in relation
%to the baseline

algo = 'z_score';

switch algo
	case 'standard'
		switch feature_selection
			case 'max_bin'
				p_value_threshold = 0.05/4;
				start_bin = 11;
				end_bin = 16;
			case 'sig_bin'
				p_value_threshold = 0.05/4;
			case 'all_bin'
				p_value_threshold = 0.05/3;
				start_bin = 13;
				end_bin = 16;
		end
	case 'diff'
		switch feature_selection
			case 'max_bin'
				p_value_threshold = 0.05/25;
				start_bin = 11;
				end_bin = 19;
			case 'sig_bin'
				p_value_threshold = 0.05/4;
			case 'all_bin'
				p_value_threshold = 0.05/3;
				start_bin = 11;
				end_bin = 20;
		end
	case 'z_score'
		switch feature_selection
			case 'max_bin'
				p_value_threshold = 0.05/13;
				start_bin = 18;
				end_bin = 20;
			case 'sig_bin'
				p_value_threshold = 0.05/4;
			case 'all_bin'
				p_value_threshold = 0.05/3;
				start_bin = 16;
				end_bin = 20;
		end      
end

save parameters.mat;

