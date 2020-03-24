%In this code we will try to create a much more sophisticated decoder that
%will take into accoubt general structure in the data and wil use
%information from different sessions.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%defining the parameters of the algorithm

S1 = load('speech_screening_analysis_beep_session1.mat');
S2 = load('speech_screening_analysis_beep_session2.mat');

%for now, we handle vowels only
targets = ["a","e","i","o","u"];
num_of_targets = size(targets,2);


%creating a struct with all the information of the two ssestions file
data = merge_data(S1, S2, targets, num_of_targets);

% M is the number of channels
M = length(data(:,1));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%finding the significant neurons per vowel
neurons = significant_neurons(data, M, num_of_targets);

display(neurons);





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%this function will merge data from multiple sessions (at the moment-2)
function data=merge_data(S1, S2, targets, num_of_targets)
    %defining the title in these two cases
    g_title1 = S1.g_title;
    g_title2 = S2.g_title;

    %getting the full data.
    full_data1 = S1.sct.fr_hz_per_trial_per_cl_per_target;
    full_data2 = S2.sct.fr_hz_per_trial_per_cl_per_target;

    % taking the data for the interpatations chosen:
    data1 = full_data1( :,sum(g_title1(:,2)==targets,2)==1 );
    data2 = full_data2( :,sum(g_title2(:,2)==targets,2)==1 );
    data = cell(size(data1,1)+size(data2,1),num_of_targets);
    data(1:size(data1,1),:) = data1;
    offset = size(data1,1);
    data((offset+1):offset + size(data2,1),:) = data2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%this function will return a (N,5, 2) matrix, in which cell in the first
%dimension will be "1" if the neuron has a significant response to the corresponding vowel and
%"0" otherwise. The second dimension contains the average bin for a significant neuron

function neurons=significant_neurons(data, M, num_of_targets)
	
	neurons = zeros(18,5,2);
	
	for i = 1:M
		for j = 1:num_of_targets
			%find the baseline vector of the relevant trials
			baseline = baseline_vector(data, i, j, M);

			%find the max bin, including its predecessor and follower
			max_bins = find_max_bins(data, i, j, M);

			%check for significance. We use p = 0.05/3 since we 
			%compare 3 different vectors to the baseline.
			%we changed it to 0.05/10 for better results
            %we check only if there was a significant rise in the freq.
            if(mean(baseline)*1.5 < mean(max_bins(1, :)))
                for k = 1:length(max_bins(1,:))
                    [h, p] = ttest2(baseline, max_bins(:, k));
                    if(p < 0.05/10)
                        neurons(i, j, 1) = 1;
                        neurons(i, j, 2) = mean(max_bins(:, 4));
                    end
                end
            end
		end
	end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%this function return a baseline vector per neuron per trials
%the baseline is defined as the average of the [-0.8,-0.3][s]

function baseline=baseline_vector(data, i, j, M)
	
	%we take 80 percent of the trials as training set
	max_trial = ceil(length(data{i,j}(:,1))*0.8);
	
	%create the baseline vector
	baseline = zeros(max_trial,1);

	for k = 1:max_trial
		baseline(k) = mean(data{i,j}(k, 13:17));
	end
	
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%returns a matrix of size (max_trial,3). For every channel and vowel, 
%each row (trial) will be: (max_bin, (max_bin+predecessor)/2, (max_bin+follower)/2)


function max_bins=find_max_bins(data, i, j, M)
	
	%we take 80 percent of the trials as training set 
	max_trial = ceil(length(data{i,j}(:,1))*0.8);
	
	max_bins = zeros(max_trial, 4);
	
	for k = 1:max_trial
		[max_bin_val, max_bin_idx] = max(data{i,j}(k, 21:30));
		max_bins(k, 1) = max_bin_val;
        %notice that the max_bin_idx is relative to the specified array, so
        %we add 20 as offset
		max_bins(k, 2) = (max_bin_val + data{i,j}(k, max_bin_idx + 21))/2;
		max_bins(k, 3) = (max_bin_val + data{i,j}(k, max_bin_idx + 19))/2;
		max_bins(k, 4) = max_bin_idx;
	end
	
end




