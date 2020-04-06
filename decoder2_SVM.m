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
training_precent_from_data = 0.80;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
training_samples_per_target = zeros(1,size(targets,2));
samples_per_target = zeros(1,size(targets,2));
for i=1:size(targets,2)
    for neuron_num = 1:M
        max_idx = ceil(length(data{neuron_num,i}(:,1))*training_precent_from_data);
        if(training_samples_per_target(i)==0)
            training_samples_per_target(i) = max_idx;
            samples_per_target(i) = length(data{neuron_num,i}(:,1));
        end
        if(training_samples_per_target(i)>max_idx)
            training_samples_per_target(i) = max_idx;
            samples_per_target(i) = length(data{neuron_num,i}(:,1));
        end
    end
end
test_samples_per_target = samples_per_target - training_samples_per_target;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%finding the significant neurons per vowel
neurons = significant_neurons(data, M, num_of_targets,training_samples_per_target);
display(neurons);

%use neurons array to extract the group of siginificant neurons:
%create data set to activate the SVM
[data_set_X,data_set_class,data_set_test,data_class_test] = create_data_set_for_SVM(1,data,M,neurons,training_samples_per_target,samples_per_target,test_samples_per_target,targets);
ecoc_model = fitcecoc(data_set_X,data_set_class);
total_test_samples = size(data_set_test,1);
successes = 0;
label_output_classifier = predict(ecoc_model,data_set_test); 
for i = 1:size(data_set_test,1)
    if(label_output_classifier{i,1} == data_class_test{i,1})
        successes = successes + 1; 
    end
end
fprintf("The accuracy in for the model is: %.2f\n",successes/total_test_samples)




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

function neurons=significant_neurons(data, M, num_of_targets,training_samples_per_target)
	
	neurons = zeros(M,num_of_targets,2);
	
	for i = 1:M
		for j = 1:num_of_targets
			%find the baseline vector of the relevant trials
			baseline = baseline_vector(data, i, j, M,training_samples_per_target);

			%find the max bin, including its predecessor and follower
			max_bins = find_max_bins(data, i, j, M,training_samples_per_target);

			%check for significance. We use p = 0.05/3 since we 
			%compare 3 different vectors to the baseline.
			%we changed it to 0.05/10 for better results
            %we check only if there was a significant rise in the freq.
            if(mean(baseline)*1.5 < mean(max_bins(1, :)))%change for std
                for k = 1:(length(max_bins(1,:))-1)
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

function baseline=baseline_vector(data, i, j, M,training_samples_per_target)
	
	%we take 80 percent of the trials as training set
	%max_trial = ceil(length(data{i,j}(:,1))*training_precent_from_data);
	max_trial = training_samples_per_target(j);
	%create the baseline vector
	baseline = zeros(max_trial,1);

	for k = 1:max_trial
		baseline(k) = mean(data{i,j}(k, 13:17));
	end
	
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%returns a matrix of size (max_trial,3). For every channel and vowel, 
%each row (trial) will be: (max_bin, (max_bin+predecessor)/2, (max_bin+follower)/2)


function max_bins=find_max_bins(data, i, j, M,training_samples_per_target)
	
	%we take 80 percent of the trials as training set 
	%max_trial = ceil(length(data{i,j}(:,1))*training_precent_from_data);
	max_trial = training_samples_per_target(j);
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



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function creates the data set for the SVM estimator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [data_set_X,data_set_class,data_set_test,data_class_test] = create_data_set_for_SVM(mode,data,M,neurons,...
    training_samples_per_target,samples_per_target,test_samples_per_target,targets)

    neuron_index = 1:M;
    significant_neurons_group = neuron_index(sum(neurons(:,:,1),2)>0);
    significant_neurons_index = 1:size(significant_neurons_group,2);
    
    N = size(data{1,1},2);
    
    data_set_class = cell(sum(training_samples_per_target),1);
    data_class_test = cell(sum(test_samples_per_target),1);
    location = 0;
    location_test = 0;
    for  j=1:size(targets,2)
        if(mode==1)
            data_set_X_chunk  = zeros(training_samples_per_target(j),2*size(significant_neurons_group,2));
            data_set_test_chunk = zeros(test_samples_per_target(j),2*size(significant_neurons_group,2));
        end
        if (mode==2)
            data_set_X_chunk = zeros(training_samples_per_target(j),size(significant_neurons_group,2));
            data_set_test_chunk = zeros(test_samples_per_target(j),size(significant_neurons_group,2));
        end
        
        for k=1:training_samples_per_target(j)
            data_set_class{location+k,1} = convertStringsToChars(targets(j));
        end
        for k=1:test_samples_per_target(j)
            data_class_test{location_test+k,1} = convertStringsToChars(targets(j));
        end
        location = location + training_samples_per_target(j);
        location_test = location_test + test_samples_per_target(j); 
        
        for i = significant_neurons_group
            baseline_add = mean(data{i,j}(1:training_samples_per_target(j),13:17),2);
            max_bin_add = max(data{i,j}(1:training_samples_per_target(j),21:30),[],2);
            baseline_add_test = mean(data{i,j}(training_samples_per_target(j)+1:samples_per_target(j),13:17),2);
            max_bin_add_test = max(data{i,j}(training_samples_per_target(j)+1:samples_per_target(j),21:30),[],2);
            %doesnt support reaction location
            if(mode==1)
                data_set_X_chunk(:,significant_neurons_index(i == significant_neurons_group)*2 - 1)  = baseline_add;
                data_set_X_chunk(:,significant_neurons_index(i == significant_neurons_group)*2) = max_bin_add;
                data_set_test_chunk(:,significant_neurons_index(i == significant_neurons_group)*2 - 1)  = baseline_add_test;
                data_set_test_chunk(:,significant_neurons_index(i == significant_neurons_group)*2) = max_bin_add_test;
            end
            if (mode==2)
                data_set_X_chunk(:,significant_neurons_index(i == significant_neurons_group)) = max_bin_add - baseline_add;
                data_set_test_chunk(:,significant_neurons_index(i == significant_neurons_group)) = max_bin_add_test - baseline_add_test;
            end
        end
        if(j==1)
            data_set_X = data_set_X_chunk;
            data_set_test = data_set_test_chunk;
        else
            data_set_X = cat(1,data_set_X,data_set_X_chunk);
            data_set_test = cat(1,data_set_test,data_set_test_chunk);
        end
    end 
end