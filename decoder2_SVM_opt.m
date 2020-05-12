%In this code we will try to create a much more sophisticated decoder that
%will take into accoubt general structure in the data and wil use
%information from different sessions.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load parameters

load('parameters.mat');

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
%cross validation. for now I write 6 as constans, could be changed to 
%parameter later

indices = 1:6;

successes = 0;
accuracy_per_trial_choosen = zeros(6,1);

max_accuracy = 0;
max_accuracy_std = 0;

for start_bin = 11:18
	for end_bin = start_bin+2:20
		
	fprintf("Current bins: %d-%d\n", start_bin, end_bin);

		for i=1:6
			
			%create the relevant indices
			train_idx = indices(indices~=i);
			
			%finding the significant neurons per vowel
			neurons = significant_neurons(data, M, num_of_targets,training_samples_per_target, train_idx,...
			feature_selection, algo, p_value_threshold, start_bin, end_bin);
			
			%use neurons array to extract the group of siginificant neurons:
			%create data set to activate the SVM
			[data_set_X,data_set_class,data_set_test,data_class_test] = create_data_set_for_SVM(algo,feature_selection,data,M,neurons,...
			training_samples_per_target,samples_per_target,test_samples_per_target,targets, train_idx, i, start_bin, end_bin);
			ecoc_model = fitcecoc(data_set_X,data_set_class);
			total_test_samples = size(data_set_test,1);
			successes = 0;
			label_output_classifier = predict(ecoc_model,data_set_test); 
			for j = 1:size(data_set_test,1)%changed
				if(label_output_classifier{j,1} == data_class_test{j,1})%changed
					successes = successes + 1; 
				end
			end 
			accuracy_per_trial_choosen(i) = successes/total_test_samples;
		end

		algo_accuracy_dist = fitdist(accuracy_per_trial_choosen, 'Normal');
		fprintf("The accuracy of the model is: %.2f\n", algo_accuracy_dist.mean);
		fprintf("The std is %.2f\n", algo_accuracy_dist.sigma);
		
		if algo_accuracy_dist.mean >= max_accuracy
			max_accuracy = algo_accuracy_dist.mean;
			if algo_accuracy_dist.sigma <= max_accuracy_std || max_accuracy_std == 0
				max_accuracy_std = algo_accuracy_dist.sigma;
				opt_start_bin = start_bin;
				opt_end_bin = end_bin;
			end
		end
		
	end
end

fprintf("The maximum accuracy achieved is %.2f\n", max_accuracy);
fprintf("The maximum accrracy std achieved is %.2f\n", max_accuracy_std);
fprintf("Optimal bins: %d-%d\n", opt_start_bin, opt_end_bin);

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
%this function will return a (N,5, 3) matrix, in which cell in the first
%dimension will be "1" if the neuron has a significant response to the corresponding vowel and
%"0" otherwise. The second dimension contains the average bin for a significant neuron
%the third layer will contain the needed info, according to the chosen algorithm

function neurons=significant_neurons(data, M, num_of_targets,training_samples_per_target, train_idx,...
		feature_selection, algo, p_value_threshold, start_bin, end_bin)
	
	%create a cell and fill with 0's
	neurons = cell(M,num_of_targets,3);
	neurons(:,:,:) = {0};
	
	for i = 1:M
		for j = 1:num_of_targets
			%find the baseline vector of the relevant trials
			baseline = baseline_vector(data, i, j,training_samples_per_target, train_idx,...
			start_bin, end_bin);

			%find the max bin, including its predecessor and follower
			max_bins = find_max_bins(data, i, j,training_samples_per_target, train_idx);

			%check for significance. We use p = 0.05/3 since we 
			%compare 3 different vectors to the baseline.
			%we changed it to 0.05/10 for better results
            %we check only if there was a significant deviation from the 
            %change for std
            
            baseline_dist = fitdist(baseline, 'Normal');
            P_min = 1;
            if(baseline_dist.mean + baseline_dist.sigma*2 < mean(max_bins(1, :)))
                for k = 1:(length(max_bins(1,:))-1)
                    [h, p] = ttest2(baseline, max_bins(:, k));
                    if(p < P_min)%chnaged
                        if(p < p_value_threshold)
                            neurons{i, j, 1} = 1;%chnaged
                            neurons{i, j, 2} = mean(max_bins(:, 4));%changed
                        end
                        P_min = p;
                    end
                end
            end
		end
	end
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function creates the data set for the SVM estimator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [data_set_X,data_set_class,data_set_test,data_class_test] = create_data_set_for_SVM(algo,feature_selection,data,M,neurons,...
    training_samples_per_target,samples_per_target,test_samples_per_target,targets, train_idx, test_idx, start_bin, end_bin)
    
    neuron_index = 1:M;
    significant_table = cell2mat(neurons(:,:,1));
    significant_neurons_group = neuron_index(sum(significant_table,2)>0);
    
    data_set_class = cell(sum(training_samples_per_target),1);
    data_class_test = cell(sum(test_samples_per_target),1);
    location = 0;
    location_test = 0;
    
    for  j=1:size(targets,2)
        %need to fill this loop
        [data_set_X_chunk,data_set_test_chunk] = choose_numbers_for_chunk(train_idx,test_idx,j,data,neurons,training_samples_per_target...
        ,significant_neurons_group,test_samples_per_target,algo,feature_selection, start_bin, end_bin);
    
        for k=1:training_samples_per_target(j)
            data_set_class{location+k,1} = convertStringsToChars(targets(j));
        end
        for k=1:test_samples_per_target(j)
            data_class_test{location_test+k,1} = convertStringsToChars(targets(j));
        end
        location = location + training_samples_per_target(j);
        location_test = location_test + test_samples_per_target(j);
        %Make the for loop to fill data set chunck
  
        if(j==1)
            data_set_X = data_set_X_chunk;
            data_set_test = data_set_test_chunk;
        else
            data_set_X = cat(1,data_set_X,data_set_X_chunk);
            data_set_test = cat(1,data_set_test,data_set_test_chunk);
        end
    end
end


function [data_set_X_chunk,data_set_test_chunk] = choose_numbers_for_chunk(train_idx,test_idx,j,data,neurons,training_samples_per_target...
    ,significant_neurons_group,test_samples_per_target,algo,feature_selection, start_bin, end_bin)
    for i = significant_neurons_group
        baseline_train = baseline_vector(data, i, j,training_samples_per_target, train_idx, start_bin, end_bin);
        max_bins_train = find_max_bins(data, i, j,training_samples_per_target, train_idx);
        max_bins_test = find_max_bins(data, i, j,test_samples_per_target, test_idx);
        baseline_test = baseline_vector(data, i, j,test_samples_per_target, test_idx, start_bin, end_bin);
        %create info_per_neuron_per_target
        active_samples = cat(1,max_bins_train(:,3),max_bins_test(:,3));
        baseline_samples = cat(1,baseline_train,baseline_test);
        best_max_bins_with_test_sample = check_val_with_ttest(baseline_samples,active_samples);
        train_size = size(max_bins_train,1);
        switch algo
            case 'standard'
                switch feature_selection
                    case 'max_bin'
                        info_for_train = [baseline_train, max_bins_train(:,1)];
                        info_for_test = [baseline_test, max_bins_test(:,1)];
                    case 'sig_bin'
                        info_for_train = [baseline_train, best_max_bins_with_test_sample(1:train_size)];
                        info_for_test = [baseline_test, best_max_bins_with_test_sample(train_size +1 :end)];
                    case 'all_bin'
                        info_for_train = [baseline_train, max_bins_train(:,1), max_bins_train( :,2), max_bins_train(:,3)];
                        info_for_test = [baseline_test, max_bins_test(:,1), max_bins_test( :,2), max_bins_test(:,3)];
                        
                end
            case 'diff'
                switch feature_selection
                    case 'max_bin'
                        info_for_train = max_bins_train(:,1) - baseline_train;
                        info_for_test = max_bins_test(:,1)-baseline_test;
                    case 'sig_bin'
                        info_for_train = best_max_bins_with_test_sample(1:train_size) - baseline_train;
                        info_for_test = best_max_bins_with_test_sample(train_size + 1:end) - baseline_test;
                    case 'all_bin'
                        info_for_train = [max_bins_train(:,1), max_bins_train( :,2), max_bins_train(:,3)] - baseline_train;
                        info_for_test = [max_bins_test(:,1), max_bins_test( :,2), max_bins_test(:,3)] - baseline_test;
                end
            case 'z_score'
                baseline_dist = fitdist(baseline_train, 'Normal');
                switch feature_selection
                    case 'max_bin'
                        info_for_train = (max_bins_train(:,1) - baseline_dist.mu)/baseline_dist.sigma;
                        info_for_test = (max_bins_test(:,1) - baseline_dist.mu)/baseline_dist.sigma;
                    case 'sig_bin'
                        info_for_train = (best_max_bins_with_test_sample(1:train_size) - baseline_dist.mu)/baseline_dist.sigma;
                        info_for_test = (best_max_bins_with_test_sample(train_size+1:end) - baseline_dist.mu)/baseline_dist.sigma;
                    case 'all_bin'
                        info_for_train = (max_bins_train(:,1:3) - baseline_dist.mu)/baseline_dist.sigma;
                        info_for_test = (max_bins_test(:,1:3) - baseline_dist.mu)/baseline_dist.sigma;
                end      
        end%end of switch algo
        
        if(i == significant_neurons_group(1))
            data_set_X_chunk = info_for_train;%change - problem with saving in times of in significance
            data_set_test_chunk = info_for_test;% fill information per neuron per target 
        else
            data_set_X_chunk = cat(2,data_set_X_chunk,info_for_train);
            data_set_test_chunk = cat(2,data_set_test_chunk,info_for_test);
        end
        
    end%end of for i = significant_neurons_group
end 

%need for these functions:



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%this function return a baseline vector per neuron per trials
%the baseline is defined as the average of the [-0.8,-0.3][s]

function baseline=baseline_vector(data, i, j,num_samples_per_target, cmpute_indexes, start_bin, end_bin)
	
	%we take 80 percent of the trials as training set
	%max_trial = ceil(length(data{i,j}(:,1))*training_precent_from_data);
	max_trial = num_samples_per_target(j);
	%create the baseline vector
	baseline = zeros(max_trial,1);

	for k = 1:max_trial
		baseline(k) = mean(data{i,j}(cmpute_indexes(k), start_bin:end_bin));
	end
	
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%returns a matrix of size (max_trial,3). For every channel and vowel, 
%each row (trial) will be: (max_bin, (max_bin+predecessor)/2, (max_bin+follower)/2)


function max_bins=find_max_bins(data, i, j,num_samples_per_target, compute_indexes)
	
	%we take 80 percent of the trials as training set 
	%max_trial = ceil(length(data{i,j}(:,1))*training_precent_from_data);
	max_trial = num_samples_per_target(j);
	max_bins = zeros(max_trial, 4);
	
	for k = 1:max_trial
		[max_bin_val, max_bin_idx] = max(data{i,j}(compute_indexes(k), 21:30));
		max_bins(k, 1) = max_bin_val;
        %notice that the max_bin_idx is relative to the specified array, so
        %we add 20 as offset
		max_bins(k, 2) = (max_bin_val + data{i,j}(compute_indexes(k), max_bin_idx + 21))/2;
		max_bins(k, 3) = (max_bin_val + data{i,j}(compute_indexes(k), max_bin_idx + 19))/2;
		max_bins(k, 4) = max_bin_idx;
	end
	
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function recive a column of baseline samples a matrix 
%that has columns of active samples(each column is for different sample)
%The function returns the column with the samllest p-value from all the
%columns
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function significant_column = check_val_with_ttest(baseline,active_samples)
    active_columns_num = size(active_samples,2);
    P_min = 1;
    min_p_value_idx = 0;
    for i=1:active_columns_num
        [h, p] = ttest2(baseline, active_samples(:,i));
        if(p <= P_min)%chnaged
            P_min = p;
            min_p_value_idx = i;
        end
    end
    significant_column = active_samples(:,min_p_value_idx);
end