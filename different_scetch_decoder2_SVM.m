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
S3 = load('speech_screening_analysis_beep_session3.mat');
S4 = load('speech_screening_analysis_beep_session6.mat');
S5 = load('speech_screening_analysis_beep_session8.mat');
S6 = load('speech_screening_analysis_session6.mat'); 
S7 = load('speech_screening_analysis_session7.mat'); 
S8 = load('speech_screening_analysis_session8.mat'); 
S9 = load('speech_screening_analysis_session9.mat');
%for now, we handle vowels only
targets = ["a","e","i","o","u"];
num_of_targets = size(targets,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%example of use of the new merge data function.
data_Structs = {S1,S2,S3,S4,S5,S6,S7,S8,S9};
beep_start = [1,1,1,0,0,0,0,0,0];
data = merge_data_from_cell(data_Structs,beep_start, targets, num_of_targets);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%creating a struct with all the information of the two ssestions file
%data = merge_data(S1, S2, targets, num_of_targets);

% M is the number of channels
M = length(data(:,1));
training_precent_from_data = 0.80;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
training_samples_per_target = zeros(1,size(targets,2));
samples_per_target = zeros(1,size(targets,2));
for i=1:size(targets,2)
    for neuron_num = 1:M
        max_idx = ceil(length(data{neuron_num,i}(:,1))*training_precent_from_data);
        if(max_idx == length(data{neuron_num,i}(:,1)))
            fprintf("WARNING:training_precent_from_data is to high to create test set");
        end
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
number_of_CV = 6; %number of cross validation.
all_possible_tests_sets = number_of_test_sets(samples_per_target,test_samples_per_target);
successes = 0;
accuracy_per_trial_choosen = zeros(number_of_CV,1);
for possible= 1:number_of_CV
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %NOTICE The next function ASSUMES that all the bins are of constant
    %size - if not valid errors can occur
    data_set_per_target = create_data_set_per_target(data,samples_per_target,M);%cell, in every place the trails in matrix form
    %indexes_as_test = choose_indexes(num_of_targets,all_possible_tests_sets);
    indexes_as_test = cell(num_of_targets,1);
    indexes_as_test(:,:) = {possible};
    [training_set_per_target,test_set_per_target] = arrange_data_set(data_set_per_target,indexes_as_test);
    [training_set,training_labels] = create_set_samples(training_set_per_target,targets,training_samples_per_target);
    [mean_baseline_per_neuron,std_baseline_per_neuron] = create_total_mean_baseline(training_set,start_bin,end_bin);
    [test_set,test_labels] = create_set_samples(test_set_per_target,targets,test_samples_per_target);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    neurons = significant_neurons(training_set_per_target,M,start_bin,end_bin,p_value_threshold);
    training_set_as_rows = create_data_for_SVM(training_set,neurons,algo,feature_selection, start_bin, end_bin,mean_baseline_per_neuron,std_baseline_per_neuron);
    test_set_as_rows = create_data_for_SVM(test_set,neurons,algo,feature_selection, start_bin, end_bin,mean_baseline_per_neuron,std_baseline_per_neuron);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    ecoc_model = fitcecoc(training_set_as_rows,training_labels);
	total_test_samples = size(test_set_as_rows,1);
	successes = 0;
	label_output_classifier = predict(ecoc_model,test_set_as_rows); 
    for j = 1:size(test_set_as_rows,1)%changed
        if(label_output_classifier{j,1} == test_labels{j,1})%changed
            successes = successes + 1;
        end
    end
    accuracy_per_trial_choosen(possible) = successes/total_test_samples;
end

algo_accuracy_dist = fitdist(accuracy_per_trial_choosen, 'Normal');
fprintf("The accuracy of the model is: %.2f\n", algo_accuracy_dist.mean);
fprintf("The std is %.2f\n", algo_accuracy_dist.sigma);



function set_as_rows = create_data_for_SVM(set,neurons,algo,feature_selection, start_bin, end_bin,mean_baseline_per_neuron,std_baseline_per_neuron)
    samples = size(set,3);
    for i=1:samples
        row = feature_selector(neurons,set(:,:,i),algo,feature_selection, start_bin, end_bin,mean_baseline_per_neuron,std_baseline_per_neuron);
        if(i == 1)
            set_as_rows = row;
        else
            set_as_rows = cat(1,set_as_rows,row);
        end
    end  
end
function  neurons = significant_neurons(training_set_per_target,M,start_bin,end_bin,p_value_threshold)
    %create a cell and fill with 0's
    num_of_targets = size(training_set_per_target,1);
	neurons = cell(M,num_of_targets,3);
	neurons(:,:,:) = {0};
    for j = 1:num_of_targets
        for i = 1:M
            baseline = basline_vector(i,training_set_per_target{j,1},start_bin,end_bin);
            max_bins = find_max_bins( i,training_set_per_target{j,1});
            
            %check for significance. We use p = 0.05/3 since we 
			%compare 3 different vectors to the baseline.
			%we changed it to 0.05/10 for better results
            %we check only if there was a significant deviation from the 
            %change for std
            baseline_dist = fitdist(baseline, 'Normal');
            neurons{i,j,3} = baseline_dist;
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



function max_bins = find_max_bins(i,training_set_per_target_chunk)
    samples_num = size(training_set_per_target_chunk,3);
    max_bins = zeros(samples_num,4);
    for sample = 1:samples_num
        [max_bin_val, max_bin_idx] = max(training_set_per_target_chunk(i,21:30,sample));
        max_bins(sample, 1) = max_bin_val;
        %notice that the max_bin_idx is relative to the specified array, so
        %we add 20 as offset
		max_bins(sample, 2) = (max_bin_val + training_set_per_target_chunk(i,max_bin_idx + 21,sample))/2;
		max_bins(sample, 3) = (max_bin_val + training_set_per_target_chunk(i,max_bin_idx + 19,sample))/2;
		max_bins(sample, 4) = max_bin_idx;
    end
end
function [mean_baseline_per_neuron,std_baseline_per_neuron] = create_total_mean_baseline(training_set,start_bin,end_bin)
    M = size(training_set,1);
    mean_baseline_per_neuron = zeros(M,1);
    std_baseline_per_neuron = zeros(M,1);
    for i = 1:M
        baseline = basline_vector(i,training_set,start_bin,end_bin);
        mean_baseline_per_neuron(i,1) = mean(baseline);
        std_baseline_per_neuron(i,1) = std(baseline);
    end
end
function baseline = basline_vector(i,training_set_per_target_chunk,start_bin,end_bin)
    samples_num = size(training_set_per_target_chunk,3);
    baseline = zeros(samples_num,1);
    for sample = 1:samples_num
        baseline(sample,1) = mean(training_set_per_target_chunk(i,start_bin:end_bin,sample));
    end

end
%return a row vector of the variable to classify
function features = feature_selector(neurons,sample,algo,feature_selection, start_bin, end_bin,mean_baseline_per_neuron,std_baseline_per_neuron)
    M = size(neurons,1);
    neuron_index = 1:M;
    significant_table = cell2mat(neurons(:,:,1));
    significant_neurons_group = neuron_index(sum(significant_table,2)>0);
    for i = significant_neurons_group
        baseline_val = mean(sample(i,start_bin:end_bin));
        [max_bin_val, max_bin_idx] = max(sample(i,21:30));
        %notice that the max_bin_idx is relative to the specified array, so
        %we add 20 as offset
		mean_max_up = (max_bin_val + sample(i,max_bin_idx + 21))/2;
		mean_max_down = (max_bin_val + sample(i,max_bin_idx + 19))/2;
        switch algo
            case 'standard'
                switch feature_selection
                    case 'max_bin'
                         info = [ baseline_val ,  max_bin_val];
                    case 'sig_bin'
                        %STILL NEED TO THINK WHAT TO DO
                    case 'all_bin'
                         info = [baseline_val, max_bin_val, mean_max_up, mean_max_down];
                end
            case 'diff'
                switch feature_selection
                    case 'max_bin'
                        info = max_bin_val - baseline_val;
                    case 'sig_bin'
                        %STILL NEED TO THINK WHAT TO DO
                    case 'all_bin'
                        info = [max_bin_val, mean_max_up, mean_max_down] - baseline_val;
                end
            case 'z_score'
                
                switch feature_selection
                    case 'max_bin'
                        info = (max_bin_val - mean_baseline_per_neuron(i,1))/std_baseline_per_neuron(i,1);
                    case 'sig_bin'
                        %STILL NEED TO THINK WHAT TO DO
                    case 'all_bin'
                        info = ([max_bin_val, mean_max_up, mean_max_down] -  mean_baseline_per_neuron(i,1))./std_baseline_per_neuron(i,1);
                end
        end%end of switch algo
        
        if(i == significant_neurons_group(1))
            features = info;
        else
            features = cat(2,features,info);
        end
    end
end
function [set,labels] = create_set_samples(sets_per_target,targets,samples_per_target)
    num_of_targets = size(targets,2);
    for target = 1:num_of_targets
        target_cell = cell(samples_per_target(target),1);
        target_cell(:,:) = {convertStringsToChars(targets(target))};
        if(target == 1)
            labels = target_cell;
            set = sets_per_target{target,1};
        else
            labels = cat(1,labels,target_cell);
            set = cat(3,set,sets_per_target{target,1});
        end
    end
end

function [training_set_per_target,test_set_per_target] = arrange_data_set(data_set_per_target,indexes_as_test)
    num_of_targets = size(indexes_as_test,1);
    training_set_per_target = cell(num_of_targets,1);
    test_set_per_target = cell(num_of_targets,1);
    for i=1:num_of_targets
        samples_per_target = size(data_set_per_target{i,1},3);
        indexes = 1:samples_per_target;
        mask = create_mask_from_vec(indexes,indexes_as_test{i,1});
        test_set_per_target{i,1} = data_set_per_target{i,1}(:,:,mask);
        training_set_per_target{i,1} = data_set_per_target{i,1}(:,:,~mask);
    end
end
%NOTICE The next function ASSUMES that all the bins are of constant
%size - if not valid errors can occur
function data_set_per_target = create_data_set_per_target(data,samples_per_target,M)
    N = size(data{1,1},2);%Numbers of bins to operate on.
    num_of_targets = size(samples_per_target,2);
    data_set_per_target = cell(num_of_targets,1);
    for target = 1:num_of_targets
        data_per_target = zeros(M,N,samples_per_target(target));
        for sample = 1:samples_per_target(target)
            for neuron=1:M
                data_per_target(neuron,:,sample) = data{neuron,target}(sample,:);
            end
        end
        data_set_per_target{target,1} = data_per_target;
    end
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function data = merge_data_from_cell(data_Structs,beep_start, targets, num_of_targets)
    data_set_size = size(data_Structs,2);
    data_devided = cell(1,data_set_size);
    data_neurons_num = 0;
    for i=1:data_set_size
        full_data_per_S = data_Structs{1,i}.sct.fr_hz_per_trial_per_cl_per_target;
        g_title_per_S = data_Structs{1,i}.g_title;
        if(beep_start(i) == 1)
            data_per_S = full_data_per_S( :,sum(g_title_per_S(:,2)==targets,2)==1 );
        else
            data_per_S = full_data_per_S( :,sum(g_title_per_S(:,2)==("beep_" + targets),2)==1 );
        end
        data_neurons_num = data_neurons_num + size(data_per_S,1);
        data_devided{1,i} = data_per_S;
    end
    data = cell(data_neurons_num,num_of_targets);
    offset = 0;
    for i = 1:data_set_size
        if(i==1)
            data(1:size(data_devided{1,i},1),:) = data_devided{1,i};
            offset = size(data_devided{1,i},1);
        else
            data((offset+1):(offset + size(data_devided{1,i},1)),:) = data_devided{1,i};
            offset = offset + size(data_devided{1,i});
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function all_possible_tests_sets = number_of_test_sets(samples_per_target,test_samples_per_target)
    all_possible_tests_sets = cell(size(samples_per_target,2),1);
    for i = 1:size(samples_per_target,2)
        all_possible_tests_sets{i,1} = nchoosek(1:samples_per_target(i),test_samples_per_target(i));
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function indexes_as_test = choose_indexes(num_of_targets,all_possible_tests_sets)
    indexes_as_test =cell(num_of_targets,1);
    for i= 1:num_of_targets
        possibiities = size(all_possible_tests_sets{i,1},1);
        indexes_as_test(i,1) = {all_possible_tests_sets{i,1}(randi(possibiities),:)};
    end
end

%create a logical mask that will return all occurances of any value in
%specifics vector that is in the values vector.
%test : mask = create_mask_from_vec(1:5,[4,2])
function mask = create_mask_from_vec(values,specifics)
    mask =sum(ones(size(specifics,2),1)*values == transpose(specifics),1) > 0;
end