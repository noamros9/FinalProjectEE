%This code porpuse is to check the validity of the code we created in the 
%decoder2_SVM file. This code will generate for any neuron a print
%presenting it's Active points and baseline points in all sessions.
%This code will also present information regarding the mean and std of each
%mean and trial.

% code for generating:(This code is taken from dacoder2_SVM)
%     - The data we work on
% and the parameters:
%     - targets, number of neurons, and some more...
% especially:
%     - test_sample_per_target
%     - samples_per_target
%     - test_samples_pre_target
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%IMPORTANT, FILL IN HERE WHICH NEURON TO CHECK:
neuron_to_check = [1,2];



%use all samples to test the functions, 
for neuron = neuron_to_check
    figure;
    title_string = sprintf("Plot of samples for the %d'th neuron in the data set, 2\\sigma",neuron);
    suptitle(title_string);
    %each neuron will have a figure, containing num_of_targets rows(for
    %each target),each row will have two plots, one showing the results of.
    %the baseline_vector function and the other the find_max_bins function.
    %detailed information on formating will be below.
    num_of_subplots = num_of_targets*2;
    for target_idx=1:num_of_targets
        num_of_trails_per_target = samples_per_target(target_idx);
        trials_idx = 1:num_of_trails_per_target;
        subplot(num_of_targets,2,2*target_idx-1);
        %start subplot
        baselines_per_target = baseline_vector(data, neuron, target_idx, M,samples_per_target, trials_idx);
        mean_baselines = mean(baselines_per_target);
        std_baselines = std(baselines_per_target); %NOTICE IF ONLY ONE SAMPLE IS GIVEN STD CAN RETURN BAD STUFF
        %plot the information:
        plot([min(trials_idx),max(trials_idx)],[mean_baselines,mean_baselines],'-','Color','b');
        hold on;
        avg_idx = (min(trials_idx) + max(trials_idx))/2;
        plot([avg_idx,avg_idx],[mean_baselines - 2*std_baselines,mean_baselines + 2*std_baselines],...
            '-+b','MarkerFaceColor','b','MarkerSize',12);
        hold on;
        plot([avg_idx,avg_idx],[mean_baselines + 2*std_baselines,mean_baselines + 2.25*std_baselines],...
            "LineStyle","none")
        hold on;
        plot([avg_idx,avg_idx],[mean_baselines - 2*std_baselines,mean_baselines - 2.25*std_baselines],...
            "LineStyle","none")
        hold on;
        scatter(trials_idx,baselines_per_target,18,'b','filled');
        grid on;
        first_plot_title = sprintf("%s - baseline samples",targets(target_idx));
        xlabel("tiral number")
        ylabel("fr per hz");
        title(first_plot_title);
        %end subplot
        
        subplot(num_of_targets,2,2*target_idx);
        %start subplot
        
        %every thing here is per target
        max_bins = find_max_bins(data, neuron, target_idx, M,samples_per_target, trials_idx);
        max_bins_vals = max_bins(:,1);
        max_bins_idxs = max_bins(:,4);
        max_bins_info = [max_bins_idxs,max_bins_vals];
        mean_max_bins = mean(max_bins_vals);
        std_max_bins = std(max_bins_vals);
        [unique_info,ia,ic] = unique(max_bins_info,'rows');
        counts_of_repititions_for_color = accumarray(ic,1);%determine the color
        plot([min(unique_info(:,1)),max(unique_info(:,1))],[mean_max_bins,mean_max_bins],'-','Color','r');
        hold on;
        average_idx = (min(unique_info(:,1)) + max(unique_info(:,1)))/2;
        plot([average_idx,average_idx],[mean_max_bins-2*std_max_bins,mean_max_bins+2*std_max_bins],...
            '-+r','MarkerFaceColor','r','MarkerSize',12);
        hold on;
        plot([average_idx,average_idx],[mean_max_bins + 2*std_max_bins,mean_max_bins + 2.25*std_max_bins],...
            "LineStyle","none")
        hold on;
        plot([average_idx,average_idx],[mean_max_bins - 2*std_max_bins,mean_max_bins - 2.25*std_max_bins],...
            "LineStyle","none")
        hold on;
        scatter(unique_info(:,1),unique_info(:,2),18,counts_of_repititions_for_color,'filled');
        colormap(autumn(samples_per_target(target_idx)));
        if(size(unique_info,1) ~= size(max_bins_info,1))%there are points that repeat in their values;
            colorbar;
        end
        xlabel("bin number");
        ylabel("fr per hz");
        grid on;
        %end subplot
        second_plot_title = sprintf("%s - active samples",targets(target_idx));
        title(second_plot_title);
    end
end




%Upload Test function:

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
%this function return a baseline vector per neuron per trials
%the baseline is defined as the average of the [-0.8,-0.3][s]

function baseline=baseline_vector(data, i, j, M,training_samples_per_target, train_idx)
	
	%we take 80 percent of the trials as training set
	%max_trial = ceil(length(data{i,j}(:,1))*training_precent_from_data);
	max_trial = training_samples_per_target(j);
	%create the baseline vector
	baseline = zeros(max_trial,1);

	for k = 1:max_trial
		baseline(k) = mean(data{i,j}(train_idx(k), 13:17));
	end
	
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%returns a matrix of size (max_trial,3). For every channel and vowel, 
%each row (trial) will be: (max_bin, (max_bin+predecessor)/2, (max_bin+follower)/2)


function max_bins=find_max_bins(data, i, j, M,training_samples_per_target, train_idx)
	
	%we take 80 percent of the trials as training set 
	%max_trial = ceil(length(data{i,j}(:,1))*training_precent_from_data);
	max_trial = training_samples_per_target(j);
	max_bins = zeros(max_trial, 4);
	
	for k = 1:max_trial
		[max_bin_val, max_bin_idx] = max(data{i,j}(train_idx(k), 21:30));
		max_bins(k, 1) = max_bin_val;
        %notice that the max_bin_idx is relative to the specified array, so
        %we add 20 as offset
		max_bins(k, 2) = (max_bin_val + data{i,j}(train_idx(k), max_bin_idx + 21))/2;
		max_bins(k, 3) = (max_bin_val + data{i,j}(train_idx(k), max_bin_idx + 19))/2;
		max_bins(k, 4) = max_bin_idx;
	end
	
end