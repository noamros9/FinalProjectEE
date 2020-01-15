%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Defining The parameters in the problem.
actions = 16;
M=10;
N=40;
max_num_of_targets = 6;
num_of_targets_for_estimate_params = 4;
%target is a value between 1 to 6 in this session - namely the number of
%experiments that were taken for each action. in order to have a test on
%the estimator.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%creating a new samples matrix from the Data_set from a target that wasn't
%been used in the dataset of the ML estimation
new_samples = create_samples_from_DataCell(sct.fr_hz_per_trial_per_cl_per_target,3,6,M,N);
%Activating the estimation:
choice = ML_G_IID(new_samples,sct.fr_hz_per_trial_per_cl_per_target,M,N,num_of_targets_for_estimate_params,g_title,actions);
%The function disp desplays to the command window the choice
disp(choice)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%The function ML_G_IID calculate the ML estimation from the given samples.
%Inputs:
%new_samples - a matrix contains the samples in each neurons
%data_cell - contains the data in a cell type
%M,N - the metrix samples,
%num_of_targets_for_estimate_params - the number of targets to use in
%estimations
%g_title a cell containing a mapping from actions to indexes and vice-verca
%number of actions/trials in this case is 16
%output: choice - a string according to g_title of the descion according
%the estimator.
function choice = ML_G_IID(new_samples,data_cell,M,N,num_of_targets_for_estimate_params,g_title,actions)
    [variance_in_each_sample,expec_in_each_sample] = Create_var_and_expec_mats(data_cell,M,N,num_of_targets_for_estimate_params,actions);
    decision_values = zeros(1,actions);
    for alpha = 1:actions
        diff_from_expec = (new_samples - expec_in_each_sample(:,:,alpha)).^2;
        decision_values(alpha) = sum(variance_in_each_sample(:,:,alpha),'all')...
            + sum(diff_from_expec./(2*variance_in_each_sample(:,:,alpha)),'all');
    end
    [min_val,decision] = min(decision_values);
    choice = g_title{decision,2};
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function gets a data cell and an action, and a target and ourputs the
%sample matrix gotten for that given action and trial in the data cell
function samples_mat = create_samples_from_DataCell(data_cell,action,target,M,N)
    samples_mat = zeros(M,N);
    for i=1:M
        samples_mat(i,:) = data_cell{i,action}(target,:);
    end
end
function [variance_in_each_sample,expec_in_each_sample] = Create_var_and_expec_mats(data_cell,M,N,num_of_targets_for_estimate_params,actions)
    variance_in_each_sample = zeros(M,N,actions);
    expec_in_each_sample = zeros(M,N,actions);
    action_DataSet = zeros(M,N,num_of_targets_for_estimate_params);
    for alpha = 1:actions
        for target = 1:num_of_targets_for_estimate_params
            action_DataSet(:,:,target) = create_samples_from_DataCell(data_cell,alpha,target,M,N);
        end
        variance_in_each_sample(:,:,alpha) = estimate_mat_var(action_DataSet);
        expec_in_each_sample(:,:,alpha) = estimate_mat_expec(action_DataSet);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function estimate the expections of the entries of the samples matrix
%for all the actions
%inputs - action_DataSet - an M X N X (number of targets for that action)
% each sub matrix is a samples matrix that was gotten from the expirement
function expec = estimate_mat_expec(action_DataSet)
    expec = sum(action_DataSet,3);
    action_DataSet_size = size(action_DataSet);
    action_DataSet_size = action_DataSet_size(3);
    expec = (1/action_DataSet_size) * expec;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function estimate the variance of the entries of the samples matrix
%for all the actions
% inputs - action_DataSet - an M X N X (number of targets for that action)
% each sub matrix is a samples matrix that was gotten from the expirement
function var_mat = estimate_mat_var(action_DataSet)
    action_DataSet_size = size(action_DataSet);
    action_DataSet_size = action_DataSet_size(3);
    var_mat = (1/action_DataSet_size)*sum((action_DataSet - estimate_mat_expec(action_DataSet)).^2,3);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
