%In this code we will try to create a much more sophisticated decoder that
%will take into accoubt general structure in the data and wil use
%information from different sessions.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%uploding the necessary data
%creating a struct with all the information of the two ssestions files
S1 = load('speech_screening_analysis_beep_session1.mat');
S2 = load('speech_screening_analysis_beep_session2.mat');
%defining the title in these two cases
g_title1 = S1.g_title;
g_title2 = S2.g_title;
%getting the full data.
full_data1 = S1.sct.fr_hz_per_trial_per_cl_per_target;
full_data2 = S2.sct.fr_hz_per_trial_per_cl_per_target;
%define all interpretations we will take care in this decoder:
targets = ["a","e","i","o","u"];
num_of_target = size(targets,2);
% taking the data for the interpatations chosen:
data1 = full_data1( :,sum(g_title1(:,2)==targets,2)==1 );
data2 = full_data2( :,sum(g_title2(:,2)==targets,2)==1 );
data = cell(size(data1,1)+size(data2,1),num_of_target);
data(1:size(data1,1),:) = data1;
offset = size(data1,1);
data((offset+1):offset + size(data2,1),:) = data2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





