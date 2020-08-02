%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This code is in charge to generate a graph to represent accurcy as a
%function of number of neurons used in decoding.
%The code does contain some assumption on our data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Changable values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

results_neurons = load("resultsNeuronsSpeech.mat");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%choose chance level and statistic demand for ttest
titles = "Speech Targets";



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



chance_level = 0.2;%NOT IN PERCENT the plot_accurcy_graph will convert it to percent
stat_demand = 0.05;%NOT IN PERCENT the plot_accurcy_graph will assume this value is not in percent
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%present error std in these jumps:
present_error_jumps_neurons = 10;
present_ssesion_std_jump = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%present asterik(*) in these locations if needed:
neurons_asterik_diff = [-2,1];
session_asterik_diff = [-0.1,1];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
neurons_labels = "Neurons number";
plot_accuracy_graph(results_neurons.results,chance_level,stat_demand,5,2,neurons_asterik_diff,present_error_jumps_neurons,neurons_labels," - " + titles)


function plot_accuracy_graph(results,chance_level,stat_demand,start_std,dash_length,asterik_diff,present_error_jumps,label,titles)
    accuracy = results(:,:,1);
    samples_per_condition = size(results,2) - 1;
    num_of_conditions = size(results,1);
    error_per_condition = zeros(num_of_conditions,1);
    %if all conditions are met then only one sample can be done, so std is
    %only on one value meaning it's zero. therefore calculate only for
    %times when not all condition are met
    %calculate std error - not std!
    error_per_condition((1:(num_of_conditions-1)),1) = sqrt(1/samples_per_condition) * std(100*accuracy(1:(num_of_conditions-1),1:samples_per_condition),0,2);
    figure;
    mean_accuracy = 100*accuracy(:,samples_per_condition+1);
    p1 = plot((1:num_of_conditions)',mean_accuracy,'LineWidth',1.5,'DisplayName','Accurcy');
    hold on;
    for i=start_std:present_error_jumps:num_of_conditions
        if (i==num_of_conditions)
            samples_occurances = 1;
        else
            samples_occurances = samples_per_condition;
        end
        plot_std_error_line(i,mean_accuracy(i),error_per_condition(i),dash_length);
        hold on;
        Isdiffer = check_ttest1_for_mean(accuracy(i,1:samples_occurances),chance_level,stat_demand);
        if(Isdiffer)%if the accuracy array is significantly bigger than chance level mark with asterik
            scatter(i + asterik_diff(1),mean_accuracy(i)+error_per_condition(i)+asterik_diff(2),'r*');
            hold on
        end
    end
    p2 = plot(1:num_of_conditions,ones(1,num_of_conditions)*100*chance_level,'--','Color',[0.8500 0.3250 0.0980],'DisplayName','Chance Level');
    legend([p1,p2]);
    legend('boxoff');
    legend('Location','southeast');
    grid on;
    xlabel(label);
    ylabel("Accuracy [%]");
    axis([0,num_of_conditions + 5,0,100]);
    axis 'auto x';
    title("Accuracy as func. of " + label + titles);
    saveas(gcf,"Accurcy as function of " + label + titles + ".jpg");

    figure;
    plot((1:num_of_conditions)',error_per_condition,'LineWidth',1.5);
    grid on;
    xlabel(label);
    ylabel("standard error [%]");
    legend("standard error");
    title("Std error of Accuracy as func. of " + label + titles)
    saveas(gcf,"std error of decoder as function of " + label+ titles +".jpg");
end

function plot_std_error_line(x,y,error_value,dash_length)
    up_thers = y + error_value;
    down_thres = y - error_value;
    plot([x,x],[down_thres,up_thers],'m');
    hold on;
    plot([x - dash_length,x + dash_length],[down_thres,down_thres],'m');
    hold on;
    plot([x - dash_length,x + dash_length],[up_thers,up_thers],'m');
end
%This function gets an array with a specified samples of accuracy of
%decoders and return 1 if it's mean is statisically large than chance_level
%specified. meaning it is unlikely to be distributed under mean of chance level.
% the p-value has to be lower than stat_demand
function Isdiffer = check_ttest1_for_mean(sample_means,chance_level,stat_demand)
    [h,p] = ttest(sample_means,chance_level,'Tail', 'right');
    Isdiffer = p<stat_demand;
end