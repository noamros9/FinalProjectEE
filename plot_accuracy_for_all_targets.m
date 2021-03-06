%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%choose chance level and statistic demand for ttest
titles = "Speech targets";
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
results_neurons = load("resultsNeuronsSpeech.mat");
pSpeech = plot_accuracy_graph(results_neurons.results,chance_level,stat_demand,5,2,neurons_asterik_diff,present_error_jumps_neurons,'c','Speech targets Accuracy');
results_neurons = load("resultsNeuronsAuditory.mat");
pAudio = plot_accuracy_graph(results_neurons.results,chance_level,stat_demand,5,2,neurons_asterik_diff,present_error_jumps_neurons,'r','Audiotory targets Accuracy');
results_neurons = load("resultsNeuronsImaginary.mat");
pImagine = plot_accuracy_graph(results_neurons.results,chance_level,stat_demand,5,2,neurons_asterik_diff,present_error_jumps_neurons,[0.4940 0.1840 0.5560],'Imaginary targets Accuracy');
p4 = plot(1:123,ones(1,123)*100*chance_level,'--','Color',[0.8500 0.3250 0.0980],'DisplayName','Chance Level');
legend([pSpeech,pAudio,pImagine,p4]);
legend('boxoff');
legend('Location','southeast');
grid on;
xlabel(neurons_labels);
ylabel("Accuracy [%]");
axis([0,123 + 5,0,100]);
axis 'auto x';
title("Mean accuracy for different targets");
saveas(gcf,"Accurcy as function of neurons number for different targets" + ".jpg");

function plotted_line = plot_accuracy_graph(results,chance_level,stat_demand,start_std,dash_length,asterik_diff,present_error_jumps,color_line,legend_name)
    accuracy = results(:,:,1);
    samples_per_condition = size(results,2) - 1;
    num_of_conditions = size(results,1);
    error_per_condition = zeros(num_of_conditions,1);
    %if all conditions are met then only one sample can be done, so std is
    %only on one value meaning it's zero. therefore calculate only for
    %times when not all condition are met
    %calculate std error - not std!
    error_per_condition((1:(num_of_conditions-1)),1) = sqrt(1/samples_per_condition) * std(100*accuracy(1:(num_of_conditions-1),1:samples_per_condition),0,2);
    mean_accuracy = 100*accuracy(:,samples_per_condition+1);
    plotted_line = plot((1:num_of_conditions)',mean_accuracy,'LineWidth',1.5,'DisplayName',legend_name,'Color',color_line);
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