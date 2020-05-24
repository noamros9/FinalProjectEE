%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This code is in charge to generate a graph to represent accurcy as a
%sunction of number of neurons
results_neurons = load("results_per_number_neurons");
results_ssesion = load("results_per_ssesion");

ssesion_num = size(results_ssesion.results,1);
colums_num_ssesion = size(results_ssesion.results,2);
M = size(results_neurons.results,1);
colums_num_neurons = size(results_neurons.results,2);

neurons_num_neuron = 1:M;
ssesions_vec = 1:ssesion_num;

mean_accuracy_ssesion = results_ssesion.results(:,colums_num_ssesion,1);
mean_std_accuracy_ssesion = results_ssesion.results(:,colums_num_ssesion,2);
mean_accuracy = transpose(results_neurons.results(:,colums_num_neurons,1));
mean_std_accurcy = transpose(results_neurons.results(:,colums_num_neurons,2));

present_std_jumps = 10;
present_ssesion_std_jump = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PART1: neurons graph %%%%%%%%%%%%%%%%%%%
figure;
plot(neurons_num_neuron,mean_accuracy,'LineWidth',3);
hold on;
for i=1:present_std_jumps:M
    plot_std_line(neurons_num_neuron(i),mean_accuracy(i),mean_std_accurcy(i),2);
    hold on
end
grid on;
xlabel("Neurons number");
ylabel("Accuracy");
title("Accuracy of decoder as function of neurons number in the decoding process");
saveas(gcf,"Accurcy as function of neuron num.jpg");

figure;
plot(neurons_num_neuron,mean_std_accurcy,'LineWidth',3);
grid on;
xlabel("Neurons number");
ylabel("\sigma(standard deviation)");
title("\sigma of decoder as function of neurons number in the decoding process");
saveas(gcf,"std of decoder as function of neuron num.jpg");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PART2 SSESION graphs %%%%%%%%%%%%%%%%%%%
figure;
plot(ssesions_vec,mean_accuracy_ssesion,'LineWidth',3);
hold on;
for i=1:present_ssesion_std_jump:ssesion_num
    plot_std_line(ssesions_vec(i),mean_accuracy_ssesion(i),mean_std_accuracy_ssesion(i),0.2);
    hold on
end
grid on;
xlabel("Sessions number");
ylabel("Accuracy");
title("Accuracy of decoder as function of Sessions number used in the decoding process");
saveas(gcf,"accuracy of decoder as function of sessions num.jpg");


figure;
plot(ssesions_vec,mean_std_accuracy_ssesion,'LineWidth',3);
grid on;
xlabel("Neurons number");
ylabel("\sigma(standard deviation)");
title("\sigma of decoder as function of sessions number in the decoding process");
saveas(gcf,"std of decoder as function of sessions num.jpg");


function plot_std_line(x,y,std_value,dash_length)
    up_thers = y + std_value;
    down_thres = y - std_value;
    plot([x,x],[down_thres,up_thers],'m');
    hold on;
    plot([x - dash_length,x + dash_length],[down_thres,down_thres],'m');
    hold on;
    plot([x - dash_length,x + dash_length],[up_thers,up_thers],'m');
end