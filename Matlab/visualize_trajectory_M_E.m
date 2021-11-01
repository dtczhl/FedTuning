% plot trajectories

clear, clc

% file in Result/ProcessedData/
filename = 'fedtuning_True__speech_command__resnet_10__M_20__E_20_00__alpha_0_10__beta_0_00__gamma_0_10__delta_0_80__penalty_1_00__1.csv';

data = readtable(fullfile('../Result/ProcessedData/', filename));
data = data{:, :};

figure(1), clf, hold on
plot(data(:, 1), data(:, 11), 'linewidth', 3)
plot(data(:, 1), data(:, 12), 'linewidth', 3)
legend({'M', 'E'}, 'location', 'east')
xlabel('Training round')
grid on
set(gca, 'fontsize', 20)
hold off
