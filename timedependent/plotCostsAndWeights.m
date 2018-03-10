clear; clc; clf;

addpath('results/');

% -------------------------------- FILE NAME EXTRACTION --------------------------------

% number of scatter points
scatterPoints = input('Number of Scatter Points: ', 's');
if length(scatterPoints) < 2
    disp('Number of scatter points must be be formatted using 2 or more digits (i.e. 02 for 2)');
    return
end

% number of time units
timeUnits = input('Number of Time Units: ', 's');
if length(scatterPoints) < 2
    disp('Number of time units must be be formatted using 2 or more digits (i.e. 02 for 2)');
    return
end

% concatenate string for filename using scatterPoints and timeUnits
fileName = strcat('LossesAndWeights_', scatterPoints, '_T', timeUnits, '_all.csv')




% -------------------------------- DATA READ --------------------------------

% read in Unmasked Network's CSV data (epoch #, loss, W1, W2, W3, ... , Wn)
data = csvread(fileName,1,1);

epochs = data(:,1); % extract epoch column

losses = data(:,2); % extract loss column

weightMatrix = data(:,3:end);   % the rest of the data is weights per epoch


% -------------------------------- PLOT TRAINING LOSS PER EPOCH --------------------------------
lossFig = figure(1);

% plotting range for loss
lossrange = 1:length(epochs);

% regular scale (loss vs. epoch)
subplot(3,1,1);
plot(epochs(lossrange), losses(lossrange), 'DisplayName', 'masked');
ylabel('normal scale');
legend('show');
grid on;

% y-log scale (loss-log vs. epoch)
subplot(3,1,2);
semilogy(epochs(lossrange), losses(lossrange), 'DisplayName', 'masked');
ylabel('log-y scale');
legend('show');
grid on;

% log-log scale (loss-log vs. epoch-log)
subplot(3,1,3);
loglog(epochs(lossrange), losses(lossrange), 'DisplayName', 'masked');
ylabel('log-log scale');
grid on;
legend('show');

xlabel('epoch');
suptitle('Training Loss');



% -------------------------------- PLOT WEIGHT CHANGE THROUGH TRAINING --------------------------------
weightFig = figure(2);

startEpoch = 9; % first epoch of weights to show during training
step = 10;      % epoch step size
endEpoch = 100; % final epoch of weights to show

epochWeightRange = startEpoch:step:endEpoch;   % get range of row indices for weights to plot

[m,n] = size(weightMatrix);

% plot the weights after epoch 1
plot(1:n, weightMatrix(1,:));
% plot at subsequent epochs until N
hold on;
for k = epochWeightRange
    plot(1:n, weightMatrix(k,:));
end
hold off;
title(strcat('Weights from Epochs ', ' ', num2str(epochs(epochWeightRange(1))), '-', num2str(epochs(epochWeightRange(end))) ))

saveas(lossFig, strcat('results/Training_Loss_' , scatterPoints, '_T', timeUnits,'_all.pdf'));
saveas(weightFig, strcat('results/Training_Weights_' , scatterPoints, '_T', timeUnits,'_all.pdf'));
