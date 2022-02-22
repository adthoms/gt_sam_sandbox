clc, clear, format long, format compact, close all

% load data
ground_truth_landmarks = importdata('../results/ground_truth_points.txt');
ground_truth_poses = importdata('../results/ground_truth_poses.txt');
measured_poses = importdata('../results/measured_poses.txt');
optimized_poses = importdata('../results/optimized_poses.txt');

% get variables
q_gt = ground_truth_poses(:,[8,5:7]);
t_gt = ground_truth_poses(:,2:4);
q_m = measured_poses(:,[8,5:7]);
t_m = measured_poses(:,2:4);
q_opt = optimized_poses(:,[8,5:7]);
t_opt = optimized_poses(:,2:4);

% plot
hold on
plotTransforms(t_gt(1:5:end,:), q_gt(1:5:end,:), 'FrameSize', 2)
plot3(t_gt(:,1),t_gt(:,2), t_gt(:,3),'k-')
plot3(ground_truth_landmarks(:,1),ground_truth_landmarks(:,2),ground_truth_landmarks(:,3),'ko','MarkerFaceColor','k')
set(gca,'TickLabelInterpreter', 'latex')
zlim([0 10])
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('z','Interpreter','latex')
box on
hold off

% Translational Root Mean Squared Error
RMSE_m = sqrt(mean((t_gt - t_m).^2))
RMSE_opt = sqrt(mean((t_gt - t_opt).^2))