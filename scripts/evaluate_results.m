clc, clear, format long, format compact, close all

% load data
ground_truth_robot_poses = importdata('../results/ground_truth_robot_poses.txt');
ground_truth_anchor_points = importdata('../results/ground_truth_anchor_points.txt');
ground_truth_tag_points = importdata('../results/ground_truth_tag_points.txt');
measured_robot_poses = importdata('../results/measured_robot_poses.txt');
optimized_robot_points = importdata('../results/optimized_robot_points.txt');
optimized_anchor_points = importdata('../results/optimized_anchor_points.txt');
optimized_tag_points = importdata('../results/optimized_tag_points.txt');

% get variables
ground_truth_robot_rotation = ground_truth_robot_poses(:,[8,5:7]);
ground_truth_robot_position = ground_truth_robot_poses(:,2:4);
measured_robot_position = measured_robot_poses(:,2:4);

% plot
hold on
plotTransforms(ground_truth_robot_position(1:5:end,:), ground_truth_robot_rotation(1:5:end,:), 'FrameSize', 2)
plot3(ground_truth_robot_position(:,1),ground_truth_robot_position(:,2), ground_truth_robot_position(:,3),'k.-')
plot3(measured_robot_position(:,1),measured_robot_position(:,2), measured_robot_position(:,3),'r.-')
plot3(ground_truth_anchor_points(:,1),ground_truth_anchor_points(:,2),ground_truth_anchor_points(:,3),'ko','MarkerFaceColor','k')
plot3(ground_truth_tag_points(:,1),ground_truth_tag_points(:,2),ground_truth_tag_points(:,3),'ko','MarkerFaceColor','m')

set(gca,'TickLabelInterpreter', 'latex')
zlim([0 10])
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('z','Interpreter','latex')
box on
hold off

% Translational Root Mean Squared Error
RMSE_robot_measured = sqrt(sum(mean((ground_truth_robot_position - measured_robot_position).^2)))
RMSE_robot_optimized = sqrt(sum(mean((ground_truth_robot_position - optimized_robot_points).^2)))

RMSE_anchor_optimized = sqrt(sum(mean((ground_truth_anchor_points - optimized_anchor_points).^2)))
RMSE_tag_optimized = sqrt(sum(mean((ground_truth_tag_points - optimized_tag_points).^2)))