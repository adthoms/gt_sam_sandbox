#include "utils.h"

#include <gtsam/geometry/SimpleCamera.h>
#include <gtsam/slam/expressions.h>
#include <gtsam/nonlinear/ExpressionFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>

int main(int argc, char* argv[]) {
  // Read in joint, tag, and anchor positions
  LoadFromTXT(JOINTS_POSITION_FILE, joint_position_map);
  LoadFromTXT(TAGS_POSITION_FILE, tag_position_map);
  LoadFromTXT(ANCHORS_POSITION_FILE, anchor_position_map);

  // Read in floor, joint, tag, and anchor associations
  LoadFromTXT(FLOOR_ANCHOR_ASSOCIATION_FILE, floor_anchor_association_map);
  LoadFromTXT(JOINT_TAG_ASSOCIATION_FILE, joint_tag_association_map);
  LoadFromTXT(TAG_ANCHOR_ASSOCIATION_FILE, tag_anchor_association_map);

  // Noise
  GaussianDistribution odometry_dist(0, ODOMETRY_NOISE);
  GaussianDistribution wireless_dist(0, WIRELESS_NOISE);

  // generate ground truth and measured robot positions
  gtsam::Pose3 init_pose =
      gtsam::Pose3(gtsam::Rot3::Ypr(0, 0, 0), gtsam::Point3(10, 0, 0));
  gtsam::Pose3 delta_pose =
      gtsam::Pose3(gtsam::Rot3::Ypr(-M_PI / 220, 0, 0),
                   gtsam::Point3(sin(M_PI / 22), sin(M_PI / 22), 0));
  auto ground_truth_robot_poses =
      GeneratePoses(init_pose, delta_pose, ODOMETRY_STEPS);
  auto measured_robot_poses =
      GeneratePoses(init_pose, delta_pose, ODOMETRY_STEPS, ODOMETRY_NOISE);

  // output to txt files
  PrintPoses(ground_truth_robot_poses,
             full_file_path + "ground_truth_robot_poses.txt");
  PrintPoses(measured_robot_poses, full_file_path + "measured_robot_poses.txt");

  // Instantiate graph
  gtsam::ExpressionFactorGraph graph;

  // Add prior factor to first robot position
  graph.addExpressionFactor(gtsam::Point3_('x', 0),
                            ground_truth_robot_poses.at(0).translation(),
                            prior_noise);

  // Simulate measurements from robot to anchors and odometry constraints
  for (size_t i = 0; i < ground_truth_robot_poses.size(); ++i) {
    const gtsam::Point3& t_WORLD_ROBOT =
        ground_truth_robot_poses.at(i).translation();

    for (const auto& [anchor_id, anchor_position] : anchor_position_map) {
      // perturb anchor point for noisy robot to anchor measurement
      gtsam::Point3 t_WORLD_ANCHOR_PERTURB(anchor_position);
      PerturbPoint(t_WORLD_ANCHOR_PERTURB, wireless_dist.GetRandomValue());

      // add robot to anchor constraint
      graph.addExpressionFactor(
          between(gtsam::Point3_('x', i), gtsam::Point3_('a', anchor_id)),
          t_WORLD_ROBOT.between(t_WORLD_ANCHOR_PERTURB), wireless_noise);
    }

    if (i > 0) {
      // add odometry constraints
      graph.addExpressionFactor(
          between(gtsam::Point3_('x', i - 1), gtsam::Point3_('x', i)),
          measured_robot_poses.at(i - 1).translation().between(
              measured_robot_poses.at(i).translation()),
          odometry_noise);
    }
  }

  // provide initial estimates using prior information
  gtsam::Values initial;
  for (size_t i = 0; i < measured_robot_poses.size(); ++i) {
    initial.insert<gtsam::Point3>(gtsam::Symbol('x', i),
                                  measured_robot_poses.at(i).translation());
  }

  for (const auto& [anchor_id, anchor_position] : anchor_position_map) {
    gtsam::Point3 t_WORLD_ANCHOR_PERTURB(anchor_position);
    PerturbPoint(t_WORLD_ANCHOR_PERTURB, wireless_dist.GetRandomValue());
    initial.insert<gtsam::Point3>(gtsam::Symbol('a', anchor_id),
                                  t_WORLD_ANCHOR_PERTURB);
  }

  // for (const auto& [tag_id, tag_position] : tag_position_map) {
  //   initial.insert<gtsam::Point3>(gtsam::Symbol('t', tag_id), tag_position);
  // }

  // optimize graph
  gtsam::Values result =
      gtsam::LevenbergMarquardtOptimizer(graph, initial).optimize();

  // print results
  std::vector<gtsam::Point3> optimized_robot_points;
  for (size_t i = 0; i < ground_truth_robot_poses.size(); ++i) {
    optimized_robot_points.push_back(
        result.at<gtsam::Point3>(gtsam::Symbol('x', i)));
  }
  PrintPoints(optimized_robot_points,
              full_file_path + "optimized_robot_points.txt");

  NodePositionMap optimized_anchor_position_map;
  for (const auto& [anchor_id, anchor_position] : anchor_position_map) {
    optimized_anchor_position_map[anchor_id] =
        result.at<gtsam::Point3>(gtsam::Symbol('a', anchor_id));
  }
  PrintMap(optimized_anchor_position_map,
           full_file_path + "optimized_anchor_position_map.txt");

  // NodePositionMap optimized_tag_position_map;
  // for (const auto& [tag_id, tag_position] : tag_position_map) {
  //   optimized_anchor_position_map[tag_id] =
  //       result.at<gtsam::Point3>(gtsam::Symbol('t', tag_id));
  // }
  // PrintMap(optimized_tag_position_map,
  //          full_file_path + "optimized_tag_position_map.txt");

  return 0;
}
