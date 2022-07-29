#include "utils.h"

#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/sam/RangeFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

// Time (sec); Sender / Antenna ID Receiver Node ID;  Range (m)
using RangeTriple = boost::tuple<double, size_t, double>;

int main(int argc, char* argv[]) {
  // noise
  GaussianDistribution geometry_dist(0, GEOMETRY_NOISE);
  GaussianDistribution odometry_dist(0, ODOMETRY_NOISE);
  GaussianDistribution wireless_dist(0, WIRELESS_NOISE);
  auto geometry_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(3) << GEOMETRY_NOISE, GEOMETRY_NOISE, GEOMETRY_NOISE)
          .finished());
  auto odometry_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(3) << ODOMETRY_NOISE, ODOMETRY_NOISE, ODOMETRY_NOISE)
          .finished());
  auto wireless_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(3) << WIRELESS_NOISE, WIRELESS_NOISE, WIRELESS_NOISE)
          .finished());
  auto wireless_range_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(1) << WIRELESS_NOISE).finished());

  // ground truth points
  std::vector<gtsam::Point3> ground_truth_joint_points;
  ground_truth_joint_points.push_back(gtsam::Point3(0, 0, 0));

  std::vector<gtsam::Point3> ground_truth_tag_points;
  ground_truth_tag_points.push_back(gtsam::Point3(-0.15, +0.15, -0.15));
  ground_truth_tag_points.push_back(gtsam::Point3(+0.15, +0.15, +0.15));
  ground_truth_tag_points.push_back(gtsam::Point3(+0.15, -0.15, +0.15));
  ground_truth_tag_points.push_back(gtsam::Point3(+0.15, +0.15, -0.15));

  std::vector<gtsam::Point3> ground_truth_anchor_points;
  ground_truth_anchor_points.push_back(gtsam::Point3(0.0, +0.3, -0.3));
  ground_truth_anchor_points.push_back(gtsam::Point3(0.0, +0.3, +0.3));
  ground_truth_anchor_points.push_back(gtsam::Point3(+0.3, 0.0, +0.3));
  ground_truth_anchor_points.push_back(gtsam::Point3(+0.3, 0.0, -0.3));

  // graph
  gtsam::NonlinearFactorGraph graph;
  static gtsam::Symbol j0('j', 0);

  // prior factors
  auto prior_noise =
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.01, 0.01, 0.01));
  for (size_t i = 0; i < ground_truth_anchor_points.size(); ++i) {
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Point3>>(
        gtsam::Symbol('a', i), ground_truth_anchor_points.at(i), prior_noise);
  }

  // relative constraints
  for (size_t i = 0; i < ground_truth_tag_points.size(); ++i) {
    // perturb tag position with respect to joint and anchor
    gtsam::Point3 t_WORLD_TAG_PERTURB_JOINT(ground_truth_tag_points.at(i));
    PerturbPoint(t_WORLD_TAG_PERTURB_JOINT, geometry_dist.GetRandomValue());

    gtsam::Point3 t_WORLD_TAG_PERTURB_ANCHOR(t_WORLD_TAG_PERTURB_JOINT);
    PerturbPoint(t_WORLD_TAG_PERTURB_ANCHOR, wireless_dist.GetRandomValue());

    // add joint to tag geometry constraint
    auto geometry_measurement =
        ground_truth_joint_points.at(0).between(ground_truth_tag_points.at(i));
    graph.emplace_shared<gtsam::BetweenFactor<gtsam::Point3>>(
        j0, gtsam::Symbol('t', i), geometry_measurement, geometry_noise);

    // add anchor to tag distance constraint (assuming 1-to-1 correspondence)
    gtsam::Point3 T_WORLD_ANCHOR(ground_truth_anchor_points.at(i));

    // Add tag to anchor distance constraint
    double range_measurement =
        gtsam::Point3(T_WORLD_ANCHOR - t_WORLD_TAG_PERTURB_ANCHOR).norm();
    graph.emplace_shared<gtsam::RangeFactor<gtsam::Point3, gtsam::Point3>>(
        gtsam::Symbol('a', i), gtsam::Symbol('t', i), range_measurement,
        wireless_range_noise);
  }

  // Print
  graph.print("Factor Graph:\n");

  // Create (deliberately inaccurate) initial estimate
  gtsam::Values initial_estimate;
  gtsam::Point3 t_WORLD_JOINT_PERTURB = ground_truth_tag_points.at(0);
  // PerturbPoint(t_WORLD_JOINT_PERTURB, geometry_dist.GetRandomValue());
  initial_estimate.insert(j0, t_WORLD_JOINT_PERTURB);

  for (size_t i = 0; i < ground_truth_tag_points.size(); ++i) {
    gtsam::Point3 t_WORLD_TAG_PERTURB = ground_truth_tag_points.at(i);
    // PerturbPoint(t_WORLD_TAG_PERTURB, wireless_dist.GetRandomValue());
    initial_estimate.insert(gtsam::Symbol('t', i), t_WORLD_TAG_PERTURB);
  }

  for (size_t i = 0; i < ground_truth_anchor_points.size(); ++i) {
    gtsam::Point3 t_WORLD_ANCHOR_PERTURB = ground_truth_anchor_points.at(i);
    // PerturbPoint(t_WORLD_ANCHOR_PERTURB, wireless_dist.GetRandomValue());
    initial_estimate.insert(gtsam::Symbol('a', i), t_WORLD_ANCHOR_PERTURB);
  }

  // Print
  initial_estimate.print("Initial Estimate:\n");

  gtsam::Values result =
      gtsam::LevenbergMarquardtOptimizer(graph, initial_estimate).optimize();
  result.print("Final Result:\n");

  // print and store results
  std::vector<gtsam::Point3> optimized_joint_points;
  std::vector<gtsam::Point3> optimized_tag_points;
  std::vector<gtsam::Point3> optimized_anchor_points;

  std::cout << "j" << 0 << ": " << result.at<gtsam::Point3>(j0) << std::endl;
  optimized_joint_points.emplace_back(result.at<gtsam::Point3>(j0));

  for (size_t i = 0; i < ground_truth_tag_points.size(); ++i) {
    std::cout << "t" << i << ": "
              << result.at<gtsam::Point3>(gtsam::Symbol('t', i)) << std::endl;
    optimized_tag_points.emplace_back(
        result.at<gtsam::Point3>(gtsam::Symbol('t', i)));
  }

  for (size_t i = 0; i < ground_truth_anchor_points.size(); ++i) {
    std::cout << "a" << i << ": "
              << result.at<gtsam::Point3>(gtsam::Symbol('a', i)) << std::endl;
    optimized_anchor_points.emplace_back(
        result.at<gtsam::Point3>(gtsam::Symbol('a', i)));
  }

  // Define orientation
  gtsam::Point3 x_1 =
      (optimized_tag_points.at(0) + optimized_tag_points.at(1)) / 2;
  gtsam::Point3 y_1 =
      (optimized_tag_points.at(2) + optimized_tag_points.at(3)) / 2;
  gtsam::Point3 x_axis = x_1 - optimized_joint_points.at(0);
  gtsam::Point3 y_axis = y_1 - optimized_joint_points.at(0);
  gtsam::Point3 z_axis = x_axis.cross(y_axis);

  gtsam::Point3 x_global(1, 0, 0);
  gtsam::Point3 y_global(0, 1, 0);
  gtsam::Point3 z_global(0, 0, 1);
  gtsam::Matrix33 AXIS_GLOBAL;
  AXIS_GLOBAL << x_axis, y_axis, z_axis;

  // ensure z axis generally points upwards
  double cos_theta = z_global.dot(z_axis) / (z_global.norm() * z_axis.norm());
  if (cos_theta < 0) {
    z_axis *= -1;
  }

  x_axis = x_axis.normalized();
  y_axis = y_axis.normalized();
  z_axis = z_axis.normalized();

  gtsam::Matrix33 AXIS_NODE;
  AXIS_NODE << x_axis, y_axis, z_axis;

  std::cout << "x: " << x_axis << std::endl;
  std::cout << "y: " << y_axis << std::endl;
  std::cout << "z: " << z_axis << std::endl;
  std::cout << "AXIS_NODE: " << AXIS_NODE << std::endl;
  std::cout << "AXIS_GLOBAL: " << AXIS_GLOBAL << std::endl;

  std::cout << "R: " << AXIS_NODE * AXIS_GLOBAL.transpose() << std::endl;

  return 0;
}
