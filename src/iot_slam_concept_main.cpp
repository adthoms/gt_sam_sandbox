#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/BearingRange.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/SimpleCamera.h>
#include <gtsam/slam/expressions.h>
#include <gtsam/nonlinear/ExpressionFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>

typedef gtsam::BearingRange<gtsam::Pose3, gtsam::Point3> BearingRange3D;

std::vector<gtsam::Point3> createPoints() {
  // Create the set of ground-truth landmarks
  std::vector<gtsam::Point3> points;
  points.push_back(gtsam::Point3(10.0, 10.0, 10.0));
  points.push_back(gtsam::Point3(-10.0, 10.0, 10.0));
  points.push_back(gtsam::Point3(-10.0, -10.0, 10.0));
  points.push_back(gtsam::Point3(10.0, -10.0, 10.0));
  points.push_back(gtsam::Point3(10.0, 10.0, -10.0));
  points.push_back(gtsam::Point3(-10.0, 10.0, -10.0));
  points.push_back(gtsam::Point3(-10.0, -10.0, -10.0));
  points.push_back(gtsam::Point3(10.0, -10.0, -10.0));

  return points;
}

std::vector<gtsam::Pose3> createPoses(const gtsam::Pose3& init,
                                      const gtsam::Pose3& delta, int steps) {
  // Create the set of ground-truth poses
  std::vector<gtsam::Pose3> poses;
  poses.push_back(init);
  for (int i = 1; i < steps; i++) {
    poses.push_back(poses[i - 1].compose(delta));
  }

  return poses;
}

int main(int argc, char* argv[]) {
  // Move around so the whole state (including the sensor tf) is observable
  gtsam::Pose3 init_pose = gtsam::Pose3();
  gtsam::Pose3 delta_pose1 = gtsam::Pose3(
      gtsam::Rot3().Yaw(2 * M_PI / 8).Pitch(M_PI / 8), gtsam::Point3(1, 0, 0));
  gtsam::Pose3 delta_pose2 =
      gtsam::Pose3(gtsam::Rot3().Pitch(-M_PI / 8), gtsam::Point3(1, 0, 0));
  gtsam::Pose3 delta_pose3 =
      gtsam::Pose3(gtsam::Rot3().Yaw(-2 * M_PI / 8), gtsam::Point3(1, 0, 0));

  int steps = 4;
  auto poses = createPoses(init_pose, delta_pose1, steps);
  auto poses2 = createPoses(init_pose, delta_pose2, steps);
  auto poses3 = createPoses(init_pose, delta_pose3, steps);

  // Concatenate poses to create trajectory
  poses.insert(poses.end(), poses2.begin(), poses2.end());
  poses.insert(poses.end(), poses3.begin(),
               poses3.end());    // std::vector of Pose3
  auto points = createPoints();  // std::vector of Point3

  // (ground-truth) sensor pose in body frame, further an unknown variable
  gtsam::Pose3 body_T_sensor_gt(gtsam::Rot3::RzRyRx(-M_PI_2, 0.0, -M_PI_2),
                                gtsam::Point3(0.25, -0.10, 1.0));

  // The graph
  gtsam::ExpressionFactorGraph graph;

  // Specify uncertainty on first pose prior and also for between factor
  // (simplicity reasons)
  auto poseNoise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << 0.3, 0.3, 0.3, 0.1, 0.1, 0.1).finished());

  // Uncertainty bearing range measurement;
  auto bearingRangeNoise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(3) << 0.01, 0.03, 0.05).finished());

  // Expressions for body-frame at key 0 and sensor-tf
  gtsam::Pose3_ x_('x', 0);
  gtsam::Pose3_ body_T_sensor_('T', 0);

  // Add a prior on the body-pose
  graph.addExpressionFactor(x_, poses[0], poseNoise);

  // Simulated measurements from pose
  for (size_t i = 0; i < poses.size(); ++i) {
    auto world_T_sensor = poses[i].compose(body_T_sensor_gt);
    for (size_t j = 0; j < points.size(); ++j) {
      // This expression is the key feature of this example: it creates a
      // differentiable expression of the measurement after being displaced by
      // sensor transform.
      auto prediction_ = gtsam::Expression<BearingRange3D>(
          BearingRange3D::Measure, gtsam::Pose3_('x', i) * body_T_sensor_,
          gtsam::Point3_('l', j));

      // Create a *perfect* measurement
      auto measurement = BearingRange3D(world_T_sensor.bearing(points[j]),
                                        world_T_sensor.range(points[j]));

      // Add factor
      graph.addExpressionFactor(prediction_, measurement, bearingRangeNoise);
    }

    // and add a between factor to the graph
    if (i > 0) {
      // And also we have a *perfect* measurement for the between factor.
      graph.addExpressionFactor(
          between(gtsam::Pose3_('x', i - 1), gtsam::Pose3_('x', i)),
          poses[i - 1].between(poses[i]), poseNoise);
    }
  }

  // Create perturbed initial
  gtsam::Values initial;
  gtsam::Pose3 delta(gtsam::Rot3::Rodrigues(-0.1, 0.2, 0.25),
                     gtsam::Point3(0.05, -0.10, 0.20));
  for (size_t i = 0; i < poses.size(); ++i)
    initial.insert(gtsam::Symbol('x', i), poses[i].compose(delta));
  for (size_t j = 0; j < points.size(); ++j)
    initial.insert<gtsam::Point3>(gtsam::Symbol('l', j),
                                  points[j] + gtsam::Point3(-0.25, 0.20, 0.15));

  // Initialize body_T_sensor wrongly (because we do not know!)
  initial.insert<gtsam::Pose3>(gtsam::Symbol('T', 0), gtsam::Pose3());

  std::cout << "initial error: " << graph.error(initial) << std::endl;
  gtsam::Values result =
      gtsam::LevenbergMarquardtOptimizer(graph, initial).optimize();
  std::cout << "final error: " << graph.error(result) << std::endl;

  initial.at<gtsam::Pose3>(gtsam::Symbol('T', 0))
      .print("\nInitial estimate body_T_sensor\n"); /* initial sensor_P_body
                                                       estimate */
  result.at<gtsam::Pose3>(gtsam::Symbol('T', 0))
      .print("\nFinal estimate body_T_sensor\n"); /* optimized sensor_P_body
                                                     estimate */
  body_T_sensor_gt.print(
      "\nGround truth body_T_sensor\n"); /* sensor_P_body ground truth */

  return 0;
}