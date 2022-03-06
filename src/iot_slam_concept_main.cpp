#include <fstream>
#include <random>

#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/BearingRange.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/SimpleCamera.h>
#include <gtsam/slam/expressions.h>
#include <gtsam/nonlinear/ExpressionFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <ros/package.h>

typedef gtsam::BearingRange<gtsam::Pose3, gtsam::Point3> BearingRange3D;

double randf(double ub, double lb) {
  double f = (double)rand() / RAND_MAX;
  return lb + f * (ub - lb);
}

void PerturbTranslation(double total_error, gtsam::Pose3& delta) {
  // divide total error among xyz axes
  total_error = std::abs(total_error);
  const double x_error = round(randf(-1, 1)) * randf(total_error, 0);
  const double y2_ub = pow(total_error, 2) - pow(x_error, 2);
  const double y_error = round(randf(-1, 1)) * randf(sqrt(y2_ub), 0);
  const double z2_ub = y2_ub + pow(y_error, 2);
  const double z_error = round(randf(-1, 1)) * sqrt(z2_ub);

  // perturb delta (assume noise is restricted to translation)
  gtsam::Pose3 delta_perturbed =
      gtsam::Pose3(gtsam::Rot3::Rodrigues(0, 0, 0),
                   gtsam::Point3(x_error, y_error, z_error));
  delta = delta.compose(delta_perturbed);
}

std::vector<gtsam::Point3> PerturbPoints(
    const std::vector<gtsam::Point3>& points, double sigma_noise = 0) {
  std::vector<gtsam::Point3> points_perturbed = points;
  if (sigma_noise > 0) {
    // initialize random noise generation
    std::normal_distribution<> gauss_noise_dist{0, sigma_noise};
    std::random_device rd{};
    std::mt19937 gen{rd()};

    gtsam::Pose3 delta_perturbed = gtsam::Pose3();
    for (auto iter = points_perturbed.begin(); iter != points_perturbed.end();
         ++iter) {
      PerturbTranslation(std::abs(gauss_noise_dist(gen)), delta_perturbed);
      *iter += delta_perturbed.translation();
    }
  }
  return points_perturbed;
}

std::vector<gtsam::Pose3> CreatePoses(const gtsam::Pose3& init,
                                      const gtsam::Pose3& delta, int steps,
                                      double sigma_noise = 0) {
  // initialize random noise generation
  std::normal_distribution<> gauss_noise_dist{0, sigma_noise};
  std::random_device rd{};
  std::mt19937 gen{rd()};

  // populate poses
  std::vector<gtsam::Pose3> poses;
  poses.push_back(init);
  for (int i = 1; i < steps; ++i) {
    // perturb
    gtsam::Pose3 delta_perturbed = delta;
    if (sigma_noise > 0) {
      PerturbTranslation(std::abs(gauss_noise_dist(gen)), delta_perturbed);
    }

    // propogate
    poses.push_back(poses[i - 1].compose(delta_perturbed));
  }

  return poses;
}

void PrintPoints(const std::vector<gtsam::Point3>& points,
                 const std::string& output_dir) {
  std::ofstream file(output_dir);
  for (gtsam::Point3 n : points) {
    file << n.x() << " " << n.y() << " " << n.z() << std::endl;
  }
}

void PrintPoses(const std::vector<gtsam::Pose3>& poses,
                const std::string& output_dir) {
  std::ofstream file(output_dir);
  int t = 0;
  for (gtsam::Pose3 n : poses) {
    const Eigen::Matrix3d R = n.matrix().block<3, 3>(0, 0);
    const Eigen::Quaterniond q(R);
    file << t << " " << n.x() << " " << n.y() << " " << n.z() << " " << q.x()
         << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    t++;
  }
}

int main(int argc, char* argv[]) {
  // seed
  srand(time(NULL));

  // noise parameters
  const double ODOMETRY_NOISE = 0.01;  // +/- 5cm relative accuracy from LVIO
  const double UWB_NOISE = 0.05;  // +/- 5cm accuracy from Humatics UWB sensors

  // UWB anchors
  std::vector<gtsam::Point3> ground_truth_points;
  ground_truth_points.push_back(gtsam::Point3(0.0, -30.0, 10.0));
  //ground_truth_points.push_back(gtsam::Point3(0, 0, 10.0));
  //ground_truth_points.push_back(gtsam::Point3(0.0, -10.0, 10.0));
  ground_truth_points.push_back(gtsam::Point3(0.0, 10.0, 10.0));
  ground_truth_points.push_back(gtsam::Point3(40.0, -30.0, 10.0));
  //ground_truth_points.push_back(gtsam::Point3(40.0, -10.0, 10.0));
  ground_truth_points.push_back(gtsam::Point3(40.0, 10.0, 10.0));
  //ground_truth_points.push_back(gtsam::Point3(40.0, 0, 10.0));

  // noise
  // og code line:gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished()
  auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << 2e-6, 2e-6, 2e-6, 2e-6, 2e-6, 2e-6).finished());
  auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << ODOMETRY_NOISE, ODOMETRY_NOISE, ODOMETRY_NOISE,
       ODOMETRY_NOISE, ODOMETRY_NOISE, ODOMETRY_NOISE)
          .finished());
  auto bearing_range_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(3) << UWB_NOISE, UWB_NOISE, UWB_NOISE).finished());

  // generate ground truth and measured poses/points
  gtsam::Pose3 init_pose =
      gtsam::Pose3(gtsam::Rot3::Ypr(0, 0, 0), gtsam::Point3(10, 0, 0));
  gtsam::Pose3 delta_pose =
      gtsam::Pose3(gtsam::Rot3::Ypr(-M_PI / 220, 0, 0),
                   gtsam::Point3(sin(M_PI / 22), sin(M_PI / 22), 0));
  auto ground_truth_poses = CreatePoses(init_pose, delta_pose, 440);
  auto measured_poses = CreatePoses(init_pose, delta_pose, 440, ODOMETRY_NOISE);

  // output to txt files
  std::string full_file_path =
      ros::package::getPath("gt_sam_sandbox") + "/results/";
  PrintPoses(ground_truth_poses, full_file_path + "ground_truth_poses.txt");
  PrintPoints(ground_truth_points, full_file_path + "ground_truth_points.txt");
  PrintPoses(measured_poses, full_file_path + "measured_poses.txt");

  // Instantiate graph
  gtsam::ExpressionFactorGraph graph;

  // Expressions for body-frame at key 0. All sensors are within the body-frame
  gtsam::Pose3_ x_('x', 0);
  graph.addExpressionFactor(x_, ground_truth_poses[0], prior_noise);

  // Simulated measurements from pose
  for (size_t i = 0; i < measured_poses.size(); ++i) {
    // sensor pose expressed in world frame
    const gtsam::Pose3 T_WORLD_BODY = ground_truth_poses[i];

    // generate noisy set of points as measured from robot
    auto measured_points = PerturbPoints(ground_truth_points, UWB_NOISE);
    for (size_t j = 0; j < measured_points.size(); ++j) {
      // Create differentiable expression of the measurement from the body-frame
      gtsam::Expression<BearingRange3D> prediction_ =
          gtsam::Expression<BearingRange3D>(BearingRange3D::Measure,
                                            gtsam::Pose3_('x', i),
                                            gtsam::Point3_('l', j));

      // Create bearing and range measurement from robot pose to iot landmark
      BearingRange3D measurement =
          BearingRange3D(T_WORLD_BODY.bearing(measured_points[j]),
                         T_WORLD_BODY.range(measured_points[j]));
      graph.addExpressionFactor(prediction_, measurement, bearing_range_noise);
    }

    // add between factor to the graph
    if (i > 0) {
      // Create odometry measurement
      graph.addExpressionFactor(
          between(gtsam::Pose3_('x', i - 1), gtsam::Pose3_('x', i)),
          measured_poses[i - 1].between(measured_poses[i]), pose_noise);
    }
  }

  // provide initial estimates
  gtsam::Values initial;
  for (size_t i = 0; i < ground_truth_poses.size(); ++i)
    initial.insert<gtsam::Pose3>(gtsam::Symbol('x', i), ground_truth_poses[i]);
  for (size_t j = 0; j < ground_truth_points.size(); ++j)
    initial.insert<gtsam::Point3>(gtsam::Symbol('l', j),
                                  ground_truth_points[j]);

  // optimize graph
  gtsam::Values result =
      gtsam::LevenbergMarquardtOptimizer(graph, initial).optimize();

  // print optimized poses
  std::vector<gtsam::Pose3> optimized_poses;
  for (size_t i = 0; i < ground_truth_poses.size(); ++i)
    optimized_poses.push_back(result.at<gtsam::Pose3>(gtsam::Symbol('x', i)));
  PrintPoses(optimized_poses, full_file_path + "optimized_poses.txt");

  return 0;
}
