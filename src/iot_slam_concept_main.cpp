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

class GaussianDistribution {
 public:
  explicit GaussianDistribution(double mean, double sigma) {
    std::normal_distribution<> gauss_noise_dist_temp{mean, sigma};
    gauss_noise_dist = gauss_noise_dist_temp;
  };
  double GetRandomValue() { return gauss_noise_dist(gen); };

 private:
  std::normal_distribution<> gauss_noise_dist;
  std::random_device rd{};
  std::mt19937 gen{rd()};
};

double randf(double ub, double lb) {
  double f = (double)rand() / RAND_MAX;
  return lb + f * (ub - lb);
}

int RandomSign() { return round(randf(-1, 1)); };

void PerturbTranslation(double perturbation, gtsam::Pose3& delta) {
  // divide total perturbation among xyz axes
  perturbation = std::abs(perturbation);
  const double x_error = RandomSign() * randf(perturbation, 0);
  const double y2_ub = pow(perturbation, 2) - pow(x_error, 2);
  const double y_error = RandomSign() * randf(sqrt(y2_ub), 0);
  const double z2_ub = y2_ub - pow(y_error, 2);
  const double z_error = RandomSign() * sqrt(z2_ub);

  // perturb delta (assume noise is restricted to translation)
  delta = delta.compose(gtsam::Pose3(gtsam::Rot3::Rodrigues(0, 0, 0),
                                     gtsam::Point3(x_error, y_error, z_error)));
}

gtsam::Point3 PerturbPoint(const gtsam::Point3& point,
                           double perturbation = 0) {
  gtsam::Point3 point_perturbed = point;
  if (perturbation > 0) {
    gtsam::Pose3 delta_perturbed = gtsam::Pose3();
    PerturbTranslation(perturbation, delta_perturbed);
    point_perturbed += delta_perturbed.translation();
  }
  return point_perturbed;
}

std::vector<gtsam::Pose3> CreatePoses(const gtsam::Pose3& init,
                                      const gtsam::Pose3& delta, int steps,
                                      double error = 0) {
  GaussianDistribution gauss_dist(error, error);
  std::vector<gtsam::Pose3> poses;
  poses.push_back(init);
  for (int i = 1; i < steps; ++i) {
    gtsam::Pose3 delta_perturbed = delta;
    if (error > 0) {
      PerturbTranslation(gauss_dist.GetRandomValue(), delta_perturbed);
    }
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

  // parameters
  const double ODOMETRY_NOISE = 0.01;  // +/- 1cm relative accuracy from LVIO
  const double WIRELESS_NOISE = 0.05;  // +/- [1,5,10]cm accuracy from UWB
  const double MAX_ANCHOR_TO_TAG_DISTANCE = 15;
  int ODOMETRY_STEPS = 442;  // 442 navigates robot in full circle

  // [4,8] wireless anchors
  std::vector<gtsam::Point3> ground_truth_anchor_points;
  ground_truth_anchor_points.push_back(gtsam::Point3(0, -30, 10));
  ground_truth_anchor_points.push_back(gtsam::Point3(0, 10, 10));
  ground_truth_anchor_points.push_back(gtsam::Point3(40, -30, 10));
  ground_truth_anchor_points.push_back(gtsam::Point3(40, 10, 10));  // 4
  ground_truth_anchor_points.push_back(gtsam::Point3(0, -30, 0));
  ground_truth_anchor_points.push_back(gtsam::Point3(0, 10, 0));
  ground_truth_anchor_points.push_back(gtsam::Point3(40, -30, 0));
  ground_truth_anchor_points.push_back(gtsam::Point3(40, 10, 0));  // 8

  // wireless tags
  std::vector<gtsam::Point3> ground_truth_tag_points;
  ground_truth_tag_points.push_back(gtsam::Point3(0.0, -20.0, 10.0));
  ground_truth_tag_points.push_back(gtsam::Point3(0.0, 0.0, 10.0));
  ground_truth_tag_points.push_back(gtsam::Point3(40.0, -20.0, 10.0));
  ground_truth_tag_points.push_back(gtsam::Point3(40.0, 0.0, 10.0));
  ground_truth_tag_points.push_back(gtsam::Point3(10.0, -30.0, 10.0));
  ground_truth_tag_points.push_back(gtsam::Point3(30.0, -30.0, 10.0));
  ground_truth_tag_points.push_back(gtsam::Point3(10.0, 10.0, 10.0));
  ground_truth_tag_points.push_back(gtsam::Point3(30.0, 10.0, 10.0));

  // noise
  GaussianDistribution odometry_dist(ODOMETRY_NOISE, ODOMETRY_NOISE);
  GaussianDistribution wireless_dist(WIRELESS_NOISE, WIRELESS_NOISE);
  auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(3) << 1e-12, 1e-12, 1e-12).finished());
  auto odometry_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(3) << ODOMETRY_NOISE, ODOMETRY_NOISE, ODOMETRY_NOISE)
          .finished());
  auto wireless_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(3) << WIRELESS_NOISE, WIRELESS_NOISE, WIRELESS_NOISE)
          .finished());

  // generate ground truth and measured robot poses
  gtsam::Pose3 init_pose =
      gtsam::Pose3(gtsam::Rot3::Ypr(0, 0, 0), gtsam::Point3(10, 0, 0));
  gtsam::Pose3 delta_pose =
      gtsam::Pose3(gtsam::Rot3::Ypr(-M_PI / 220, 0, 0),
                   gtsam::Point3(sin(M_PI / 22), sin(M_PI / 22), 0));
  auto ground_truth_robot_poses =
      CreatePoses(init_pose, delta_pose, ODOMETRY_STEPS);
  auto measured_robot_poses =
      CreatePoses(init_pose, delta_pose, ODOMETRY_STEPS, ODOMETRY_NOISE);

  // output to txt files
  std::string full_file_path =
      ros::package::getPath("gt_sam_sandbox") + "/results/";
  PrintPoses(ground_truth_robot_poses,
             full_file_path + "ground_truth_robot_poses.txt");
  PrintPoints(ground_truth_anchor_points,
              full_file_path + "ground_truth_anchor_points.txt");
  PrintPoints(ground_truth_tag_points,
              full_file_path + "ground_truth_tag_points.txt");
  PrintPoses(measured_robot_poses, full_file_path + "measured_robot_poses.txt");

  // Instantiate graph
  gtsam::ExpressionFactorGraph graph;

  // Add prior factor to first robot position
  graph.addExpressionFactor(gtsam::Point3_('x', 0),
                            ground_truth_robot_poses[0].translation(),
                            prior_noise);

  // Simulate measurements from anchors to robot and tags
  for (size_t i = 0; i < ground_truth_robot_poses.size(); ++i) {
    const gtsam::Point3& t_WORLD_ROBOT =
        ground_truth_robot_poses[i].translation();
    for (size_t j = 0; j < ground_truth_anchor_points.size(); ++j) {
      const gtsam::Point3& t_WORLD_ANCHOR = ground_truth_anchor_points[j];
      const gtsam::Point3 t_WORLD_ROBOT_PERTURB =
          PerturbPoint(t_WORLD_ROBOT, wireless_dist.GetRandomValue());
      graph.addExpressionFactor(
          between(gtsam::Point3_('a', j), gtsam::Point3_('x', i)),
          t_WORLD_ANCHOR.between(t_WORLD_ROBOT_PERTURB), wireless_noise);

      for (size_t k = 0; k < ground_truth_tag_points.size(); ++k) {
        const gtsam::Point3& t_WORLD_TAG = ground_truth_tag_points[k];
        if (t_WORLD_ANCHOR.distance(t_WORLD_TAG) <=
            MAX_ANCHOR_TO_TAG_DISTANCE) {
          const gtsam::Point3 t_WORLD_TAG_PERTURB =
              PerturbPoint(t_WORLD_TAG, wireless_dist.GetRandomValue());
          graph.addExpressionFactor(
              between(gtsam::Point3_('a', j), gtsam::Point3_('t', k)),
              t_WORLD_ANCHOR.between(t_WORLD_TAG_PERTURB), wireless_noise);
        }
      }
    }

    if (i > 0) {
      graph.addExpressionFactor(
          between(gtsam::Point3_('x', i - 1), gtsam::Point3_('x', i)),
          measured_robot_poses[i - 1].translation().between(
              measured_robot_poses[i].translation()),
          odometry_noise);
    }
  }
  graph.print();

  // provide initial estimates
  gtsam::Values initial;
  for (size_t i = 0; i < ground_truth_robot_poses.size(); ++i) {
    initial.insert<gtsam::Point3>(gtsam::Symbol('x', i),
                                  ground_truth_robot_poses[i].translation());
  }
  for (size_t i = 0; i < ground_truth_anchor_points.size(); ++i) {
    initial.insert<gtsam::Point3>(gtsam::Symbol('a', i),
                                  ground_truth_anchor_points[i]);
  }
  for (size_t i = 0; i < ground_truth_tag_points.size(); ++i) {
    initial.insert<gtsam::Point3>(gtsam::Symbol('t', i),
                                  ground_truth_tag_points[i]);
  }

  // optimize graph
  gtsam::Values result =
      gtsam::LevenbergMarquardtOptimizer(graph, initial).optimize();

  // print optimized tag positions
  std::vector<gtsam::Point3> optimized_tag_position;
  for (size_t i = 0; i < ground_truth_tag_points.size(); ++i) {
    optimized_tag_position.push_back(
        result.at<gtsam::Point3>(gtsam::Symbol('t', i)));
  }

  PrintPoints(optimized_tag_position,
              full_file_path + "optimized_tag_points.txt");

  return 0;
}
