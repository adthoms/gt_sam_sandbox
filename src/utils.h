#pragma once

#include <fstream>
#include <random>

#include <beam_utils/filesystem.h>
#include <beam_utils/math.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/BearingRange.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <ros/package.h>

using AssociationSet = std::set<int>;
using NodePositionMap = std::map<int, gtsam::Point3>;
using NodeAssociationMap = std::map<int, AssociationSet>;

// parameters
const double GEOMETRY_NOISE = 0.02;    // construction tolerance
const double ODOMETRY_NOISE = 0.05;    // +/- 1cm relative accuracy from LVIO
const double WIRELESS_NOISE = 0.10;    // +/- [1,5,10]cm accuracy from UWB
const double MAX_WIRELESS_RANGE = 15;  // max wireless sensing
const int ODOMETRY_STEPS = 442;        // 442 navigates robot in full circle
const double TAG_OFFSET = 0.2;         // offset between joint and tag
const double ANCHOR_OFFSET = 0.4;      // offset between joint and anchor
const double NUM_TAG_ANCHOR_MEASUREMENTS = 2;  // number of measurements

// variables
NodePositionMap joint_position_map;
NodePositionMap tag_position_map;
NodePositionMap anchor_position_map;
NodeAssociationMap joint_tag_association_map;
NodeAssociationMap tag_anchor_association_map;
NodeAssociationMap floor_anchor_association_map;

// input
std::string JOINTS_POSITION_FILE =
    ros::package::getPath("gt_sam_sandbox") + "/input/joint_positions.txt";
std::string TAGS_POSITION_FILE =
    ros::package::getPath("gt_sam_sandbox") + "/input/tag_positions.txt";
std::string ANCHORS_POSITION_FILE =
    ros::package::getPath("gt_sam_sandbox") + "/input/anchor_positions.txt";

std::string FLOOR_ANCHOR_ASSOCIATION_FILE =
    ros::package::getPath("gt_sam_sandbox") +
    "/input/floor_anchor_association.txt";
std::string JOINT_TAG_ASSOCIATION_FILE =
    ros::package::getPath("gt_sam_sandbox") +
    "/input/joint_tag_association.txt";
std::string TAG_ANCHOR_ASSOCIATION_FILE =
    ros::package::getPath("gt_sam_sandbox") +
    "/input/tag_anchor_association.txt";

// ouput
std::string full_file_path =
    ros::package::getPath("gt_sam_sandbox") + "/results/";

/**
 * @brief This class constructs a Gaussian distribution for generating noise
 * with mean and sigma
 * @param mean mean value of distribution
 * @param sigma standard deviation of distribution
 */
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

/**
 * @brief Randomly assign a positive or negative sign with equal probability
 */
int RandomSign() { return round(beam::randf(-1, 1)); };

/**
 * @brief Load a text file with format [ID X Y Z] into a NodePositionMap map
 */
bool LoadFromTXT(const std::string& input_pose_file_path,
                 NodePositionMap& node_position_map) {
  std::ifstream file(input_pose_file_path);
  std::string s;
  std::string delim{" "};

  while (std::getline(file, s)) {
    std::vector<double> vals;
    if (beam::StringToNumericValues(delim, s, vals)) {
      const int node_id = vals.at(0);
      const gtsam::Point3 node_position =
          gtsam::Point3(vals.at(1), vals.at(2), vals.at(3));
      // assume no duplicate associations
      node_position_map[node_id] = node_position;
    }
  }

  return true;
}

/**
 * @brief load a text file with format [ID_a ID_b] into a
 * NodeAssociationMap map
 */
bool LoadFromTXT(const std::string& input_pose_file_path,
                 NodeAssociationMap& node_association_map) {
  std::ifstream file(input_pose_file_path);
  std::string s;
  std::string delim{" "};

  while (std::getline(file, s)) {
    std::vector<double> vals;
    if (beam::StringToNumericValues(delim, s, vals)) {
      const int node_id = vals.at(0);
      node_association_map[node_id].insert(vals.at(1));
    }
  }

  return true;
}

/**
 * @brief perturb the position of a point with a total perturbation error
 * randomly assigned to each axis
 * @param perturbation total perturbation error
 */
void PerturbPoint(gtsam::Point3& perturbed_point, double perturbation = 0) {
  if (perturbation > 0) {
    // divide total perturbation among xyz axes
    const double total2_error = perturbation * perturbation;
    const double x2_error = beam::randf(total2_error, 0);
    const double y2_error = beam::randf(total2_error - x2_error, 0);
    const double z2_error = total2_error - x2_error - y2_error;
    gtsam::Point3 delta(RandomSign() * sqrt(x2_error),
                        RandomSign() * sqrt(y2_error),
                        RandomSign() * sqrt(z2_error));
    perturbed_point += delta;
  }
}

/**
 * @brief perturb the position of a pose with a total perturbation error
 * randomly assigned to each axis
 * @param perturbation total perturbation error
 */
void PerturbPose(gtsam::Pose3& perturbed_pose, double perturbation = 0) {
  if (perturbation > 0) {
    // assume error in translation only
    gtsam::Point3 perturbed_translation;
    PerturbPoint(perturbed_translation, perturbation);

    // perturb delta (assume noise is restricted to translation)
    gtsam::Pose3 delta_perturbed =
        gtsam::Pose3(gtsam::Rot3::Rodrigues(0, 0, 0), perturbed_translation);
    perturbed_pose = perturbed_pose.compose(delta_perturbed);
  }
}

/**
 * @brief generate vector of poses with prescribed noise
 * @param init initial pose of vector
 * @param delta relative translation between pose
 * @param steps number of transitions between pose
 * @param error total error on pose position
 */
std::vector<gtsam::Pose3> GeneratePoses(const gtsam::Pose3& init,
                                        const gtsam::Pose3& delta, int steps,
                                        double error = 0) {
  // assume average error with equal std dev
  GaussianDistribution gauss_dist(error, error);
  std::vector<gtsam::Pose3> poses;

  // perturb initial pose
  gtsam::Pose3 delta_init;
  if (error > 0) {
    PerturbPose(delta_init,
                gauss_dist.GetRandomValue() * delta.translation().norm());
  }
  poses.push_back(init.compose(delta_init));

  // perturb poses incrementally to simulate drift
  for (int i = 1; i < steps; ++i) {
    gtsam::Pose3 delta_perturbed = delta;
    if (error > 0) {
      PerturbPose(delta_perturbed,
                  gauss_dist.GetRandomValue() * delta.translation().norm());
    }
    poses.push_back(poses[i - 1].compose(delta_perturbed));
  }

  return poses;
}

/**
 * @brief Write points to a txt file
 * @param points a vector of points
 * @param output_dir directory to write file
 */
void PrintPoints(const std::vector<gtsam::Point3>& points,
                 const std::string& output_dir) {
  std::ofstream file(output_dir);
  for (gtsam::Point3 n : points) {
    file << n.x() << " " << n.y() << " " << n.z() << std::endl;
  }
}

/**
 * @brief Write poses to a txt file
 * @param poses a vector of poses
 * @param output_dir directory to write file
 */
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

/**
 * @brief Write NodePositionMap to a txt file
 * @param node_position_map instance of NodePositionMap to be written
 * @param output_dir directory to write file
 */
void PrintMap(const NodePositionMap& node_position_map,
              const std::string& output_dir) {
  std::ofstream file(output_dir);
  for (const auto& [id, position] : node_position_map) {
    file << id << " " << position.x() << " " << position.y() << " "
         << position.z() << std::endl;
  }
}

/**
 * @brief Write NodeAssociationMap to a txt file
 * @param node_position_map instance of NodeAssociationMap to be written
 * @param output_dir directory to write file
 */
void PrintMap(const NodeAssociationMap& node_association_map,
              const std::string& output_dir) {
  std::ofstream file(output_dir);
  for (const auto& [id, association] : node_association_map) {
    file << id << " ";
    for (auto iter = association.begin(); iter != association.end(); ++iter) {
      file << *iter << " ";
    }
    file << std::endl;
  }
}
