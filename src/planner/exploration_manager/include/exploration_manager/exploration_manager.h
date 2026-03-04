#ifndef _EXPLORATION_MANAGER_H_
#define _EXPLORATION_MANAGER_H_

// Third-party libraries
#include <Eigen/Eigen>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Standard C++ libraries
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

// ROS core
#include <ros/ros.h>

// Plan environment
#include <plan_env/frontier_map2d.h>
#include <plan_env/object_map2d.h>
#include <plan_env/sdf_map2d.h>
#include <plan_env/value_map2d.h>
#include <plan_env/ig_value_map.h>

// Path searching
#include <path_searching/astar2d.h>

using Eigen::Vector2d;
using Eigen::Vector3d;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

namespace apexnav_planner {
class SDFMap2D;
class FrontierMap2D;
struct ExplorationParam;
struct ExplorationData;

struct SemanticFrontier {
  Vector2d position;      ///< 2D position of the frontier
  double semantic_value;  ///< Semantic value at the frontier location
  double path_length;     ///< Path length to reach this frontier
  vector<Vector2d> path;  ///< Complete path to the frontier

  bool operator<(const SemanticFrontier& other) const
  {
    if (fabs(semantic_value - other.semantic_value) < 1e-2) {
      // If semantic values are approximately equal (within 1%), sort by path length (ascending)
      return path_length < other.path_length;
    }
    // Otherwise, sort by semantic value (descending)
    return semantic_value > other.semantic_value;
  }
};

enum EXPL_RESULT {
  EXPLORATION,               ///< Normal exploration mode
  SEARCH_BEST_OBJECT,        ///< Found high-confidence object
  SEARCH_OVER_DEPTH_OBJECT,  ///< Searching over-depth object
  SEARCH_SUSPICIOUS_OBJECT,  ///< Investigating suspicious object
  NO_PASSABLE_FRONTIER,      ///< No reachable frontiers available
  NO_COVERABLE_FRONTIER,     ///< No coverable frontiers found
  SEARCH_EXTREME,            ///< Extreme search mode activated
  SEARCH_SIMILAR_OBJECT      ///< Searching similar objects for re-detection
};

class ExplorationManager {
public:
  ExplorationManager() = default;
  ~ExplorationManager();  // Explicit destructor declaration for shared_ptr with forward declaration

  void initialize(ros::NodeHandle& nh);

  int planNextBestPoint(const Vector3d& pos, const double& yaw);
  void getSortedSemanticFrontiers(const Vector2d& cur_pos, const vector<Vector2d>& frontiers,
      vector<SemanticFrontier>& sem_frontiers);
  void calcSemanticFrontierInfo(const vector<SemanticFrontier>& sem_frontiers, double& std_dev,
      double& max_to_mean, double& mean, bool if_print = false);

  /// Compute adaptive alpha based on max semantic value across all frontiers
  /// alpha(t) = alpha_base + (1 - alpha_base) * max_fi(V_HSVM(fi)) / tau_conf
  double computeAdaptiveAlpha(const vector<Vector2d>& frontiers);

  shared_ptr<ExplorationData> ed_;            ///< Exploration data container
  shared_ptr<ExplorationParam> ep_;           ///< Exploration parameters
  unique_ptr<Astar2D> path_finder_;           ///< A* path finding algorithm
  shared_ptr<FrontierMap2D> frontier_map2d_;  ///< 2D frontier map
  shared_ptr<ObjectMap2D> object_map2d_;      ///< 2D object map
  shared_ptr<SDFMap2D> sdf_map_;              ///< Signed distance field map

  // Suspicious target locking (prevent oscillation)
  void setSuspiciousTargetLock(bool locked, const Vector2d& pos = Vector2d(0, 0), int object_id = -1);
  bool isSuspiciousTargetLocked() const { return suspicious_target_locked_; }
  Vector2d getLockedSuspiciousPos() const { return locked_suspicious_pos_; }
  int getLockedSuspiciousObjectId() const { return locked_suspicious_object_id_; }

  // Path Search Utils (public for stucking recovery fallback)
  bool searchObjectPath(const Vector3d& start,
      const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
      Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path,
      int object_id = -1);  ///< object_id for failed approach point checking (-1 to skip check)

  typedef shared_ptr<ExplorationManager> Ptr;

private:
  // Exploration Policy
  void chooseExplorationPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
      Vector2d& next_best_pos, vector<Vector2d>& next_best_path);
  void findClosestFrontierPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
      Vector2d& next_best_pos, vector<Vector2d>& next_best_path);
  void findHighestSemanticsFrontierPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
      Vector2d& next_best_pos, vector<Vector2d>& next_best_path);
  void hybridExplorePolicy(Vector2d cur_pos, vector<Vector2d> frontiers, Vector2d& next_best_pos,
      vector<Vector2d>& next_best_path);
  void findTSPTourPolicy(Vector2d cur_pos, vector<Vector2d> frontiers, Vector2d& next_best_pos,
      vector<Vector2d>& next_best_path);

  // WTRP (Weighted Traveling Repairman Problem) Policy
  void wtrpExplorePolicy(Vector2d cur_pos, vector<Vector2d> frontiers, Vector2d& next_best_pos,
      vector<Vector2d>& next_best_path);
  void solveWTRP(const Eigen::MatrixXd& cost_matrix, const vector<double>& weights,
      vector<int>& best_order, double& best_cost);
  void computeSoftmaxWeights(const vector<SemanticFrontier>& sem_frontiers,
      double temperature, vector<double>& weights);
  bool searchObjectPathExtreme(const Vector3d& start,
      const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
      Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path);
  bool searchFrontierPath(const Vector2d& start, const Vector2d& end, Eigen::Vector2d& refined_pos,
      std::vector<Eigen::Vector2d>& refined_path);
  void shortenPath(vector<Vector2d>& path);

  // Helper functions for object path searching
  Vector2d findNearestObjectPoint(
      const Vector3d& start, const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud);
  std::vector<Vector2d> findCandidateObjectPoints(
      const Vector3d& start, const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
      int max_candidates = 5);  ///< Find multiple candidate approach points
  bool trySearchObjectPathWithDistance(const Vector2d& start2d, const Vector2d& object_pose,
      double distance, double max_search_time, Eigen::Vector2d& refined_pos,
      std::vector<Eigen::Vector2d>& refined_path, const std::string& debug_msg);

  // TSP Optimization Methods
  void computeATSPTour(
      const Vector2d& cur_pos, const vector<Vector2d>& frontiers, vector<int>& indices);
  void computeATSPCostMatrix(
      const Vector2d& cur_pos, const vector<Vector2d>& frontiers, Eigen::MatrixXd& cost_matrix);
  double computePathCost(const Vector2d& pos1, const Vector2d& pos2);
  vector<Vector2i> allNeighbors(const Eigen::Vector2i& idx, int grid_radius);

  ros::ServiceClient tsp_client_;         ///< ROS service client for TSP solver

  // Suspicious target locking state
  bool suspicious_target_locked_;       ///< Whether a suspicious target is locked
  Vector2d locked_suspicious_pos_;      ///< Locked target position
  int locked_suspicious_object_id_;     ///< Locked suspicious target object ID (-1 if unknown)
  unique_ptr<RayCaster2D> ray_caster2d_;  ///< Ray casting for collision checking

  // High-confidence object approach point locking (prevent oscillation between candidate points)
  bool object_approach_locked_;              ///< Whether an approach point is locked
  Vector2d locked_object_approach_pos_;      ///< Locked approach point position
  int locked_object_id_;                     ///< ID of the locked object (-1 if none)

  // WTRP target hysteresis (prevent oscillation between frontiers with similar values)
  bool has_last_wtrp_target_;                ///< Whether we have a previous WTRP target
  Vector2d last_wtrp_target_;                ///< Last selected WTRP target position
  double last_wtrp_target_value_;            ///< Semantic value of last target

public:
  /// Lock the current approach point for high-confidence object navigation
  void setObjectApproachLock(bool locked, const Vector2d& pos = Vector2d(0, 0), int object_id = -1);
  bool isObjectApproachLocked() const { return object_approach_locked_; }
  Vector2d getLockedObjectApproachPos() const { return locked_object_approach_pos_; }
  int getLockedObjectId() const { return locked_object_id_; }

  /// Reset WTRP target hysteresis state (call when frontier becomes dormant or unreachable)
  void resetWTRPHysteresis();
  bool hasWTRPTarget() const { return has_last_wtrp_target_; }
  Vector2d getLastWTRPTarget() const { return last_wtrp_target_; }
};

inline bool ExplorationManager::searchFrontierPath(const Vector2d& start, const Vector2d& end,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path)
{
  path_finder_->reset();
  if (path_finder_->astarSearch(start, end, 0.25, 0.01) == Astar2D::REACH_END) {
    refined_pos = end;
    refined_path = path_finder_->getPath();
    return true;
  }
  return false;
}

inline bool ExplorationManager::searchObjectPathExtreme(const Vector3d& start,
    const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path)
{
  // Maximum acceptable distance from final position to object (same as normal search)
  constexpr double MAX_ACCEPTABLE_OBJECT_DISTANCE = 2.0;  // meters

  Vector2d object_pose = findNearestObjectPoint(start, object_cloud);
  if (object_pose.x() < -999.0)
    return false;  // Error finding nearest point

  Vector2d start2d = Vector2d(start(0), start(1));
  path_finder_->reset();
  if (path_finder_->astarSearch(start2d, object_pose, 0.25, 0.2, Astar2D::SAFETY_MODE::EXTREME) ==
      Astar2D::REACH_END) {
    refined_path = path_finder_->getPath();

    // Check if path endpoint is close enough to object
    if (!refined_path.empty()) {
      double dist_to_object = (refined_path.back() - object_pose).norm();
      if (dist_to_object > MAX_ACCEPTABLE_OBJECT_DISTANCE) {
        ROS_WARN("[ObjectPathExtreme] Path endpoint too far from object: %.2fm > %.2fm",
                 dist_to_object, MAX_ACCEPTABLE_OBJECT_DISTANCE);
        return false;
      }
    }

    refined_pos = object_pose;
    return true;
  }
  return false;
}

inline void ExplorationManager::shortenPath(vector<Vector2d>& path)
{
  if (path.empty()) {
    ROS_ERROR("Empty path to shorten");
    return;
  }

  // Shorten the path by keeping only critical intermediate points
  const double dist_thresh = 3.0;  // Minimum distance threshold for waypoint retention
  vector<Vector2d> short_tour = { path.front() };

  for (int i = 1; i < (int)path.size() - 1; ++i) {
    if ((path[i] - short_tour.back()).norm() > dist_thresh)
      short_tour.push_back(path[i]);
    else {
      // Add waypoints only when necessary to avoid collision
      ray_caster2d_->input(short_tour.back(), path[i + 1]);
      Eigen::Vector2i idx;
      while (ray_caster2d_->nextId(idx) && ros::ok()) {
        if (sdf_map_->getInflateOccupancy(idx) == 1 ||
            sdf_map_->getOccupancy(idx) == SDFMap2D::UNKNOWN) {
          short_tour.push_back(path[i]);
          break;
        }
      }
    }
  }

  // Always include the final destination
  if ((path.back() - short_tour.back()).norm() > 1e-3)
    short_tour.push_back(path.back());

  // Ensure minimum path complexity (at least three points)
  if (short_tour.size() == 2)
    short_tour.insert(short_tour.begin() + 1, 0.5 * (short_tour[0] + short_tour[1]));

  path = short_tour;
}

inline vector<Eigen::Vector2i> ExplorationManager::allNeighbors(
    const Eigen::Vector2i& idx, int grid_radius)
{
  vector<Eigen::Vector2i> neighbors;

  for (int x = -grid_radius; x <= grid_radius; ++x) {
    for (int y = -grid_radius; y <= grid_radius; ++y) {
      if (x == 0 && y == 0)
        continue;  // Skip center point
      Eigen::Vector2i offset(x, y);
      neighbors.push_back(idx + offset);
    }
  }
  return neighbors;
}

}  // namespace apexnav_planner

#endif