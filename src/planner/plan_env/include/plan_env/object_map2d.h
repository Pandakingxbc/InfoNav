#ifndef _OBJECT_MAP2D_H_
#define _OBJECT_MAP2D_H_

// ROS and system includes
#include <ros/ros.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <list>
#include <utility>
#include <set>
#include <map>

// Internal mapping components
#include <plan_env/sdf_map2d.h>
#include <plan_env/raycast2d.h>

// ROS messages for Patience-Aware Navigation
#include <plan_env/SemanticScores.h>

// PCL for point cloud processing
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <unordered_map>
#include <set>

using Eigen::Vector2d;
using Eigen::Vector2i;
using Eigen::Vector3d;
using std::list;
using std::pair;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

class RayCaster2D;

namespace apexnav_planner {
class SDFMap2D;

static const char* T_COLORS[] = {
  "\033[0m",     ///< Default color [0]
  "\033[1;31m",  ///< Red color [1]
  "\033[1;32m",  ///< Green color [2]
  "\033[1;33m",  ///< Yellow color [3]
  "\033[1;34m",  ///< Blue color [4]
  "\033[1;35m",  ///< Purple color [5]
  "\033[1;36m"   ///< Cyan color [6]
};

struct DetectedObject {
  pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> cloud;  ///< 3D point cloud of detected object
  double score;                                           ///< Confidence score from detector (0-1)
  int label;                                              ///< Semantic class label from detection
};

struct Viewpoint2D {
  Vector2d pos_;   ///< 2D position of viewpoint in world coordinates
  double yaw_;     ///< Heading angle in radians
  int visib_num_;  ///< Number of visible objects from this viewpoint
};

/**
 * @brief Object cluster with multi-modal information
 *
 * Represents a semantic object cluster combining 2D grid information,
 * 3D point clouds from multiple detections, confidence scores, and
 * geometric properties for robust object representation.
 */
struct ObjectCluster {
  /******* 2D Grid Information *******/
  int id_;                               ///< Unique cluster identifier
  vector<Vector2d> cells_;               ///< All 2D grid cells belonging to this cluster
  unordered_map<int, int> seen_counts_;  ///< Observation count per grid cell
  unordered_map<int, char> visited_;     ///< Visited flag per grid cell
  Vector2d average_;                     ///< Centroid position of all grid cells
  Vector2d box_min2d_, box_max2d_;       ///< 2D bounding box (min/max corners)
  Vector3d box_min3d_, box_max3d_;       ///< 3D bounding box from point clouds
  int max_seen_count_;                   ///< Maximum observation count across all cells
  vector<Vector2d> good_cells_;          ///< High-confidence cells (frequently observed)
  int best_label_;                       ///< Most confident semantic label

  /******* 3D Point Cloud Information *******/
  vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> clouds_;  ///< Point clouds per semantic
                                                                    ///< class
  vector<double> confidence_scores_;    ///< Confidence scores per semantic class
  vector<int> observation_nums_;        ///< Number of observations per class
  vector<int> observation_cloud_sums_;  ///< Total point count per class

  /******* Patience-Aware Navigation (Paper Section 3.5) *******/
  double fused_confidence_;  ///< Temporal fused confidence (compared with τ(t))

  /******* Detection Direction Information *******/
  Vector2d first_detection_pos_;      ///< Robot position when object was first detected
  bool has_detection_pos_;            ///< Whether detection position is recorded

  /******* VLM Verification State *******/
  bool vlm_verified_;                 ///< Whether this object has been VLM verified
  bool vlm_result_;                   ///< VLM verification result (true=confirmed real target)
  bool vlm_pending_;                  ///< Whether VLM verification is in progress
  bool vlm_uncertain_;                ///< Whether VLM returned UNCERTAIN (verified but inconclusive)
  double vlm_adjusted_confidence_;    ///< Confidence after VLM adjustment (-1 if not adjusted)

  /**
   * @brief Constructor to initialize multi-class storage
   * @param size Number of semantic classes to support (default: 6)
   *        label 0 = target object, label 1-5 = similar objects
   */
  ObjectCluster(int size = 6)
    : clouds_(size)
    , confidence_scores_(size, 0.0)
    , observation_nums_(size, 0)
    , observation_cloud_sums_(size, 0)
    , fused_confidence_(0.0)
    , first_detection_pos_(Vector2d::Zero())
    , has_detection_pos_(false)
    , vlm_verified_(false)
    , vlm_result_(false)
    , vlm_pending_(false)
    , vlm_uncertain_(false)
    , vlm_adjusted_confidence_(-1.0)
  {
  }
};

class ObjectMap2D {
public:
  ObjectMap2D(SDFMap2D* sdf_map, ros::NodeHandle& nh);
  ~ObjectMap2D() = default;

  int searchSingleObjectCluster(const DetectedObject& detected_object);
  void inputObservationObjectsCloud(
      const vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> observation_clouds,
      const double& itm_score);
  void setConfidenceThreshold(double val);
  void clearObjects();  // Clear all objects for new episode

  void getAllConfidenceObjectClouds(pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_clouds);
  void getTopConfidenceObjectCloud(
      vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>>& top_object_clouds,
      bool limited_confidence = true, bool extreme = false);

  /**
   * @brief Get point clouds of similar objects (best_label_ != 0)
   *
   * Returns objects detected as similar to target but not confidently
   * classified as target. Used for re-detection attempts when no frontier.
   *
   * @param similar_clouds Output: point clouds of similar objects
   * @param object_ids Output: corresponding object IDs
   */
  void getSimilarObjectClouds(
      std::vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>>& similar_clouds,
      std::vector<int>& object_ids);

  void getObjects(
      vector<vector<Vector2d>>& clusters, vector<Vector2d>& averages, vector<int>& labels);
  void getObjectBoxes(vector<pair<Vector2d, Vector2d>>& boxes);
  void getObjectBoxes(vector<pair<Vector3d, Vector3d>>& boxes);
  void getObjectBoxes(vector<Vector3d>& bmin, vector<Vector3d>& bmax);
  int getObjectGrid(const Eigen::Vector2d& pos) const;
  int getObjectGrid(const Eigen::Vector2i& id) const;
  int getObjectGrid(const int& adr) const;

  void publishObjectClouds();
  void wrapYaw(double& yaw);

  // ====== Patience-Aware Navigation Methods (Paper Section 3.5) ======

  /**
   * @brief Get current dynamic confidence threshold τ(t)
   * @return Current threshold τ(t) = τ_high - (t/T_max) × (τ_high - τ_low)
   */
  double getCurrentConfidenceThreshold() const;

  /**
   * @brief Update current exploration step (called each frame)
   * @param step Current step count
   */
  void setCurrentStep(int step);

  /**
   * @brief Set current target object category
   * @param category Target category name (e.g., "bed", "toilet", "plant")
   */
  void setTargetCategory(const std::string& category);

  /**
   * @brief Check if an object is a confirmed target (dynamic judgment)
   * @param obj_idx Object index
   * @return confidence >= τ(t)
   */
  bool isConfirmedTarget(int obj_idx) const;

  /**
   * @brief Check if any confirmed target exists (dynamic judgment)
   * @return Whether at least one confirmed target exists
   */
  bool hasConfirmedTarget() const;

  /**
   * @brief Get all confirmed target point clouds (dynamic judgment)
   * @param confirmed_clouds Output: list of confirmed target point clouds
   */
  void getConfirmedTargetClouds(
      std::vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>>& confirmed_clouds);


  /**
   * @brief Get current exploration step count
   * @return Current step count
   */
  int getCurrentStep() const { return current_step_; }

  /**
   * @brief Get current target category
   * @return Current target category name
   */
  const std::string& getTargetCategory() const { return current_target_category_; }

  // ====== VLM Validation Methods ======

  /**
   * @brief Mark an object as invalid (failed VLM validation)
   * @param object_id The object ID to mark as invalid
   */
  void markObjectAsInvalid(int object_id);

  /**
   * @brief Check if an object is marked as invalid
   * @param object_id The object ID to check
   * @return true if object is marked invalid
   */
  bool isObjectInvalid(int object_id) const;

  /**
   * @brief Get object info for VLM verification prompt
   * @param object_id Target object ID
   * @param fused_confidence Output: fused confidence of the object
   * @param observation_count Output: observation count for target label (label 0)
   * @param current_threshold Output: current dynamic threshold tau(t)
   * @return true if object exists
   */
  bool getObjectInfoForVLM(int object_id, double& fused_confidence,
                           int& observation_count, double& current_threshold) const;

  /**
   * @brief Get similar objects info string for VLM prompt
   * Format: "label_name:conf:count,label_name:conf:count,..."
   * @param object_id Target object ID
   * @return Similar objects info string, or empty string if no similar objects
   */
  std::string getSimilarObjectsInfoForVLM(int object_id) const;

  /**
   * @brief Get the ID of the current navigation target object
   * @return Object ID, or -1 if no target
   */
  int getCurrentTargetObjectId() const { return current_target_object_id_; }

  /**
   * @brief Set the current navigation target object ID
   * @param object_id The object ID being navigated to
   */
  void setCurrentTargetObjectId(int object_id) { current_target_object_id_ = object_id; }

  /**
   * @brief Clear all invalid object markers (for new episode)
   */
  void clearInvalidObjects();

  // ====== VLM Verification Methods ======

  /**
   * @brief Apply VLM verification result to object confidence
   *
   * If VLM confirms the object is real (is_real_target=true):
   *   - Significantly boost fused_confidence (weighted fusion with high target)
   *   - Mark vlm_verified=true, vlm_result=true
   *
   * If VLM rejects the object as false positive (is_real_target=false):
   *   - Significantly reduce fused_confidence (weighted fusion with low target)
   *   - Mark vlm_verified=true, vlm_result=false
   *   - Optionally mark as invalid to prevent re-selection
   *
   * @param object_id The object ID to update
   * @param decision_level VLM judgment: 1=CONFIRM, 0=UNCERTAIN, -1=REJECT
   * @param vlm_confidence VLM's confidence in its judgment (0-1)
   */
  void applyVLMVerificationResult(int object_id, int decision_level, double vlm_confidence);

  /**
   * @brief Check if an object has been VLM verified
   * @param object_id The object ID to check
   * @return true if VLM verification has been performed
   */
  bool isObjectVLMVerified(int object_id) const;

  /**
   * @brief Get VLM verification result for an object
   * @param object_id The object ID to check
   * @return true if VLM confirmed as real target, false otherwise
   */
  bool getObjectVLMResult(int object_id) const;

  /**
   * @brief Check if VLM returned UNCERTAIN for an object
   * @param object_id The object ID to check
   * @return true if VLM returned UNCERTAIN (verified but inconclusive)
   */
  bool isObjectVLMUncertain(int object_id) const;

  /**
   * @brief Set VLM pending state for an object
   * @param object_id The object ID
   * @param pending Whether VLM verification is pending
   */
  void setObjectVLMPending(int object_id, bool pending);

  /**
   * @brief Check if VLM verification is pending for an object
   * @param object_id The object ID to check
   * @return true if VLM verification is in progress
   */
  bool isObjectVLMPending(int object_id) const;

  /**
   * @brief Get object IDs for top confidence objects (same order as getTopConfidenceObjectCloud)
   * @param object_ids Output: vector of object IDs corresponding to top_object_clouds
   * @param limited_confidence Whether to apply confidence threshold filtering
   */
  void getTopConfidenceObjectIds(std::vector<int>& object_ids, bool limited_confidence = true);

  // ====== Multi-View VLM Validation Methods ======

  /**
   * @brief Observation viewpoint for VLM validation
   * Contains position, yaw angle, and whether it's reachable
   */
  struct ObservationViewpoint {
    Vector2d position;      ///< 2D position in world coordinates
    double yaw;             ///< Facing angle (towards object center)
    bool is_reachable;      ///< Whether this viewpoint is reachable
    bool has_line_of_sight; ///< Whether there's clear line of sight to object
    double distance;        ///< Distance to object center
  };

  /**
   * @brief Compute optimal observation viewpoints for VLM validation
   *
   * Generates candidate viewpoints around the target object based on:
   * - Object size (bounding box)
   * - Camera horizontal FOV (to ensure object fits in view)
   * - Minimum safe distance from obstacles
   * - Multiple angles (front, left 45°, right 45°)
   *
   * @param object_id Target object ID
   * @param robot_pos Current robot position
   * @param robot_yaw Current robot yaw (used to prioritize frontal views)
   * @param hfov_deg Camera horizontal FOV in degrees (default: 79°)
   * @param viewpoints Output: vector of candidate viewpoints sorted by quality
   * @return true if at least one valid viewpoint found
   */
  bool computeObservationViewpoints(
      int object_id,
      const Vector2d& robot_pos,
      double robot_yaw,
      double hfov_deg,
      std::vector<ObservationViewpoint>& viewpoints);

  /**
   * @brief Get object center position
   * @param object_id Target object ID
   * @param center Output: object center in 2D
   * @return true if object exists
   */
  bool getObjectCenter(int object_id, Vector2d& center) const;

  /**
   * @brief Get object bounding box size
   * @param object_id Target object ID
   * @param size Output: (width, height) in 2D
   * @return true if object exists
   */
  bool getObjectSize2D(int object_id, Vector2d& size) const;

  /**
   * @brief Check if there's clear line of sight from position to object
   * @param from_pos Observer position
   * @param object_id Target object ID
   * @return true if line of sight is clear (no obstacles)
   */
  bool hasLineOfSight(const Vector2d& from_pos, int object_id) const;

  /**
   * @brief Find the nearest object to a given position
   * @param pos Query position
   * @param max_distance Maximum search distance (default: 2.0m)
   * @return Object ID of nearest object, or -1 if none found within max_distance
   */
  int findNearestObjectId(const Vector2d& pos, double max_distance = 2.0) const;

  pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> all_object_clouds_;
  pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> over_depth_object_cloud_;

private:
  double fusionConfidenceScore(
      int total_last, double c_last, int n_now, double c_now, int total_now, int sum);
  void updateObjectBestLabel(int obj_idx);
  Eigen::Vector4d getColor(const double& h, double alpha);

  bool haveOverlap(
      const Vector2d& min1, const Vector2d& max1, const Vector2d& min2, const Vector2d& max2);
  void createNewObjectCluster(
      const std::vector<Eigen::Vector2d>& cells, const DetectedObject& detected_object);
  void mergeCellsIntoObjectCluster(const int& object_id, const std::vector<Eigen::Vector2d>& cells,
      const DetectedObject& detected_object);

  vector<Eigen::Vector2i> fourNeighbors(const Eigen::Vector2i& idx);
  vector<Eigen::Vector2i> allNeighbors(const Eigen::Vector2i& idx);
  vector<Eigen::Vector2i> allGridsDistance(const Eigen::Vector2i& idx, const double& dist);
  bool isConfidenceObject(const ObjectCluster& obj);
  bool isNeighborUnknown(const Eigen::Vector2i& idx);
  bool isSatisfyObject(const Eigen::Vector2i& idx);
  bool isSatisfyObject(const Eigen::Vector2d& pos);
  bool isObjectClustered(const int& adr);
  bool isObjectClustered(const Eigen::Vector2d& pos);
  bool isObjectClustered(const Eigen::Vector2i& idx);
  void printFusionInfo(const ObjectCluster& obj, int label, const char* state);

  // Wrapper of sdf map
  int toAdr(const Eigen::Vector2d& pos);
  int toAdr(const Eigen::Vector2i& idx);
  bool knownFree(const Eigen::Vector2i& idx);
  bool inMap(const Eigen::Vector2i& idx);
  bool checkSafety(const Eigen::Vector2i& idx);
  bool checkSafety(const Eigen::Vector2d& pos);

  // ==================== Data Members ====================
  ros::Publisher object_cloud_pub_;  ///< Publisher for colored object visualization

  // Object storage and indexing
  vector<int> object_indexs_;      ///< Grid cell to object ID mapping
  vector<char> object_buffer_;     ///< Object occupancy grid buffer
  vector<ObjectCluster> objects_;  ///< Collection of all object clusters

  // Algorithm parameters
  bool use_observation_;     ///< Whether to use observation-based confidence reduction
  bool is_vis_cloud_;        ///< Whether to publish visualization clouds
  int fusion_type_;          ///< Confidence fusion algorithm type (0=replace, 1=weighted, 2=max)
  int min_observation_num_;  ///< Minimum observations required for confidence
  double min_confidence_;    ///< Minimum confidence threshold for object acceptance
  double resolution_;        ///< Grid resolution in meters
  double leaf_size_;         ///< Voxel size for point cloud downsampling

  // ====== Patience-Aware Navigation Parameters (Paper Section 3.5) ======
  double tau_high_;           ///< Initial strict threshold (default 0.4)
  double tau_low_;            ///< Final relaxed threshold (default 0.2)
  int T_max_;                 ///< Maximum exploration steps (default 500)
  int current_step_;          ///< Current exploration step count
  double min_detection_confidence_;  ///< Minimum detection confidence to add to ObjectMap (filter low-confidence detections)

  // ====== Target-specific thresholds ======
  // Each object has its own tau_high and tau_low for dynamic threshold
  struct TargetThresholdConfig {
    double tau_high;  ///< Initial strict threshold for this object type
    double tau_low;   ///< Final relaxed threshold for this object type
  };
  std::unordered_map<std::string, TargetThresholdConfig> target_threshold_configs_;
  std::string current_target_category_;  ///< Current search target category

  /**
   * @brief Initialize target-specific threshold mapping table
   * @param nh ROS NodeHandle for reading parameters
   */
  void initTargetSpecificThresholds(ros::NodeHandle& nh);

  /**
   * @brief Get threshold config for current target category
   * @return TargetThresholdConfig with tau_high and tau_low
   */
  TargetThresholdConfig getTargetThresholdConfig() const;

  /**
   * @brief Callback for SemanticScores message to extract target category
   */
  void semanticScoresCallback(const plan_env::SemanticScores::ConstPtr& msg);

  // System integration
  SDFMap2D* sdf_map_;
  unique_ptr<RayCaster2D> raycaster_;
  ros::Subscriber semantic_scores_sub_;  ///< Subscriber for target category from SemanticScores
  ros::NodeHandle nh_;  ///< NodeHandle for reloading parameters on episode reset

  // ====== VLM Validation State ======
  std::set<int> invalid_object_ids_;     ///< Set of object IDs that failed VLM validation
  int current_target_object_id_;         ///< Current navigation target object ID (-1 if none)

  // VLM confidence fusion parameters
  double vlm_positive_weight_;           ///< Weight for VLM positive result (default 0.7)
  double vlm_positive_boost_;            ///< Target confidence for VLM positive (default 0.85)
  double vlm_negative_weight_;           ///< Weight for VLM negative result (default 0.8)
  double vlm_negative_penalty_;          ///< Target confidence for VLM negative (default 0.05)

  // ====== Failed Approach Points (Approach Detection Check) ======
  // Records approach points that failed detection check (robot reached but couldn't see object)
  // Used to avoid repeatedly navigating to same invalid approach points
  std::map<int, std::vector<Eigen::Vector2d>> failed_approach_points_;  ///< object_id -> failed positions

  // ====== Unreachable Objects ======
  // Objects that have too many failed approach attempts are marked unreachable
  // These will be excluded from navigation target selection
  std::set<int> unreachable_objects_;  ///< Set of object IDs marked as unreachable

public:
  // ====== Failed Approach Point Management ======

  /**
   * @brief Add a failed approach point for an object
   * Called when robot reaches target position but detection check fails
   * @param object_id The object ID
   * @param point The approach point that failed
   */
  void addFailedApproachPoint(int object_id, const Eigen::Vector2d& point);

  /**
   * @brief Check if an approach point has failed before
   * @param object_id The object ID
   * @param point The approach point to check
   * @param threshold Distance threshold for considering points as "same" (default 0.5m)
   * @return true if this point (or nearby point) has failed before
   */
  bool isApproachPointFailed(int object_id, const Eigen::Vector2d& point, double threshold = 0.5) const;

  /**
   * @brief Clear failed approach points for an object (e.g., when object is removed)
   * @param object_id The object ID, or -1 to clear all
   */
  void clearFailedApproachPoints(int object_id = -1);

  /**
   * @brief Get number of failed approach points for an object
   * @param object_id The object ID
   * @return Number of failed approach points
   */
  int getFailedApproachPointCount(int object_id) const;

  /**
   * @brief Mark an object as unreachable (too many failed approach attempts)
   * Unreachable objects will be excluded from navigation target selection
   * @param object_id The object ID to mark as unreachable
   */
  void markObjectUnreachable(int object_id);

  /**
   * @brief Check if an object is marked as unreachable
   * @param object_id The object ID to check
   * @return true if object is unreachable
   */
  bool isObjectUnreachable(int object_id) const;

  /**
   * @brief Clear unreachable status for an object (e.g., if new observation suggests reachability)
   * @param object_id The object ID, or -1 to clear all
   */
  void clearUnreachableStatus(int object_id = -1);

  /**
   * @brief Get count of unreachable objects
   * @return Number of objects marked as unreachable
   */
  int getUnreachableObjectCount() const { return static_cast<int>(unreachable_objects_.size()); }

  // ====== Detection Direction Methods ======

  /**
   * @brief Set current robot position (called before object detection)
   * Used to record first detection position for objects
   * @param pos Current robot position in 2D
   */
  void setCurrentRobotPosition(const Vector2d& pos) { current_robot_pos_ = pos; }

  /**
   * @brief Get first detection position for an object
   * @param object_id The object ID
   * @param detection_pos Output: first detection position
   * @return true if object exists and has detection position recorded
   */
  bool getFirstDetectionPosition(int object_id, Vector2d& detection_pos) const;

private:
  Vector2d current_robot_pos_;  ///< Current robot position for recording detection direction
};

inline void ObjectMap2D::printFusionInfo(const ObjectCluster& obj, int label, const char* state)
{
  // Use purple for high-confidence label 0, green for others
  if (label == 0 && obj.confidence_scores_[label] >= min_confidence_)
    ROS_WARN("%s%s id = %d label = %d confidence score = %.3lf %s", T_COLORS[5], state, obj.id_,
        label, obj.confidence_scores_[label], T_COLORS[0]);
  else
    ROS_WARN("%s%s id = %d label = %d confidence score = %.3lf %s", T_COLORS[2], state, obj.id_,
        label, obj.confidence_scores_[label], T_COLORS[0]);
}

inline bool ObjectMap2D::isSatisfyObject(const Eigen::Vector2i& idx)
{
  Vector2d pos;
  sdf_map_->indexToPos(idx, pos);
  return isSatisfyObject(pos);
}

inline bool ObjectMap2D::isSatisfyObject(const Eigen::Vector2d& pos)
{
  if (sdf_map_->isInMap(pos) && sdf_map_->getOccupancy(pos) == SDFMap2D::OCCUPIED)
    return true;
  return false;
}

inline bool ObjectMap2D::isObjectClustered(const int& adr)
{
  if (object_indexs_[adr] == -1)
    return false;
  return true;
}

inline bool ObjectMap2D::isObjectClustered(const Eigen::Vector2i& idx)
{
  return isObjectClustered(toAdr(idx));
}

inline bool ObjectMap2D::isObjectClustered(const Eigen::Vector2d& pos)
{
  Eigen::Vector2i idx;
  sdf_map_->posToIndex(pos, idx);
  return isObjectClustered(idx);
}

inline bool ObjectMap2D::haveOverlap(
    const Vector2d& min1, const Vector2d& max1, const Vector2d& min2, const Vector2d& max2)
{
  // Check for separation along each axis
  Vector2d bmin, bmax;
  for (int i = 0; i < 2; ++i) {
    bmin[i] = max(min1[i], min2[i]);
    bmax[i] = min(max1[i], max2[i]);
    if (bmin[i] > bmax[i] + 1e-3)
      return false;
  }
  return true;
}

inline void ObjectMap2D::wrapYaw(double& yaw)
{
  while (yaw < -M_PI) yaw += 2 * M_PI;
  while (yaw > M_PI) yaw -= 2 * M_PI;
}

inline vector<Eigen::Vector2i> ObjectMap2D::fourNeighbors(const Eigen::Vector2i& idx)
{
  vector<Eigen::Vector2i> neighbors(4);
  neighbors[0] = idx + Eigen::Vector2i(-1, 0);
  neighbors[1] = idx + Eigen::Vector2i(1, 0);
  neighbors[2] = idx + Eigen::Vector2i(0, -1);
  neighbors[3] = idx + Eigen::Vector2i(0, 1);
  return neighbors;
}

inline vector<Eigen::Vector2i> ObjectMap2D::allNeighbors(const Eigen::Vector2i& idx)
{
  vector<Eigen::Vector2i> neighbors(8);
  int count = 0;
  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      if (x == 0 && y == 0)
        continue;  // Skip center cell
      neighbors[count++] = idx + Eigen::Vector2i(x, y);
    }
  }
  return neighbors;
}

inline vector<Eigen::Vector2i> ObjectMap2D::allGridsDistance(
    const Eigen::Vector2i& idx, const double& dist)
{
  vector<Eigen::Vector2i> grids;
  int cnt = ceil(dist / resolution_);  // Convert distance to grid cells

  for (int x = -cnt; x <= cnt; ++x) {
    for (int y = -cnt; y <= cnt; ++y) {
      if (x == 0 && y == 0)
        continue;  // Skip center cell

      // Check if grid point is within circular distance
      Vector2d step_dist = Vector2d(x * resolution_, y * resolution_);
      if (step_dist.norm() <= dist) {
        Eigen::Vector2i grid_idx = idx + Eigen::Vector2i(x, y);
        grids.push_back(grid_idx);
      }
    }
  }
  return grids;
}

inline bool ObjectMap2D::isNeighborUnknown(const Eigen::Vector2i& idx)
{
  auto nbrs = fourNeighbors(idx);
  for (auto nbr : nbrs) {
    if (sdf_map_->getOccupancy(nbr) == SDFMap2D::UNKNOWN)
      return true;
  }
  return false;
}

inline int ObjectMap2D::toAdr(const Eigen::Vector2d& pos)
{
  Eigen::Vector2i idx;
  sdf_map_->posToIndex(pos, idx);
  return sdf_map_->toAddress(idx);
}

inline int ObjectMap2D::toAdr(const Eigen::Vector2i& idx)
{
  return sdf_map_->toAddress(idx);
}

inline bool ObjectMap2D::knownFree(const Eigen::Vector2i& idx)
{
  return sdf_map_->getOccupancy(idx) == SDFMap2D::FREE;
}

inline bool ObjectMap2D::inMap(const Eigen::Vector2i& idx)
{
  return sdf_map_->isInMap(idx);
}

inline int ObjectMap2D::getObjectGrid(const int& adr) const
{
  return int(object_buffer_[adr]);
}

inline int ObjectMap2D::getObjectGrid(const Eigen::Vector2i& id) const
{
  if (!sdf_map_->isInMap(id))
    return -1;  // Invalid position
  return int(getObjectGrid(sdf_map_->toAddress(id)));
}

inline int ObjectMap2D::getObjectGrid(const Eigen::Vector2d& pos) const
{
  Eigen::Vector2i id;
  sdf_map_->posToIndex(pos, id);
  return getObjectGrid(id);
}

inline Eigen::Vector4d ObjectMap2D::getColor(const double& h, double alpha)
{
  double h1 = h;
  if (h1 < 0.0 || h1 > 1.0) {
    std::cout << "h out of range" << std::endl;
    h1 = 0.0;
  }

  double lambda;
  Eigen::Vector4d color1, color2;

  // HSV color wheel interpolation with 6 segments
  if (h1 >= -1e-4 && h1 < 1.0 / 6) {
    lambda = (h1 - 0.0) * 6;
    color1 = Eigen::Vector4d(1, 0, 0, 1);
    color2 = Eigen::Vector4d(1, 0, 1, 1);
  }
  else if (h1 >= 1.0 / 6 && h1 < 2.0 / 6) {
    lambda = (h1 - 1.0 / 6) * 6;
    color1 = Eigen::Vector4d(1, 0, 1, 1);
    color2 = Eigen::Vector4d(0, 0, 1, 1);
  }
  else if (h1 >= 2.0 / 6 && h1 < 3.0 / 6) {
    lambda = (h1 - 2.0 / 6) * 6;
    color1 = Eigen::Vector4d(0, 0, 1, 1);
    color2 = Eigen::Vector4d(0, 1, 1, 1);
  }
  else if (h1 >= 3.0 / 6 && h1 < 4.0 / 6) {
    lambda = (h1 - 3.0 / 6) * 6;
    color1 = Eigen::Vector4d(0, 1, 1, 1);
    color2 = Eigen::Vector4d(0, 1, 0, 1);
  }
  else if (h1 >= 4.0 / 6 && h1 < 5.0 / 6) {
    lambda = (h1 - 4.0 / 6) * 6;
    color1 = Eigen::Vector4d(0, 1, 0, 1);
    color2 = Eigen::Vector4d(1, 1, 0, 1);
  }
  else if (h1 >= 5.0 / 6 && h1 <= 1.0 + 1e-4) {
    lambda = (h1 - 5.0 / 6) * 6;
    color1 = Eigen::Vector4d(1, 1, 0, 1);
    color2 = Eigen::Vector4d(1, 0, 0, 1);
  }

  Eigen::Vector4d fcolor = (1 - lambda) * color1 + lambda * color2;
  fcolor(3) = alpha;

  return fcolor;
}

inline void ObjectMap2D::publishObjectClouds()
{
  // Create colored point cloud container
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined_colored_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);

  // Process each detected object
  for (const auto& object : objects_) {
    if (object.best_label_ == -1)
      continue;  // Skip objects without valid classification

    // Get the best-confidence point cloud for this object
    const auto& cloud = object.clouds_[object.best_label_];

    // Color and transform each point
    for (const auto& point : cloud->points) {
      pcl::PointXYZRGB colored_point;
      colored_point.x = point.x;
      colored_point.y = point.y;
      colored_point.z = point.z + 1.0;  // Elevate for better visibility

      // Generate class-specific color with soft blending
      Eigen::Vector4d col = getColor(object.best_label_ / 5.0, 1.0);
      double blend_factor = 0.5;  // Softening factor for pastel colors
      uint8_t r = col(0) * 255 * blend_factor + 255 * (1.0 - blend_factor);
      uint8_t g = col(1) * 255 * blend_factor + 255 * (1.0 - blend_factor);
      uint8_t b = col(2) * 255 * blend_factor + 255 * (1.0 - blend_factor);
      colored_point.r = r;
      colored_point.g = g;
      colored_point.b = b;

      combined_colored_cloud->points.push_back(colored_point);
    }
  }

  if (combined_colored_cloud->points.empty())
    return;  // No objects to visualize

  // Configure point cloud metadata
  combined_colored_cloud->width = combined_colored_cloud->points.size();
  combined_colored_cloud->height = 1;
  combined_colored_cloud->is_dense = true;

  // Publish the combined visualization
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(*combined_colored_cloud, output);
  output.header.frame_id = "world";
  output.header.stamp = ros::Time::now();
  object_cloud_pub_.publish(output);
}
}  // namespace apexnav_planner

#endif
