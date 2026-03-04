/**
 * @file object_map2d.cpp
 * @brief Implementation of semantic object mapping system for autonomous navigation
 *
 * This file contains the complete implementation of the ObjectMap2D class,
 * providing semantic object detection, multi-view fusion, confidence scoring,
 * and 3D point cloud processing capabilities. The system integrates vision-based
 * object detection with occupancy mapping to create robust semantic representations.
 *
 * @author Zhiyang
 */

#include <plan_env/object_map2d.h>
#include <algorithm>  // for std::transform, std::min
#include <iomanip>    // for std::setprecision
#include <sstream>    // for std::stringstream

namespace apexnav_planner {
ObjectMap2D::ObjectMap2D(SDFMap2D* sdf_map, ros::NodeHandle& nh)
{
  // Initialize core mapping components
  this->sdf_map_ = sdf_map;
  int voxel_num = sdf_map_->getVoxelNum();
  object_buffer_ = vector<char>(voxel_num, 0);  // Object occupancy flags per grid cell
  object_indexs_ = vector<int>(voxel_num, -1);  // Object ID mapping per grid cell

  // Initialize point cloud containers
  all_object_clouds_.reset(new pcl::PointCloud<pcl::PointXYZ>());
  over_depth_object_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());

  // Store NodeHandle for parameter reloading on episode reset
  nh_ = nh;

  // Load configuration parameters
  min_confidence_ = -1.0;  // Default to accept all detections
  nh.param("object/min_observation_num", min_observation_num_, 2);
  nh.param("object/fusion_type", fusion_type_, 1);
  nh.param("object/use_observation", use_observation_, true);
  nh.param("object/vis_cloud", is_vis_cloud_, false);

  // ====== Patience-Aware Navigation Parameters (Paper Section 3.5) ======
  nh.param("object/tau_high", tau_high_, 0.4);
  nh.param("object/tau_low", tau_low_, 0.2);
  nh.param("object/T_max", T_max_, 500);
  nh.param("object/min_detection_confidence", min_detection_confidence_, 0.1);
  current_step_ = 0;
  current_target_category_ = "";

  // Initialize VLM validation state
  current_target_object_id_ = -1;

  // VLM confidence fusion parameters
  // Use moderate penalty to avoid over-penalizing due to viewpoint issues
  nh.param("object/vlm_positive_weight", vlm_positive_weight_, 0.7);
  nh.param("object/vlm_positive_boost", vlm_positive_boost_, 0.85);
  nh.param("object/vlm_negative_weight", vlm_negative_weight_, 0.5);   // Reduced from 0.8 to be more conservative
  nh.param("object/vlm_negative_penalty", vlm_negative_penalty_, 0.15); // Raised from 0.05 to be less extreme

  // Initialize target-specific thresholds
  initTargetSpecificThresholds(nh);

  ROS_INFO("[ObjectMap2D] Patience-Aware Navigation enabled:");
  ROS_INFO("  - tau_high: %.2f", tau_high_);
  ROS_INFO("  - tau_low: %.2f", tau_low_);
  ROS_INFO("  - T_max: %d steps", T_max_);
  ROS_INFO("  - min_detection_confidence: %.2f", min_detection_confidence_);

  // Setup ROS communication
  object_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/object/clouds", 10);

  // Subscribe to SemanticScores for target category (Patience-Aware Navigation)
  semantic_scores_sub_ = nh.subscribe<plan_env::SemanticScores>(
      "/habitat/semantic_scores", 1, &ObjectMap2D::semanticScoresCallback, this);

  // Configure raycasting for spatial queries
  raycaster_.reset(new RayCaster2D);
  resolution_ = sdf_map_->getResolution();
  Eigen::Vector2d origin, size;
  sdf_map_->getRegion(origin, size);
  raycaster_->setParams(resolution_, origin);

  // Set point cloud processing parameters
  leaf_size_ = 0.1f;  // Voxel grid leaf size for downsampling
}

void ObjectMap2D::setConfidenceThreshold(double val)
{
  min_confidence_ = val;
  ROS_INFO("Set Confidence Threshold = %f", val);
}

void ObjectMap2D::clearObjects()
{
  // Record how many objects we're clearing for logging
  size_t num_objects = objects_.size();

  // Clear all object clusters
  objects_.clear();

  // Reset object index buffers to -1 (no object)
  std::fill(object_indexs_.begin(), object_indexs_.end(), -1);

  // Reset object occupancy buffer to 0 (not occupied by object)
  std::fill(object_buffer_.begin(), object_buffer_.end(), 0);

  // Clear point cloud containers and force memory release
  size_t all_points = 0, over_depth_points = 0;
  if (all_object_clouds_) {
    all_points = all_object_clouds_->size();
    all_object_clouds_->clear();
    all_object_clouds_->points.shrink_to_fit();  // Force memory release
    all_object_clouds_->width = 0;
    all_object_clouds_->height = 0;
  }
  if (over_depth_object_cloud_) {
    over_depth_points = over_depth_object_cloud_->size();
    over_depth_object_cloud_->clear();
    over_depth_object_cloud_->points.shrink_to_fit();  // Force memory release
    over_depth_object_cloud_->width = 0;
    over_depth_object_cloud_->height = 0;
  }

  // Reset Patience-Aware Navigation state for new episode
  current_step_ = 0;
  current_target_category_ = "";

  // Reset VLM validation state
  invalid_object_ids_.clear();
  current_target_object_id_ = -1;

  // Clear failed approach points and unreachable objects for new episode
  failed_approach_points_.clear();
  unreachable_objects_.clear();

  // ====== Reload threshold parameters from ROS parameter server ======
  // This ensures Python-side config changes (from habitat_evaluation.py) take effect
  // Python publishes patience_nav/threshold_by_size to /object/threshold/* params
  nh_.param("object/tau_high", tau_high_, tau_high_);
  nh_.param("object/tau_low", tau_low_, tau_low_);
  nh_.param("object/T_max", T_max_, T_max_);
  nh_.param("object/min_detection_confidence", min_detection_confidence_, min_detection_confidence_);

  // Reload target-specific thresholds
  initTargetSpecificThresholds(nh_);

  ROS_INFO("[ObjectMap2D] Cleared %zu objects + %zu point cloud points for new episode",
           num_objects, all_points + over_depth_points);
  ROS_INFO("[ObjectMap2D] Reloaded thresholds: tau_high=%.2f, tau_low=%.2f, T_max=%d",
           tau_high_, tau_low_, T_max_);
}

// ==================== VLM Validation Methods ====================

void ObjectMap2D::markObjectAsInvalid(int object_id)
{
  invalid_object_ids_.insert(object_id);

  // If the invalid object is the current target, clear the target to force replanning
  if (current_target_object_id_ == object_id) {
    ROS_WARN("[ObjectMap2D] Clearing current target (was object id=%d)", object_id);
    current_target_object_id_ = -1;
  }

  ROS_WARN("[ObjectMap2D] Marked object id=%d as invalid (VLM validation failed)", object_id);
}

bool ObjectMap2D::isObjectInvalid(int object_id) const
{
  return invalid_object_ids_.find(object_id) != invalid_object_ids_.end();
}

void ObjectMap2D::clearInvalidObjects()
{
  size_t count = invalid_object_ids_.size();
  invalid_object_ids_.clear();
  current_target_object_id_ = -1;
  ROS_INFO("[ObjectMap2D] Cleared %zu invalid object markers", count);
}

// ==================== VLM Verification Methods ====================

void ObjectMap2D::applyVLMVerificationResult(int object_id, int decision_level, double vlm_confidence)
{
  if (object_id < 0 || object_id >= static_cast<int>(objects_.size())) {
    ROS_ERROR("[VLM] Invalid object_id %d for VLM result application", object_id);
    return;
  }

  ObjectCluster& obj = objects_[object_id];
  double old_confidence = obj.fused_confidence_;

  // 3-level decision handling:
  // decision_level = 1: CONFIRM (boost confidence)
  // decision_level = 0: UNCERTAIN (keep original confidence)
  // decision_level = -1: REJECT (reduce confidence)

  if (decision_level == 1) {
    // VLM confirmed as real target -> significantly boost confidence
    double new_confidence = (1.0 - vlm_positive_weight_) * old_confidence +
                            vlm_positive_weight_ * vlm_positive_boost_;
    obj.fused_confidence_ = new_confidence;
    obj.confidence_scores_[0] = new_confidence;

    ROS_INFO("[VLM] Object %d CONFIRMED as real target! Confidence: %.3f -> %.3f",
             object_id, old_confidence, new_confidence);

    // Mark as verified with positive result
    obj.vlm_verified_ = true;
    obj.vlm_result_ = true;

  } else if (decision_level == -1) {
    // VLM rejected as false positive -> moderately reduce confidence
    // Do NOT mark as invalid - let confidence threshold decide selection
    // This allows recovery if VLM made a mistake due to viewpoint issues
    double new_confidence = (1.0 - vlm_negative_weight_) * old_confidence +
                            vlm_negative_weight_ * vlm_negative_penalty_;
    obj.fused_confidence_ = new_confidence;
    obj.confidence_scores_[0] = new_confidence;

    // NOTE: Removed markObjectAsInvalid() - selection now based purely on confidence
    // Object can still be selected if confidence recovers above threshold

    ROS_WARN("[VLM] Object %d REJECTED: Confidence: %.3f -> %.3f (selection based on confidence threshold)",
             object_id, old_confidence, new_confidence);

    // Mark as verified with negative result
    obj.vlm_verified_ = true;
    obj.vlm_result_ = false;

  } else {
    // UNCERTAIN (decision_level == 0) -> keep original confidence
    // Mark as verified to prevent infinite re-verification loops
    ROS_INFO("[VLM] Object %d UNCERTAIN - keeping original confidence: %.3f (VLM conf: %.2f)",
             object_id, old_confidence, vlm_confidence);

    // Mark as verified to prevent re-triggering, but result is false (not confirmed)
    obj.vlm_verified_ = true;   // Prevent infinite re-verification
    obj.vlm_result_ = false;    // But not confirmed as target
    obj.vlm_uncertain_ = true;  // New flag to track UNCERTAIN status
  }

  obj.vlm_pending_ = false;
  obj.vlm_adjusted_confidence_ = obj.fused_confidence_;
}

bool ObjectMap2D::isObjectVLMVerified(int object_id) const
{
  if (object_id < 0 || object_id >= static_cast<int>(objects_.size())) {
    return false;
  }
  return objects_[object_id].vlm_verified_;
}

bool ObjectMap2D::getObjectVLMResult(int object_id) const
{
  if (object_id < 0 || object_id >= static_cast<int>(objects_.size())) {
    return false;
  }
  return objects_[object_id].vlm_result_;
}

bool ObjectMap2D::isObjectVLMUncertain(int object_id) const
{
  if (object_id < 0 || object_id >= static_cast<int>(objects_.size())) {
    return false;
  }
  return objects_[object_id].vlm_uncertain_;
}

void ObjectMap2D::setObjectVLMPending(int object_id, bool pending)
{
  if (object_id < 0 || object_id >= static_cast<int>(objects_.size())) {
    ROS_ERROR("[VLM] Invalid object_id %d for setting VLM pending", object_id);
    return;
  }
  objects_[object_id].vlm_pending_ = pending;
  if (pending) {
    ROS_INFO("[VLM] Object %d VLM verification pending", object_id);
  }
}

bool ObjectMap2D::isObjectVLMPending(int object_id) const
{
  if (object_id < 0 || object_id >= static_cast<int>(objects_.size())) {
    return false;
  }
  return objects_[object_id].vlm_pending_;
}

// ==================== Failed Approach Point Management ====================

void ObjectMap2D::addFailedApproachPoint(int object_id, const Eigen::Vector2d& point)
{
  failed_approach_points_[object_id].push_back(point);
  ROS_WARN("[ObjectMap2D] Added failed approach point (%.2f, %.2f) for object %d. Total failed: %zu",
           point.x(), point.y(), object_id, failed_approach_points_[object_id].size());
}

bool ObjectMap2D::isApproachPointFailed(int object_id, const Eigen::Vector2d& point, double threshold) const
{
  auto it = failed_approach_points_.find(object_id);
  if (it == failed_approach_points_.end()) {
    return false;
  }

  for (const auto& failed_point : it->second) {
    if ((failed_point - point).norm() < threshold) {
      return true;
    }
  }
  return false;
}

void ObjectMap2D::clearFailedApproachPoints(int object_id)
{
  if (object_id < 0) {
    // Clear all
    size_t total = 0;
    for (const auto& pair : failed_approach_points_) {
      total += pair.second.size();
    }
    failed_approach_points_.clear();
    ROS_INFO("[ObjectMap2D] Cleared all %zu failed approach points", total);
  } else {
    auto it = failed_approach_points_.find(object_id);
    if (it != failed_approach_points_.end()) {
      size_t count = it->second.size();
      failed_approach_points_.erase(it);
      ROS_INFO("[ObjectMap2D] Cleared %zu failed approach points for object %d", count, object_id);
    }
  }
}

int ObjectMap2D::getFailedApproachPointCount(int object_id) const
{
  auto it = failed_approach_points_.find(object_id);
  if (it == failed_approach_points_.end()) {
    return 0;
  }
  return static_cast<int>(it->second.size());
}

void ObjectMap2D::markObjectUnreachable(int object_id)
{
  unreachable_objects_.insert(object_id);
  ROS_WARN("[ObjectMap2D] Marked object %d as UNREACHABLE (failed approach count: %d)",
           object_id, getFailedApproachPointCount(object_id));
}

bool ObjectMap2D::isObjectUnreachable(int object_id) const
{
  return unreachable_objects_.find(object_id) != unreachable_objects_.end();
}

void ObjectMap2D::clearUnreachableStatus(int object_id)
{
  if (object_id < 0) {
    size_t count = unreachable_objects_.size();
    unreachable_objects_.clear();
    ROS_INFO("[ObjectMap2D] Cleared all %zu unreachable object markers", count);
  } else {
    if (unreachable_objects_.erase(object_id) > 0) {
      ROS_INFO("[ObjectMap2D] Cleared unreachable status for object %d", object_id);
    }
  }
}

bool ObjectMap2D::getFirstDetectionPosition(int object_id, Vector2d& detection_pos) const
{
  if (object_id < 0 || object_id >= static_cast<int>(objects_.size())) {
    return false;
  }
  const auto& obj = objects_[object_id];
  if (!obj.has_detection_pos_) {
    return false;
  }
  detection_pos = obj.first_detection_pos_;
  return true;
}

void ObjectMap2D::getTopConfidenceObjectIds(std::vector<int>& object_ids, bool limited_confidence)
{
  object_ids.clear();
  double tau_t = getCurrentConfidenceThreshold();

  // Collect objects that pass filtering (same logic as getTopConfidenceObjectCloud)
  std::vector<std::pair<int, double>> id_confidence_pairs;

  for (const auto& object : objects_) {
    // Skip objects that failed VLM validation
    if (isObjectInvalid(object.id_)) {
      continue;
    }

    if (!limited_confidence) {
      // Include all non-invalid objects
      id_confidence_pairs.push_back({object.id_, object.confidence_scores_[0]});
    } else {
      // Apply confidence filtering with two-tier strategy
      int max_func_score = 0, best_label = -1;

      for (int label = 0; label < (int)object.clouds_.size(); label++) {
        auto obs_sum = object.observation_cloud_sums_[label];
        auto score = object.confidence_scores_[label];
        int func_score = obs_sum * score;
        if (func_score > max_func_score) {
          max_func_score = func_score;
          best_label = label;
        }
      }

      bool shouldInclude = false;

      // Tier 1: best_label is target (0) and meets confidence threshold
      if (best_label == 0 && isConfidenceObject(object)) {
        shouldInclude = true;
      }
      // Tier 2: best_label is NOT target, but conf[0] is significantly high (1.2x threshold)
      else if (object.confidence_scores_[0] >= 1.2 * tau_t &&
               object.observation_nums_[0] >= min_observation_num_) {
        shouldInclude = true;
      }

      if (shouldInclude) {
        id_confidence_pairs.push_back({object.id_, object.confidence_scores_[0]});
      }
    }
  }

  // Sort by confidence (descending) - same order as getTopConfidenceObjectCloud
  std::sort(id_confidence_pairs.begin(), id_confidence_pairs.end(),
            [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
              return a.second > b.second;
            });

  // Extract IDs
  for (const auto& pair : id_confidence_pairs) {
    object_ids.push_back(pair.first);
  }
}

/**
 * @brief Process observation clouds to adjust detection confidence
 *
 * This function handles negative evidence from visual observations where
 * objects were expected but not detected. It computes spatial overlap
 * between observation regions and existing detections to reduce confidence
 * scores, improving the robustness of the semantic mapping system.
 *
 * @param observation_clouds Vector of point clouds representing observed regions
 * @param itm_score Image-text matching score for context weighting
 */
void ObjectMap2D::inputObservationObjectsCloud(
    const vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> observation_clouds,
    const double& itm_score)
{
  // Only process observations in fusion mode with observation enabled
  if (fusion_type_ != 1 || !use_observation_)
    return;

  // Process each observation cloud against corresponding objects
  for (int i = 0; i < (int)observation_clouds.size(); i++) {
    auto observation_cloud = observation_clouds[i];
    auto object = objects_[i];

    if (observation_cloud->points.empty())
      continue;

    // Check overlap with each possible object classification
    for (int label = 0; label < 5; ++label) {
      if (object.confidence_scores_[label] < 1e-3)
        continue;  // Skip labels with negligible confidence

      // Setup spatial search for overlap computation
      pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
      kdtree.setInputCloud(observation_cloud);
      double distance_threshold = leaf_size_ * 1.1;  // Spatial overlap threshold
      int overlap_count = 0;

      // Count overlapping points between object and observation clouds
      for (const auto& point : object.clouds_[label]->points) {
        std::vector<int> point_idx_search;
        std::vector<float> point_squared_distance;
        if (kdtree.nearestKSearch(point, 1, point_idx_search, point_squared_distance) > 0) {
          // Points within threshold are considered overlapping
          if (point_squared_distance[0] <= distance_threshold * distance_threshold) {
            overlap_count++;
          }
        }
      }

      // Skip if no spatial overlap detected
      if (overlap_count == 0)
        continue;

      // Update confidence scores based on negative observation evidence
      auto& merged_object = objects_[i];
      merged_object.observation_cloud_sums_[label] += overlap_count;
      int total_last = merged_object.clouds_[label]->points.size();
      double confidence_last = merged_object.confidence_scores_[label];
      int observation_now = overlap_count;
      double confidence_now = 0.0;  // Negative evidence has zero confidence
      if (label == 0)
        confidence_now = itm_score;  // Use ITM score for primary label
      int total_now = merged_object.clouds_[label]->points.size();

      // Apply confidence fusion algorithm
      merged_object.confidence_scores_[label] = fusionConfidenceScore(total_last, confidence_last,
          observation_now, confidence_now, total_now, merged_object.observation_cloud_sums_[label]);
      printFusionInfo(merged_object, label, "[Observation]");
      // ROS_WARN("[Observation] id = %d label = %d overlap_count = %d object_cloud = %ld",
      //     merged_object.id_, label, overlap_count, object.clouds_[label]->points.size());
    }
  }
}

int ObjectMap2D::searchSingleObjectCluster(const DetectedObject& detected_object)
{
  auto object_cloud = detected_object.cloud;

  // Initialize clustering analysis variables
  int point_num = object_cloud->points.size();
  int obj_idx = -1;
  vector<Eigen::Vector2d> object_point2Ds;
  vector<char> flag_2d(sdf_map_->getVoxelNum(), 0);  // Duplicate point prevention

  // Process each point in the detected object cloud
  for (int i = 0; i < point_num; i++) {
    Eigen::Vector2i idx;
    Eigen::Vector2d pt_w;
    pt_w << object_cloud->points[i].x, object_cloud->points[i].y;
    sdf_map_->posToIndex(pt_w, idx);
    int adr = sdf_map_->toAddress(idx);

    // Skip duplicate grid cells
    if (flag_2d[adr] == 1)
      continue;

    flag_2d[adr] = 1;

    // Validate if point satisfies object characteristics
    if (isSatisfyObject(pt_w)) {
      object_buffer_[adr] = 1;  // Mark cell as containing object
      object_point2Ds.push_back(pt_w);
    }
  }

  // Return early if no valid object points found
  if (object_point2Ds.empty()) {
    return -1;
  }

  // Search for existing object clusters in neighborhood
  for (auto pt_w : object_point2Ds) {
    Eigen::Vector2i idx;
    sdf_map_->posToIndex(pt_w, idx);

    // Get neighboring grid cells within clustering radius
    auto nbrs = allGridsDistance(idx, 0.08);
    nbrs.push_back(idx);

    // Check neighbors for existing object associations
    for (auto nbr : nbrs) {
      int nbr_adr = sdf_map_->toAddress(nbr);
      if (object_indexs_[nbr_adr] != -1) {
        // Found existing object cluster - use first match
        // TODO: Implement multi-object merging for complex scenarios
        obj_idx = object_indexs_[nbr_adr];
        break;
      }
    }
  }

  // Apply voxel grid filtering to reduce point cloud density
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  voxel_filter.setInputCloud(object_cloud);
  voxel_filter.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
  voxel_filter.filter(*detected_object.cloud);

  // Either merge with existing cluster or create new one
  if (obj_idx != -1) {
    mergeCellsIntoObjectCluster(obj_idx, object_point2Ds, detected_object);
  }
  else {
    createNewObjectCluster(object_point2Ds, detected_object);
    obj_idx = object_indexs_[toAdr(object_point2Ds[0])];
  }

  // Update classification and visualization
  updateObjectBestLabel(obj_idx);

  // ====== Update fused_confidence for Patience-Aware Navigation ======
  if (obj_idx != -1 && obj_idx < static_cast<int>(objects_.size())) {
    auto& obj = objects_[obj_idx];
    // Use the best label's confidence as fused_confidence
    if (obj.best_label_ >= 0 && obj.best_label_ < static_cast<int>(obj.confidence_scores_.size())) {
      obj.fused_confidence_ = obj.confidence_scores_[obj.best_label_];
    }
  }

  if (is_vis_cloud_)
    publishObjectClouds();

  // Validate successful clustering
  if (obj_idx == -1) {
    ROS_ERROR("[bug] Why not find the object cluster!?");
    return obj_idx;
  }

  return obj_idx;
}

void ObjectMap2D::updateObjectBestLabel(int obj_idx)
{
  double max_score = 0.1;  // Minimum threshold for valid classification
  int best_label = -1;

  // Evaluate each possible classification label using only fused confidence
  // Note: confidence_scores_ already incorporates observation history through
  // fusionConfidenceScore(), so no need to multiply by observation count again
  for (int label = 0; label < (int)objects_[obj_idx].clouds_.size(); label++) {
    auto score = objects_[obj_idx].confidence_scores_[label];

    if (score > max_score) {
      max_score = score;
      best_label = label;
    }
  }
  objects_[obj_idx].best_label_ = best_label;
}

void ObjectMap2D::createNewObjectCluster(
    const std::vector<Eigen::Vector2d>& cells, const DetectedObject& detected_object)
{
  int label = detected_object.label;

  // Initialize new object cluster with unique ID
  ObjectCluster obj;
  obj.id_ = (int)objects_.size();
  obj.max_seen_count_ = 0;
  obj.good_cells_.clear();
  obj.seen_counts_.clear();
  obj.best_label_ = -1;

  // Process spatial cells and establish grid associations
  std::vector<Eigen::Vector2d> real_new_cells;
  for (auto cell : cells) {
    int adr = toAdr(cell);
    object_indexs_[adr] = obj.id_;  // Associate grid cell with object
    obj.visited_[adr] = 1;

    // Track high-confidence observations for label 0
    if (label == 0) {
      obj.seen_counts_[adr] = 1;
      obj.max_seen_count_ = 1;
      obj.good_cells_.push_back(cell);
    }
  }

  // Compute spatial properties of the object cluster
  obj.cells_ = cells;
  obj.average_.setZero();
  obj.box_max2d_ = obj.cells_.front();
  obj.box_min2d_ = obj.cells_.front();

  for (auto cell : obj.cells_) {
    obj.average_ += cell;
    for (int i = 0; i < 2; ++i) {
      obj.box_min2d_[i] = min(obj.box_min2d_[i], cell[i]);
      obj.box_max2d_[i] = max(obj.box_max2d_[i], cell[i]);
    }
  }
  obj.average_ /= double(obj.cells_.size());

  // Initialize point cloud storage for the detected label
  obj.clouds_[label].reset(new pcl::PointCloud<pcl::PointXYZ>());
  *obj.clouds_[label] = *detected_object.cloud;

  // Compute 3D bounding box from point cloud
  obj.box_max3d_ = Vector3d(obj.clouds_[label]->points[0].x, obj.clouds_[label]->points[0].y,
      obj.clouds_[label]->points[0].z);
  obj.box_min3d_ = Vector3d(obj.clouds_[label]->points[0].x, obj.clouds_[label]->points[0].y,
      obj.clouds_[label]->points[0].z);

  for (auto pt : obj.clouds_[label]->points) {
    Vector3d vec_pt = Vector3d(pt.x, pt.y, pt.z);
    for (int i = 0; i < 3; ++i) {
      obj.box_min3d_[i] = min(obj.box_min3d_[i], vec_pt[i]);
      obj.box_max3d_[i] = max(obj.box_max3d_[i], vec_pt[i]);
    }
  }

  // Initialize confidence tracking for this object
  obj.confidence_scores_[label] = detected_object.score;
  obj.observation_cloud_sums_[label] = detected_object.cloud->points.size();
  obj.observation_nums_[label] = 1;

  // Record first detection position for direction-aware path planning
  obj.first_detection_pos_ = current_robot_pos_;
  obj.has_detection_pos_ = true;

  // Add to global object registry
  objects_.push_back(obj);
  ROS_INFO("[New Object] id=%d, first_detection_pos=(%.2f, %.2f)",
           obj.id_, obj.first_detection_pos_.x(), obj.first_detection_pos_.y());
  printFusionInfo(obj, label, "[New Object Cluster]");
}

void ObjectMap2D::mergeCellsIntoObjectCluster(const int& merged_object_id,
    const std::vector<Eigen::Vector2d>& new_cells, const DetectedObject& detected_object)
{
  int label = detected_object.label;
  const auto last_objects = objects_;

  ObjectCluster& merged_object = objects_[merged_object_id];
  std::vector<Eigen::Vector2d> real_new_cells;

  // Process new spatial cells for integration
  for (auto new_cell : new_cells) {
    int adr = toAdr(new_cell);
    object_indexs_[adr] = merged_object_id;  // Associate with this object cluster

    // Add only genuinely new cells to avoid duplicates
    if (!merged_object.visited_.count(adr)) {
      real_new_cells.push_back(new_cell);
      merged_object.visited_[adr] = 1;
    }

    // Track observation frequency for high-confidence detections (label 0)
    if (label == 0) {
      if (merged_object.seen_counts_.count(adr))
        merged_object.seen_counts_[adr] += 1;
      else
        merged_object.seen_counts_[adr] = 1;

      // Update maximum observation count for this cluster
      if (merged_object.seen_counts_[adr] > merged_object.max_seen_count_)
        merged_object.max_seen_count_ = merged_object.seen_counts_[adr];
    }
  }

  // Extend cluster's spatial coverage
  merged_object.cells_.insert(
      merged_object.cells_.end(), real_new_cells.begin(), real_new_cells.end());

  // Update high-confidence cell tracking for label 0
  if (label == 0) {
    merged_object.good_cells_.clear();
    for (auto cell : merged_object.cells_) {
      int adr = toAdr(cell);
      if (merged_object.seen_counts_.count(adr)) {
        // Cells with sufficient observations are considered "good"
        if (merged_object.seen_counts_[adr] >= min(4, merged_object.max_seen_count_))
          merged_object.good_cells_.push_back(cell);
      }
    }
    ROS_ERROR("merged_object good cells size = %ld", merged_object.good_cells_.size());
  }

  // Recompute spatial properties
  merged_object.average_.setZero();
  merged_object.box_max2d_ = merged_object.cells_.front();
  merged_object.box_min2d_ = merged_object.cells_.front();
  for (auto cell : merged_object.cells_) {
    merged_object.average_ += cell;
    for (int i = 0; i < 2; ++i) {
      merged_object.box_min2d_[i] = min(merged_object.box_min2d_[i], cell[i]);
      merged_object.box_max2d_[i] = max(merged_object.box_max2d_[i], cell[i]);
    }
  }
  merged_object.average_ /= double(merged_object.cells_.size());

  // Handle point cloud fusion based on observation history
  if (!merged_object.observation_nums_[label]) {
    // First observation of this label - initialize directly
    merged_object.clouds_[label].reset(new pcl::PointCloud<pcl::PointXYZ>());
    *merged_object.clouds_[label] = *(detected_object.cloud);
    merged_object.confidence_scores_[label] = detected_object.score;
    merged_object.observation_cloud_sums_[label] = detected_object.cloud->points.size();
    merged_object.observation_nums_[label] = 1;
    printFusionInfo(merged_object, label, "[New Label Merged]");
  }
  else {
    // Merge with existing observations using point cloud fusion
    pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    *merged_cloud = *(merged_object.clouds_[label]);  // Copy existing cloud
    *merged_cloud += *(detected_object.cloud);        // Add new observations

    // Apply voxel grid downsampling to manage point cloud size
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(merged_cloud);
    voxel_filter.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
    voxel_filter.filter(*merged_cloud);
    merged_object.clouds_[label] = merged_cloud;

    // Update 3D bounding box from merged point cloud
    merged_object.box_max3d_ = Vector3d(merged_object.clouds_[label]->points[0].x,
        merged_object.clouds_[label]->points[0].y, merged_object.clouds_[label]->points[0].z);
    merged_object.box_min3d_ = Vector3d(merged_object.clouds_[label]->points[0].x,
        merged_object.clouds_[label]->points[0].y, merged_object.clouds_[label]->points[0].z);

    for (auto pt : merged_object.clouds_[label]->points) {
      Vector3d vec_pt = Vector3d(pt.x, pt.y, pt.z);
      for (int i = 0; i < 3; ++i) {
        merged_object.box_min3d_[i] = min(merged_object.box_min3d_[i], vec_pt[i]);
        merged_object.box_max3d_[i] = max(merged_object.box_max3d_[i], vec_pt[i]);
      }
    }

    // Update observation tracking
    merged_object.observation_nums_[label]++;
    merged_object.observation_cloud_sums_[label] += detected_object.cloud->points.size();

    // Prepare confidence fusion parameters
    int last_total = last_objects[merged_object_id].clouds_[label]->points.size();
    double last_total_confidence = last_objects[merged_object_id].confidence_scores_[label];
    int now_observation = detected_object.cloud->points.size();
    double now_confidence = detected_object.score;
    int now_total = merged_object.clouds_[label]->points.size();

    // Apply confidence fusion strategy based on fusion type
    if (fusion_type_ == 0)
      merged_object.confidence_scores_[label] = now_confidence;  // Replace with current
    else if (fusion_type_ == 1)
      merged_object.confidence_scores_[label] =
          fusionConfidenceScore(last_total, last_total_confidence, now_observation, now_confidence,
              now_total, merged_object.observation_cloud_sums_[label]);  // Weighted fusion
    else if (fusion_type_ == 2)
      merged_object.confidence_scores_[label] =
          max(merged_object.confidence_scores_[label], now_confidence);  // Maximum confidence
    printFusionInfo(merged_object, label, "[Fusion]");
  }
}

/**
 * @brief Fusion algorithm for combining confidence scores from multiple observations
 *
 * This function implements a weighted confidence fusion strategy that combines
 * historical confidence with new observations, considering both the quantity
 * of evidence and the quality of individual detections.
 *
 * @param total_num_last Number of points in previous observation
 * @param c_last Previous confidence score
 * @param n_num_now Number of points in current observation
 * @param c_now Current confidence score
 * @param total_now Total points after fusion
 * @param sum Cumulative observation count
 * @return Fused confidence score combining all evidence
 */
double ObjectMap2D::fusionConfidenceScore(
    int total_num_last, double c_last, int n_num_now, double c_now, int total_now, int sum)
{
  double n_now = (double)n_num_now;
  double w_last, w_now, final_score;
  // Calculate weighted fusion based on observation counts
  w_last = (sum - n_now) / sum;                   // Weight for historical evidence
  w_now = n_now / sum;                            // Weight for current observation
  final_score = w_last * c_last + w_now * c_now;  // Weighted combination
  return final_score;
}

bool ObjectMap2D::checkSafety(const Eigen::Vector2i& idx)
{
  if (sdf_map_->getOccupancy(idx) == SDFMap2D::UNKNOWN ||
      sdf_map_->getOccupancy(idx) == SDFMap2D::OCCUPIED || sdf_map_->getInflateOccupancy(idx) == 1)
    return false;
  return true;
}

bool ObjectMap2D::checkSafety(const Eigen::Vector2d& pos)
{
  Eigen::Vector2i idx;
  sdf_map_->posToIndex(pos, idx);
  return checkSafety(idx);
}

void ObjectMap2D::getObjects(
    vector<vector<Eigen::Vector2d>>& clusters, vector<Vector2d>& averages, vector<int>& labels)
{
  clusters.clear();
  averages.clear();
  labels.clear();
  for (auto object : objects_) {
    clusters.push_back(object.cells_);
    averages.push_back(object.average_);
    labels.push_back(object.best_label_);
  }
}

void ObjectMap2D::getObjectBoxes(vector<pair<Eigen::Vector2d, Eigen::Vector2d>>& boxes)
{
  boxes.clear();
  for (auto object : objects_) {
    Vector2d center = (object.box_max2d_ + object.box_min2d_) * 0.5;
    Vector2d scale = object.box_max2d_ - object.box_min2d_;
    boxes.push_back(make_pair(center, scale));
  }
}

void ObjectMap2D::getObjectBoxes(vector<pair<Eigen::Vector3d, Eigen::Vector3d>>& boxes)
{
  boxes.clear();
  for (auto object : objects_) {
    Vector3d center = (object.box_max3d_ + object.box_min3d_) * 0.5;
    Vector3d scale = object.box_max3d_ - object.box_min3d_;
    boxes.push_back(make_pair(center, scale));
  }
}

void ObjectMap2D::getObjectBoxes(vector<Vector3d>& bmin, vector<Vector3d>& bmax)
{
  bmin.clear();
  bmax.clear();
  for (auto object : objects_) {
    bmin.push_back(object.box_min3d_);
    bmax.push_back(object.box_max3d_);
  }
}

void ObjectMap2D::getAllConfidenceObjectClouds(
    pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_clouds)
{
  object_clouds.reset(new pcl::PointCloud<pcl::PointXYZ>());

  // Extract high-confidence object cells
  for (auto object : objects_) {
    if (object.confidence_scores_[0] >= min_confidence_) {
      for (auto cell : object.good_cells_) {
        pcl::PointXYZ point;
        point.x = cell[0];
        point.y = cell[1];
        point.z = 0;
        object_clouds->push_back(point);
      }
    }
  }
}

void ObjectMap2D::getTopConfidenceObjectCloud(
    vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>>& top_object_clouds,
    bool limited_confidence, bool extreme)
{
  top_object_clouds.clear();
  vector<ObjectCluster> top_objects;

  // Confidence filtering strategy
  // TODO: May need logic adjustment for relaxed no limited_confidence conditions
  if (!limited_confidence) {
    // Include all objects without confidence filtering (but skip VLM-invalid ones)
    for (auto object : objects_) {
      if (!isObjectInvalid(object.id_)) {
        top_objects.push_back(object);
      }
    }

    // Sort by confidence score in descending order
    std::sort(
        top_objects.begin(), top_objects.end(), [](const ObjectCluster& a, const ObjectCluster& b) {
          return a.confidence_scores_[0] > b.confidence_scores_[0];
        });

    // Extract point clouds for top-ranked objects
    for (auto top_obj : top_objects) {
      if (top_obj.confidence_scores_[0] <= 0.01)
        break;  // Skip extremely low confidence objects

      pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> top_object_cloud;
      top_object_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());

      for (auto cell : top_obj.good_cells_) {
        pcl::PointXYZ point;
        point.x = cell(0);
        point.y = cell(1);
        point.z = 0;
        top_object_cloud->push_back(point);
      }
      top_object_clouds.push_back(top_object_cloud);
    }

    // Fallback for extreme mode when no high-confidence objects exist
    if (extreme && top_object_clouds.empty()) {
      pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> others_object_cloud;
      others_object_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());

      // Include all object cells regardless of confidence
      for (auto object : objects_) {
        for (auto cell : object.cells_) {
          pcl::PointXYZ point;
          point.x = cell(0);
          point.y = cell(1);
          point.z = 0;
          others_object_cloud->push_back(point);
        }
      }
      top_object_clouds.push_back(others_object_cloud);
    }
  }
  else {
    // Apply confidence filtering with two-tier strategy:
    // Tier 1: best_label == 0 (normal target objects)
    // Tier 2: best_label != 0 but high conf[0] (target objects overshadowed by similar objects)
    double tau_t = getCurrentConfidenceThreshold();

    for (auto object : objects_) {
      // Skip objects that failed VLM validation
      if (isObjectInvalid(object.id_)) {
        ROS_WARN_ONCE("[TopConfidence] Skipping VLM-invalid object id=%d", object.id_);
        continue;
      }

      int max_func_score = 0, best_label = -1;

      // Find best label using functional score (observation count * confidence)
      for (int label = 0; label < (int)object.clouds_.size(); label++) {
        auto obs_sum = object.observation_cloud_sums_[label];
        auto score = object.confidence_scores_[label];
        int func_score = obs_sum * score;
        if (func_score > max_func_score) {
          max_func_score = func_score;
          best_label = label;
        }
      }

      bool shouldInclude = false;

      // Tier 1: best_label is target (0) and meets confidence threshold
      if (best_label == 0 && isConfidenceObject(object)) {
        shouldInclude = true;
        ROS_WARN("[TopConfidence] id=%d TIER1: best_label=0, conf[0]=%.3f >= tau=%.3f",
                 object.id_, object.confidence_scores_[0], tau_t);
      }
      // Tier 2: best_label is NOT target, but conf[0] is significantly high (1.2x threshold)
      // This catches target objects that are overshadowed by similar object detections
      else if (object.confidence_scores_[0] >= 1.2 * tau_t &&
               object.observation_nums_[0] >= min_observation_num_) {
        shouldInclude = true;
        ROS_WARN("[TopConfidence] id=%d TIER2: best_label=%d, conf[0]=%.3f >= 1.2*tau=%.3f",
                 object.id_, best_label, object.confidence_scores_[0], 1.2 * tau_t);
      }

      if (shouldInclude) {
        top_objects.push_back(object);
      }
    }

    // Sort by target label confidence (descending) - highest conf[0] first
    std::sort(
        top_objects.begin(), top_objects.end(), [](const ObjectCluster& a, const ObjectCluster& b) {
          return a.confidence_scores_[0] > b.confidence_scores_[0];
        });

    // Extract point clouds from filtered objects
    // Use cells_ as fallback if good_cells_ is empty (for Tier 2 objects)
    for (auto top_obj : top_objects) {
      pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> top_object_cloud;
      top_object_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());

      const auto& cells_to_use = top_obj.good_cells_.empty() ? top_obj.cells_ : top_obj.good_cells_;

      for (auto cell : cells_to_use) {
        pcl::PointXYZ point;
        point.x = cell(0);
        point.y = cell(1);
        point.z = 0;
        top_object_cloud->push_back(point);
      }

      if (!top_object_cloud->points.empty()) {
        top_object_clouds.push_back(top_object_cloud);
      }
    }
  }
}

bool ObjectMap2D::isConfidenceObject(const ObjectCluster& obj)
{
  // Use dynamic patience-aware threshold tau(t) instead of static min_confidence_
  double tau_t = getCurrentConfidenceThreshold();
  if (obj.confidence_scores_[0] >= tau_t &&
      obj.observation_nums_[0] >= min_observation_num_)
    return true;
  return false;
}

void ObjectMap2D::getSimilarObjectClouds(
    std::vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>>& similar_clouds,
    std::vector<int>& object_ids)
{
  similar_clouds.clear();
  object_ids.clear();

  // Collect objects with best_label_ != 0 (similar objects, not target)
  std::vector<std::pair<int, double>> candidates;  // (object_idx, priority_score)

  for (size_t i = 0; i < objects_.size(); ++i) {
    const auto& obj = objects_[i];

    // Skip VLM-invalid objects
    if (isObjectInvalid(obj.id_)) continue;

    // Only select similar objects (best_label > 0)
    if (obj.best_label_ <= 0) continue;

    // Priority score: target class confidence * observations, or best_label score as fallback
    double priority = obj.confidence_scores_[0] * obj.observation_nums_[0];
    if (priority < 0.01) {
      priority = obj.confidence_scores_[obj.best_label_] *
                 obj.observation_nums_[obj.best_label_];
    }

    candidates.push_back({static_cast<int>(i), priority});
  }

  // Sort by priority (descending)
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  // Extract point clouds and IDs
  for (const auto& cand : candidates) {
    const auto& obj = objects_[cand.first];

    pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> cloud(
        new pcl::PointCloud<pcl::PointXYZ>());

    const auto& cells = obj.good_cells_.empty() ? obj.cells_ : obj.good_cells_;
    for (const auto& cell : cells) {
      pcl::PointXYZ pt;
      pt.x = cell(0);
      pt.y = cell(1);
      pt.z = 0;
      cloud->push_back(pt);
    }

    if (!cloud->empty()) {
      similar_clouds.push_back(cloud);
      object_ids.push_back(obj.id_);
    }
  }

  ROS_INFO("[getSimilarObjectClouds] Found %zu similar objects (best_label != 0)",
           similar_clouds.size());
}

// ==================== Patience-Aware Navigation Implementation ====================

void ObjectMap2D::initTargetSpecificThresholds(ros::NodeHandle& nh)
{
  // Load target-specific thresholds from ROS parameters
  // Each object type has its own (tau_high, tau_low) for dynamic threshold
  // τ(t, obj) = obj.tau_high - (t/T_max) × (obj.tau_high - obj.tau_low)

  // Object categories based on HM3D and MP3D datasets:
  // HM3D: chair, bed, potted plant, toilet, tv, couch
  // MP3D: chair, table, picture, cabinet, pillow, couch, bed, nightstand,
  //       potted plant, sink, toilet, stool, towel, tv, shower, bathtub,
  //       counter, fireplace, gym equipment, seating, clothes

  // Large objects: bed, couch, bathtub, fireplace, gym equipment
  std::vector<std::string> large_objects = {"bed", "couch", "sofa", "bathtub", "fireplace", "gym equipment"};
  // Medium-large objects: tv, counter, cabinet, shower, seating
  std::vector<std::string> medium_large_objects = {"tv", "tv_monitor", "counter", "cabinet", "shower", "seating"};
  // Medium objects: chair, sink, table, nightstand, stool (toilet moved to separate category)
  std::vector<std::string> medium_objects = {"chair", "sink", "table", "nightstand", "stool"};
  // Toilet: separate category for better tuning
  std::vector<std::string> toilet_objects = {"toilet"};
  // Small objects: potted plant, pillow, towel, clothes
  std::vector<std::string> small_objects = {"plant", "potted plant", "pillow", "towel", "clothes"};
  // Fine-grained objects: picture
  std::vector<std::string> fine_objects = {"picture"};

  // Load category thresholds from parameters (with defaults)
  double large_high, large_low;
  double medium_large_high, medium_large_low;
  double medium_high, medium_low;
  double toilet_high, toilet_low;
  double small_high, small_low;
  double fine_high, fine_low;

  // Use passed-in NodeHandle to read parameters (same namespace as other object/ params)
  nh.param("object/threshold/large_high", large_high, 0.50);
  nh.param("object/threshold/large_low", large_low, 0.30);
  nh.param("object/threshold/medium_large_high", medium_large_high, 0.45);
  nh.param("object/threshold/medium_large_low", medium_large_low, 0.28);
  nh.param("object/threshold/medium_high", medium_high, 0.45);
  nh.param("object/threshold/medium_low", medium_low, 0.25);
  nh.param("object/threshold/toilet_high", toilet_high, 0.50);
  nh.param("object/threshold/toilet_low", toilet_low, 0.30);
  nh.param("object/threshold/small_high", small_high, 0.35);
  nh.param("object/threshold/small_low", small_low, 0.18);
  nh.param("object/threshold/fine_high", fine_high, 0.30);
  nh.param("object/threshold/fine_low", fine_low, 0.15);

  // Apply thresholds to each category
  for (const auto& obj : large_objects) {
    target_threshold_configs_[obj] = {large_high, large_low};
  }
  for (const auto& obj : medium_large_objects) {
    target_threshold_configs_[obj] = {medium_large_high, medium_large_low};
  }
  for (const auto& obj : medium_objects) {
    target_threshold_configs_[obj] = {medium_high, medium_low};
  }
  for (const auto& obj : toilet_objects) {
    target_threshold_configs_[obj] = {toilet_high, toilet_low};
  }
  for (const auto& obj : small_objects) {
    target_threshold_configs_[obj] = {small_high, small_low};
  }
  for (const auto& obj : fine_objects) {
    target_threshold_configs_[obj] = {fine_high, fine_low};
  }

  ROS_INFO("[ObjectMap2D] Loaded target-specific threshold configs:");
  ROS_INFO("  - Large (bed, couch, bathtub...):        tau_high=%.2f, tau_low=%.2f", large_high, large_low);
  ROS_INFO("  - Medium-large (tv, counter, cabinet):  tau_high=%.2f, tau_low=%.2f", medium_large_high, medium_large_low);
  ROS_INFO("  - Medium (chair, sink, table...):       tau_high=%.2f, tau_low=%.2f", medium_high, medium_low);
  ROS_INFO("  - Toilet:                               tau_high=%.2f, tau_low=%.2f", toilet_high, toilet_low);
  ROS_INFO("  - Small (plant, pillow, towel):         tau_high=%.2f, tau_low=%.2f", small_high, small_low);
  ROS_INFO("  - Fine (picture):                       tau_high=%.2f, tau_low=%.2f", fine_high, fine_low);
}

ObjectMap2D::TargetThresholdConfig ObjectMap2D::getTargetThresholdConfig() const
{
  if (current_target_category_.empty()) {
    // Default to global tau_high and tau_low
    return {tau_high_, tau_low_};
  }

  auto it = target_threshold_configs_.find(current_target_category_);
  if (it != target_threshold_configs_.end()) {
    return it->second;
  }
  // Unknown category uses global defaults
  return {tau_high_, tau_low_};
}

double ObjectMap2D::getCurrentConfidenceThreshold() const
{
  // Get target-specific config (each object type has its own tau_high and tau_low)
  TargetThresholdConfig config = getTargetThresholdConfig();

  // Paper Equation: τ(t, obj) = obj.tau_high - (t/T_max) × (obj.tau_high - obj.tau_low)
  double t = std::min(current_step_, T_max_);
  double tau_t = config.tau_high - (t / static_cast<double>(T_max_)) * (config.tau_high - config.tau_low);

  return tau_t;
}

void ObjectMap2D::setCurrentStep(int step)
{
  current_step_ = step;
}

void ObjectMap2D::setTargetCategory(const std::string& category)
{
  // Convert to lowercase for matching
  std::string lower_category = category;
  std::transform(lower_category.begin(), lower_category.end(),
                 lower_category.begin(), ::tolower);
  current_target_category_ = lower_category;

  TargetThresholdConfig config = getTargetThresholdConfig();
  ROS_INFO("[ObjectMap2D] Target category set to '%s', tau_high=%.2f, tau_low=%.2f",
           category.c_str(), config.tau_high, config.tau_low);
}

bool ObjectMap2D::isConfirmedTarget(int obj_idx) const
{
  if (obj_idx < 0 || obj_idx >= static_cast<int>(objects_.size()))
    return false;

  // Skip objects marked as invalid (VLM rejected)
  if (isObjectInvalid(obj_idx))
    return false;

  double tau_t = getCurrentConfidenceThreshold();
  return objects_[obj_idx].fused_confidence_ >= tau_t;
}

bool ObjectMap2D::hasConfirmedTarget() const
{
  double tau_t = getCurrentConfidenceThreshold();

  for (const auto& obj : objects_) {
    // Skip objects marked as invalid (VLM rejected)
    if (isObjectInvalid(obj.id_))
      continue;

    // FIX: Only check target label (index 0) confidence, not fused_confidence_
    // fused_confidence_ may come from non-target labels (e.g., label=1 with high conf)
    // This must be consistent with getTopConfidenceObjectCloud's filtering logic
    // which requires best_label == 0 for navigation target selection
    if (obj.confidence_scores_[0] >= tau_t && obj.observation_nums_[0] >= min_observation_num_)
      return true;
  }
  return false;
}

void ObjectMap2D::getConfirmedTargetClouds(
    std::vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>>& confirmed_clouds)
{
  confirmed_clouds.clear();
  double tau_t = getCurrentConfidenceThreshold();

  for (const auto& obj : objects_) {
    // Skip objects marked as invalid (VLM rejected)
    if (isObjectInvalid(obj.id_))
      continue;

    // Dynamic judgment: only return confirmed targets
    if (obj.fused_confidence_ < tau_t)
      continue;

    if (obj.best_label_ < 0 || obj.best_label_ >= static_cast<int>(obj.clouds_.size()))
      continue;

    if (obj.clouds_[obj.best_label_] && !obj.clouds_[obj.best_label_]->empty()) {
      confirmed_clouds.push_back(obj.clouds_[obj.best_label_]);
    }
  }
}

void ObjectMap2D::semanticScoresCallback(const plan_env::SemanticScores::ConstPtr& msg)
{
  // Extract target object from SemanticScores message
  if (msg->target_object.empty())
    return;

  // Only update if target category changed
  if (msg->target_object != current_target_category_) {
    setTargetCategory(msg->target_object);
  }
}

// ==================== Multi-View VLM Validation Methods ====================

bool ObjectMap2D::getObjectCenter(int object_id, Vector2d& center) const
{
  if (object_id < 0 || object_id >= static_cast<int>(objects_.size()))
    return false;

  center = objects_[object_id].average_;
  return true;
}

bool ObjectMap2D::getObjectSize2D(int object_id, Vector2d& size) const
{
  if (object_id < 0 || object_id >= static_cast<int>(objects_.size()))
    return false;

  const auto& obj = objects_[object_id];
  size = obj.box_max2d_ - obj.box_min2d_;
  return true;
}

bool ObjectMap2D::hasLineOfSight(const Vector2d& from_pos, int object_id) const
{
  if (object_id < 0 || object_id >= static_cast<int>(objects_.size()))
    return false;

  Vector2d object_center = objects_[object_id].average_;

  // Use raycaster to check line of sight
  raycaster_->input(from_pos, object_center);
  Vector2i ray_idx;

  while (raycaster_->nextId(ray_idx)) {
    // Check if ray hits an obstacle (but not the object itself)
    if (sdf_map_->getOccupancy(ray_idx) == SDFMap2D::OCCUPIED) {
      // Check if this cell belongs to our target object
      int cell_obj_id = getObjectGrid(ray_idx);
      if (cell_obj_id != object_id) {
        // Hit an obstacle that's not our target object
        return false;
      }
    }
  }

  return true;
}

int ObjectMap2D::findNearestObjectId(const Vector2d& pos, double max_distance) const
{
  int nearest_id = -1;
  double min_dist = max_distance;

  for (size_t i = 0; i < objects_.size(); ++i) {
    // Skip invalid objects
    if (isObjectInvalid(objects_[i].id_))
      continue;

    double dist = (objects_[i].average_ - pos).norm();
    if (dist < min_dist) {
      min_dist = dist;
      nearest_id = objects_[i].id_;
    }
  }

  return nearest_id;
}

bool ObjectMap2D::computeObservationViewpoints(
    int object_id,
    const Vector2d& robot_pos,
    double robot_yaw,
    double hfov_deg,
    std::vector<ObservationViewpoint>& viewpoints)
{
  viewpoints.clear();

  if (object_id < 0 || object_id >= static_cast<int>(objects_.size())) {
    ROS_WARN("[ObjectMap2D] Invalid object_id %d for viewpoint computation", object_id);
    return false;
  }

  const auto& obj = objects_[object_id];
  Vector2d object_center = obj.average_;
  Vector2d object_size = obj.box_max2d_ - obj.box_min2d_;

  // Compute optimal observation distance based on object size and camera FOV
  // Distance = (object_width / 2) / tan(hfov / 2) * margin_factor
  double hfov_rad = hfov_deg * M_PI / 180.0;
  double object_width = std::max(object_size.x(), object_size.y());

  // Relaxed constraint: object should occupy ~30% of FOV width for good visibility
  double fov_fill_factor = 0.3;  // Object fills 30% of FOV
  double min_dist_for_fov = (object_width / fov_fill_factor) / (2.0 * std::tan(hfov_rad / 2.0));

  // Relaxed constraints for practical navigation
  const double min_safe_distance = 0.3;    // Reduced from 0.5m
  const double max_observation_dist = 5.0; // Increased from 3.0m
  const double robot_radius = 0.18;        // Robot collision radius

  double optimal_distance = std::max(min_safe_distance,
                                     std::min(max_observation_dist, min_dist_for_fov));

  ROS_INFO("[VLM Viewpoint] Object %d: center=(%.2f,%.2f), size=(%.2f,%.2f), optimal_dist=%.2f",
           object_id, object_center.x(), object_center.y(),
           object_size.x(), object_size.y(), optimal_distance);

  // Compute direction from robot to object (this will be the "front" direction)
  Vector2d robot_to_object = object_center - robot_pos;
  double front_angle = std::atan2(robot_to_object.y(), robot_to_object.x());

  // Generate candidate viewpoints at different angles
  // Angles are measured from the object looking outward
  std::vector<double> candidate_angles = {
    front_angle + M_PI,           // Front (directly facing object from robot's approach)
    front_angle + M_PI + M_PI/4,  // Left 45°
    front_angle + M_PI - M_PI/4,  // Right 45°
    front_angle + M_PI + M_PI/3,  // Left 60°
    front_angle + M_PI - M_PI/3   // Right 60°
  };

  for (double angle : candidate_angles) {
    // Normalize angle
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;

    ObservationViewpoint vp;

    // Compute viewpoint position (at optimal_distance from object center)
    vp.position.x() = object_center.x() + optimal_distance * std::cos(angle);
    vp.position.y() = object_center.y() + optimal_distance * std::sin(angle);

    // Viewpoint should face towards object center
    vp.yaw = std::atan2(object_center.y() - vp.position.y(),
                        object_center.x() - vp.position.x());

    vp.distance = optimal_distance;
    vp.is_reachable = false;
    vp.has_line_of_sight = false;

    // Check if position is valid (in map, not occupied)
    if (!sdf_map_->isInMap(vp.position)) {
      ROS_DEBUG("[VLM Viewpoint] Position (%.2f,%.2f) out of map", vp.position.x(), vp.position.y());
      continue;
    }

    if (sdf_map_->getOccupancy(vp.position) == SDFMap2D::OCCUPIED) {
      ROS_DEBUG("[VLM Viewpoint] Position (%.2f,%.2f) is occupied", vp.position.x(), vp.position.y());
      continue;
    }

    // Relaxed: allow positions in inflation zone if far enough from obstacles
    // (actual path planning will handle safety)
    double dist_to_obstacle = sdf_map_->getDistance(vp.position);
    if (dist_to_obstacle < robot_radius * 0.8) {  // Allow very close to inflation zone
      ROS_DEBUG("[VLM Viewpoint] Position (%.2f,%.2f) too close to obstacle (%.2f < %.2f)",
                vp.position.x(), vp.position.y(), dist_to_obstacle, robot_radius * 0.8);
      continue;
    }

    // Mark as potentially reachable (actual path check done later in FSM)
    vp.is_reachable = true;

    // Check line of sight to object (optional - don't reject if no LoS, just mark it)
    vp.has_line_of_sight = hasLineOfSight(vp.position, object_id);

    // Accept viewpoint regardless of line-of-sight (object may be partially visible)
    // This allows validation from different angles even with some occlusion
    viewpoints.push_back(vp);
    ROS_DEBUG("[VLM Viewpoint] Viewpoint: pos=(%.2f,%.2f), yaw=%.1f°, dist=%.2f, LoS=%s",
              vp.position.x(), vp.position.y(), vp.yaw * 180 / M_PI, vp.distance,
              vp.has_line_of_sight ? "yes" : "no");
  }

  // Sort viewpoints by quality (prefer closer to robot's current position)
  std::sort(viewpoints.begin(), viewpoints.end(),
            [&robot_pos](const ObservationViewpoint& a, const ObservationViewpoint& b) {
              double dist_a = (a.position - robot_pos).norm();
              double dist_b = (b.position - robot_pos).norm();
              return dist_a < dist_b;
            });

  ROS_INFO("[VLM Viewpoint] Found %zu valid viewpoints for object %d",
           viewpoints.size(), object_id);

  return !viewpoints.empty();
}

bool ObjectMap2D::getObjectInfoForVLM(int object_id, double& fused_confidence,
                                       int& observation_count, double& current_threshold) const
{
  if (object_id < 0 || object_id >= static_cast<int>(objects_.size())) {
    fused_confidence = 0.0;
    observation_count = 0;
    current_threshold = getCurrentConfidenceThreshold();
    return false;
  }

  const auto& obj = objects_[object_id];
  fused_confidence = obj.fused_confidence_;
  observation_count = obj.observation_nums_[0];  // Target label observation count
  current_threshold = getCurrentConfidenceThreshold();

  return true;
}

std::string ObjectMap2D::getSimilarObjectsInfoForVLM(int object_id) const
{
  if (object_id < 0 || object_id >= static_cast<int>(objects_.size())) {
    return "";
  }

  const auto& obj = objects_[object_id];

  // Label name mapping (same as detection system)
  static const std::vector<std::string> label_names = {
    "target",    // label 0 - skip this one
    "similar1",  // label 1
    "similar2",  // label 2
    "similar3",  // label 3
    "similar4",  // label 4
    "similar5"   // label 5
  };

  std::stringstream ss;
  bool first = true;

  // Iterate through similar objects (labels 1-5)
  for (size_t label = 1; label < obj.confidence_scores_.size() && label < label_names.size(); ++label) {
    double conf = obj.confidence_scores_[label];
    int count = obj.observation_nums_[label];

    // Only include if has meaningful confidence and observations
    if (conf > 0.1 && count > 0) {
      if (!first) ss << ",";
      first = false;

      // Format: "label_name:conf:count"
      ss << label_names[label] << ":" << std::fixed << std::setprecision(2) << conf << ":" << count;
    }
  }

  return ss.str();
}

}  // namespace apexnav_planner