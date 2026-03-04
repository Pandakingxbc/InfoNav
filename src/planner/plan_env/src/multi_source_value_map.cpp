/**
 * @file multi_source_value_map.cpp
 * @brief Implementation of multi-source semantic value mapping with rotation sampling
 *
 * This file implements the MultiSourceValueMap class which provides:
 * 1. Multi-source semantic value maps (one per semantic hypothesis)
 * 2. VLM rotation sampling for comprehensive environment understanding
 * 3. Confidence-weighted fusion of multiple semantic sources
 * 4. Dynamic semantic-geometric fusion with reliability estimation
 *
 * Reference: "Value Map Construction and Fusion" design document
 *
 * @author Zager-Zhang
 */

#include <plan_env/multi_source_value_map.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace apexnav_planner {

MultiSourceValueMap::MultiSourceValueMap(SDFMap2D* sdf_map, ros::NodeHandle& nh)
  : sdf_map_(sdf_map),
    sources_initialized_(false),
    current_sensor_yaw_(0.0)
{
  // Load parameters from ROS parameter server
  nh.param("value_map/fov_angle_deg", fov_angle_, 79.0);
  fov_angle_ = fov_angle_ * M_PI / 180.0;  // Convert to radians

  nh.param("value_map/enable_rotation_sampling", enable_rotation_sampling_, true);
  nh.param("value_map/num_rotation_samples", num_rotation_samples_, 3);

  nh.param("value_map/semantic_variance_threshold", semantic_variance_threshold_, 0.01);
  nh.param("value_map/min_lambda", min_lambda_, 0.2);
  nh.param("value_map/max_lambda", max_lambda_, 0.8);

  // Subscribe to SemanticScores messages from Habitat
  semantic_scores_sub_ = nh.subscribe<plan_env::SemanticScores>(
      "/habitat/semantic_scores",
      10,
      &MultiSourceValueMap::semanticScoresCallback,
      this
  );

  // Subscribe to episode reset signals
  episode_reset_sub_ = nh.subscribe<std_msgs::Empty>(
      "/habitat/episode_reset",
      10,
      &MultiSourceValueMap::episodeResetCallback,
      this
  );

  ROS_INFO("[MultiSourceValueMap] Initialized with rotation sampling %s, %d samples",
      enable_rotation_sampling_ ? "enabled" : "disabled", num_rotation_samples_);
  ROS_INFO("[MultiSourceValueMap] Subscribed to /habitat/semantic_scores");
  ROS_INFO("[MultiSourceValueMap] Subscribed to /habitat/episode_reset");
}

void MultiSourceValueMap::addSemanticSource(const string& prompt, double weight)
{
  int buffer_size = sdf_map_->getVoxelNum();
  semantic_sources_.emplace_back(prompt, weight, buffer_size);
  normalizeWeights();

  ROS_INFO("[MultiSourceValueMap] Added semantic source: '%s' (weight: %.3f)",
      prompt.c_str(), weight);
}

void MultiSourceValueMap::clearSemanticSources()
{
  semantic_sources_.clear();
  ROS_INFO("[MultiSourceValueMap] Cleared all semantic sources");
}

void MultiSourceValueMap::resetForNewEpisode()
{
  // Clear all semantic sources and their value/confidence buffers
  clearSemanticSources();

  // Reset initialization state
  sources_initialized_ = false;
  current_target_object_.clear();

  // Clear cached sensor state
  current_sensor_pos_ = Vector2d::Zero();
  current_sensor_yaw_ = 0.0;
  current_free_grids_.clear();

  ROS_INFO("[MultiSourceValueMap] Reset for new episode");
}

void MultiSourceValueMap::normalizeWeights()
{
  if (semantic_sources_.empty())
    return;

  double sum = 0.0;
  for (const auto& source : semantic_sources_) {
    sum += source.weight;
  }

  if (sum > 1e-6) {
    for (auto& source : semantic_sources_) {
      source.weight /= sum;
    }
  }
}

void MultiSourceValueMap::updateSensorState(const Vector2d& sensor_pos,
    const double& sensor_yaw, const vector<Vector2i>& free_grids)
{
  current_sensor_pos_ = sensor_pos;
  current_sensor_yaw_ = sensor_yaw;
  current_free_grids_ = free_grids;
}

void MultiSourceValueMap::updateSemanticSource(int source_idx,
    const Vector2d& sensor_pos, const double& sensor_yaw,
    const vector<Vector2i>& free_grids, const double& itm_score)
{
  if (source_idx < 0 || source_idx >= semantic_sources_.size()) {
    ROS_ERROR("[MultiSourceValueMap] Invalid source index: %d", source_idx);
    return;
  }

  SemanticSource& source = semantic_sources_[source_idx];

  // Update value map using confidence-weighted fusion (same as ValueMap)
  for (const auto& grid : free_grids) {
    Vector2d pos;
    sdf_map_->indexToPos(grid, pos);
    int adr = sdf_map_->toAddress(grid);

    // Calculate FOV-based confidence for current observation
    double now_confidence = getFovConfidence(sensor_pos, sensor_yaw, pos);
    double now_value = itm_score;

    // Retrieve existing confidence and value
    double last_confidence = source.confidence_buffer_[adr];
    double last_value = source.value_buffer_[adr];

    // Apply confidence-weighted fusion with quadratic confidence combination
    // This gives more weight to observations with higher confidence
    double total_confidence = now_confidence + last_confidence;
    if (total_confidence > 1e-6) {
      source.confidence_buffer_[adr] =
          (now_confidence * now_confidence + last_confidence * last_confidence) /
          total_confidence;
      source.value_buffer_[adr] =
          (now_confidence * now_value + last_confidence * last_value) /
          total_confidence;
    }
  }
}

void MultiSourceValueMap::updateWithRotationSampling(const Vector2d& sensor_pos,
    const double& sensor_yaw, const vector<Vector2i>& free_grids,
    const cv::Mat& rgb_image, const vector<double>& rotation_angles)
{
  if (!enable_rotation_sampling_) {
    ROS_WARN_THROTTLE(10.0, "[MultiSourceValueMap] Rotation sampling is disabled");
    return;
  }

  if (semantic_sources_.empty()) {
    ROS_WARN_THROTTLE(10.0, "[MultiSourceValueMap] No semantic sources to update");
    return;
  }

  // TODO: This is a placeholder for VLM rotation sampling integration
  // In the full implementation, this would:
  // 1. Rotate the camera view through specified angles
  // 2. For each rotation, query VLM with all semantic prompts
  // 3. Update corresponding semantic value maps
  //
  // For now, we just update with the current view
  ROS_WARN_THROTTLE(30.0,
      "[MultiSourceValueMap] Rotation sampling not yet integrated with VLM server");

  // Placeholder: Update each semantic source with same ITM score
  // (In practice, each source would get its own ITM score from VLM)
  for (size_t i = 0; i < semantic_sources_.size(); ++i) {
    // This would be replaced with actual VLM query:
    // double itm_score = queryVLM(rgb_image, semantic_sources_[i].prompt);
    double itm_score = 0.5;  // Placeholder
    updateSemanticSource(i, sensor_pos, sensor_yaw, free_grids, itm_score);
  }
}

double MultiSourceValueMap::getFusedSemanticValue(const Vector2i& idx)
{
  if (semantic_sources_.empty())
    return 0.0;

  int adr = sdf_map_->toAddress(idx);
  double fused_value = 0.0;

  // Weighted sum: V_semantic(p) = Σ w_k * V^(k)_sem(p)
  for (const auto& source : semantic_sources_) {
    fused_value += source.weight * source.value_buffer_[adr];
  }

  return fused_value;
}

double MultiSourceValueMap::getFusedConfidence(const Vector2i& idx)
{
  if (semantic_sources_.empty())
    return 0.0;

  int adr = sdf_map_->toAddress(idx);
  double fused_confidence = 0.0;

  // Weighted sum of confidences
  for (const auto& source : semantic_sources_) {
    fused_confidence += source.weight * source.confidence_buffer_[adr];
  }

  return fused_confidence;
}

double MultiSourceValueMap::getTotalValue(const Vector2i& idx,
    double frontier_value, double lambda)
{
  double semantic_value = getFusedSemanticValue(idx);

  // V_total(p) = λ(t) * V_semantic(p) + (1-λ(t)) * V_geo(p)
  return lambda * semantic_value + (1.0 - lambda) * frontier_value;
}

double MultiSourceValueMap::estimateSemanticReliability()
{
  if (semantic_sources_.empty())
    return min_lambda_;  // No semantic info, rely on geometry

  // Collect semantic values from all sources across all free space
  vector<double> all_semantic_values;
  Eigen::Vector2d map_min, map_max;
  sdf_map_->getMapBoundary(map_min, map_max);

  Vector2i min_idx, max_idx;
  sdf_map_->posToIndex(map_min, min_idx);
  sdf_map_->posToIndex(map_max, max_idx);

  // Sample semantic values from free space
  for (int x = min_idx(0); x <= max_idx(0); x += 2) {  // Subsample for efficiency
    for (int y = min_idx(1); y <= max_idx(1); y += 2) {
      Vector2i idx(x, y);
      if (!sdf_map_->isInMap(idx))
        continue;

      // Only consider free space
      if (sdf_map_->getOccupancy(idx) != SDFMap2D::FREE)
        continue;

      double value = getFusedSemanticValue(idx);
      double confidence = getFusedConfidence(idx);

      // Only include cells with some observation confidence
      if (confidence > 0.01) {
        all_semantic_values.push_back(value);
      }
    }
  }

  if (all_semantic_values.empty())
    return min_lambda_;  // No observations yet

  // Calculate statistics
  double mean = std::accumulate(all_semantic_values.begin(),
                    all_semantic_values.end(), 0.0) / all_semantic_values.size();

  double variance = 0.0;
  for (double val : all_semantic_values) {
    variance += (val - mean) * (val - mean);
  }
  variance /= all_semantic_values.size();

  double std_dev = std::sqrt(variance);

  // Find max value
  double max_value = *std::max_element(all_semantic_values.begin(),
                                        all_semantic_values.end());

  // Estimate reliability based on:
  // 1. Variance (high variance = clear semantic structure)
  // 2. Max value (high max = strong semantic signal)
  // 3. Max-to-mean ratio (high ratio = distinctive target region)

  double max_to_mean = (mean > 1e-6) ? (max_value / mean) : 1.0;

  // Heuristic: high variance and high max-to-mean → high reliability → high λ
  double variance_score = std::min(1.0, variance / semantic_variance_threshold_);
  double max_score = std::min(1.0, max_value);
  double ratio_score = std::min(1.0, (max_to_mean - 1.0) / 0.5);  // Normalize around 1.5

  double reliability = (variance_score + max_score + ratio_score) / 3.0;

  // Map reliability to lambda range
  double lambda = min_lambda_ + reliability * (max_lambda_ - min_lambda_);

  ROS_INFO_THROTTLE(5.0,
      "[MultiSourceValueMap] Semantic reliability: var=%.4f, max=%.3f, "
      "max/mean=%.2f → λ=%.3f",
      variance, max_value, max_to_mean, lambda);

  return lambda;
}

double MultiSourceValueMap::getFovConfidence(const Vector2d& sensor_pos,
    const double& sensor_yaw, const Vector2d& pt_pos)
{
  // Calculate relative position vector from sensor to target point
  Vector2d rel_pos = pt_pos - sensor_pos;
  double angle_to_point = atan2(rel_pos(1), rel_pos(0));

  // Normalize angles to [-π, π] range for consistent angular arithmetic
  double normalized_sensor_yaw = normalizeAngle(sensor_yaw);
  double normalized_angle_to_point = normalizeAngle(angle_to_point);
  double relative_angle = normalizeAngle(normalized_angle_to_point - normalized_sensor_yaw);

  // Apply cosine-squared FOV confidence model
  // Points in center of FOV get higher confidence
  double value = std::cos(relative_angle / (fov_angle_ / 2) * (M_PI / 2));
  return value * value;  // Square for stronger center weighting
}

void MultiSourceValueMap::semanticScoresCallback(
    const plan_env::SemanticScores::ConstPtr& msg)
{
  // Check if message has valid data
  if (msg->prompts.empty() || msg->scores.empty() || msg->weights.empty()) {
    ROS_WARN("[MultiSourceValueMap] Received empty SemanticScores message");
    return;
  }

  // Verify all arrays have the same length
  size_t num_sources = msg->prompts.size();
  if (msg->scores.size() != num_sources || msg->weights.size() != num_sources) {
    ROS_ERROR("[MultiSourceValueMap] SemanticScores arrays have mismatched lengths: "
              "prompts=%zu, scores=%zu, weights=%zu",
              msg->prompts.size(), msg->scores.size(), msg->weights.size());
    return;
  }

  // Check if this is a new target object or first initialization
  bool target_changed = (msg->target_object != current_target_object_);

  if (!sources_initialized_ || target_changed) {
    // Initialize or reinitialize semantic sources
    ROS_INFO("[MultiSourceValueMap] %s semantic sources for target '%s'",
             target_changed ? "Reinitializing" : "Initializing",
             msg->target_object.c_str());

    clearSemanticSources();

    for (size_t i = 0; i < num_sources; ++i) {
      addSemanticSource(msg->prompts[i], msg->weights[i]);
    }

    current_target_object_ = msg->target_object;
    sources_initialized_ = true;

    ROS_INFO("[MultiSourceValueMap] Initialized %zu semantic sources", num_sources);

    // Log details for debugging
    ROS_INFO("[MultiSourceValueMap] Semantic sources:");
    for (size_t i = 0; i < num_sources; ++i) {
      ROS_INFO("  [%zu] weight=%.3f, prompt='%s'",
               i, msg->weights[i], msg->prompts[i].c_str());
    }
  }

  // Update each semantic source with its corresponding score
  // Note: We need current sensor state (position, yaw, free_grids) to update
  // These should be cached from the perception/odometry system
  if (current_free_grids_.empty()) {
    ROS_WARN_THROTTLE(5.0,
        "[MultiSourceValueMap] No free grids available for update. "
        "Make sure to call updateSemanticSource with sensor state.");
    return;
  }

  for (size_t i = 0; i < num_sources && i < semantic_sources_.size(); ++i) {
    // Update this semantic source's value map with the ITM score from Habitat
    updateSemanticSource(
        i,                        // source index
        current_sensor_pos_,      // sensor position
        current_sensor_yaw_,      // sensor yaw
        current_free_grids_,      // free grids in current view
        msg->scores[i]            // ITM cosine score from BLIP2
    );
  }

  ROS_DEBUG("[MultiSourceValueMap] Updated %zu semantic sources with scores", num_sources);
}

void MultiSourceValueMap::episodeResetCallback(const std_msgs::Empty::ConstPtr& msg)
{
  ROS_INFO("[MultiSourceValueMap] Received episode reset signal");
  resetForNewEpisode();
}

}  // namespace apexnav_planner
