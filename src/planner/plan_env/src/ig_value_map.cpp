/**
 * @file ig_value_map.cpp
 * @brief Implementation of Information Gain (IG) Value Map
 *
 * Reference: main.tex Section III-D "VLM-Based Future IG Estimation"
 *
 * This file implements the IGValueMap class which provides exploration
 * guidance by tracking visual-semantic connectivity scores from VLM.
 *
 * @author Zhi Yang
 */

#include <plan_env/ig_value_map.h>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace apexnav_planner {

IGValueMap::IGValueMap(SDFMap2D* sdf_map, ros::NodeHandle& nh)
  : sdf_map_(sdf_map),
    has_sensor_state_(false),
    current_sensor_yaw_(0.0)
{
  // Initialize buffers
  int voxel_num = sdf_map_->getVoxelNum();
  ig_value_buffer_.resize(voxel_num, 0.0);
  ig_confidence_buffer_.resize(voxel_num, 0.0);

  // Load parameters
  nh.param("ig_value_map/fov_angle_deg", fov_angle_, 79.0);
  fov_angle_ = fov_angle_ * M_PI / 180.0;  // Convert to radians

  // Subscribe to IGScore messages from Habitat
  ig_score_sub_ = nh.subscribe<plan_env::IGScore>(
      "/habitat/ig_score",
      10,
      &IGValueMap::igScoreCallback,
      this
  );

  ROS_INFO("[IGValueMap] Initialized with FOV angle %.1f deg", fov_angle_ * 180.0 / M_PI);
  ROS_INFO("[IGValueMap] Subscribed to /habitat/ig_score");
}

void IGValueMap::reset()
{
  // Clear all buffers
  std::fill(ig_value_buffer_.begin(), ig_value_buffer_.end(), 0.0);
  std::fill(ig_confidence_buffer_.begin(), ig_confidence_buffer_.end(), 0.0);

  // Reset sensor state
  has_sensor_state_ = false;
  current_sensor_pos_ = Vector2d::Zero();
  current_sensor_yaw_ = 0.0;
  current_free_grids_.clear();

  ROS_INFO("[IGValueMap] Reset for new episode");
}

void IGValueMap::updateSensorState(const Vector2d& sensor_pos, const double& sensor_yaw,
    const vector<Vector2i>& free_grids)
{
  current_sensor_pos_ = sensor_pos;
  current_sensor_yaw_ = sensor_yaw;
  current_free_grids_ = free_grids;
  has_sensor_state_ = true;
}

void IGValueMap::updateIGMap(const Vector2d& sensor_pos, const double& sensor_yaw,
    const vector<Vector2i>& free_grids, const double& ig_score)
{
  for (const auto& grid : free_grids) {
    Vector2d pos;
    sdf_map_->indexToPos(grid, pos);
    int adr = sdf_map_->toAddress(grid);

    // Calculate FOV-based confidence for current observation
    double now_confidence = getFovConfidence(sensor_pos, sensor_yaw, pos);
    double now_value = ig_score;

    // Retrieve existing confidence and value
    double last_confidence = ig_confidence_buffer_[adr];
    double last_value = ig_value_buffer_[adr];

    // Apply confidence-weighted fusion with quadratic confidence combination
    // Same fusion strategy as semantic value map for consistency
    double total_confidence = now_confidence + last_confidence;
    if (total_confidence > 1e-6) {
      ig_confidence_buffer_[adr] =
          (now_confidence * now_confidence + last_confidence * last_confidence) /
          total_confidence;
      ig_value_buffer_[adr] =
          (now_confidence * now_value + last_confidence * last_value) /
          total_confidence;
    }
  }
}

double IGValueMap::getFovConfidence(const Vector2d& sensor_pos, const double& sensor_yaw,
    const Vector2d& pt_pos)
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

void IGValueMap::igScoreCallback(const plan_env::IGScore::ConstPtr& msg)
{
  if (!has_sensor_state_) {
    ROS_WARN_THROTTLE(5.0, "[IGValueMap] No sensor state available for IG update. "
                           "Make sure to call updateSensorState() first.");
    return;
  }

  if (current_free_grids_.empty()) {
    ROS_WARN_THROTTLE(5.0, "[IGValueMap] No free grids available for update.");
    return;
  }

  // Update IG map with received score
  updateIGMap(
      current_sensor_pos_,
      current_sensor_yaw_,
      current_free_grids_,
      msg->ig_score
  );

  ROS_DEBUG_THROTTLE(2.0, "[IGValueMap] Updated IG map with score %.3f "
                          "(corridor: %.3f, doorway: %.3f, passage: %.3f)",
                     msg->ig_score, msg->corridor_score,
                     msg->doorway_score, msg->passage_score);
}

double IGValueMap::getAverageIGScore() const
{
  double sum = 0.0;
  int count = 0;

  for (size_t i = 0; i < ig_value_buffer_.size(); ++i) {
    if (ig_confidence_buffer_[i] > 0.01) {
      sum += ig_value_buffer_[i];
      count++;
    }
  }

  return (count > 0) ? (sum / count) : 0.0;
}

double IGValueMap::getMaxIGValue() const
{
  double max_val = 0.0;

  for (size_t i = 0; i < ig_value_buffer_.size(); ++i) {
    if (ig_confidence_buffer_[i] > 0.01) {
      max_val = std::max(max_val, ig_value_buffer_[i]);
    }
  }

  return max_val;
}

}  // namespace apexnav_planner
