#ifndef _IG_VALUE_MAP_H_
#define _IG_VALUE_MAP_H_

/**
 * @file ig_value_map.h
 * @brief Information Gain (IG) Value Map for exploration guidance
 *
 * Reference: main.tex Section III-D "VLM-Based Future IG Estimation"
 *
 * This class implements the IG value map that stores and updates exploration
 * potential values based on VLM-evaluated visual-semantic connectivity.
 * Higher IG values indicate areas that likely lead to unexplored regions.
 *
 * The IG score is computed as:
 *   IG(f_i) ∝ (1/3) * Σ_{j=1}^{3} BLIP2-ITM(I_i, T_j)
 *
 * where T_j are connectivity prompts:
 *   1. "This view shows a corridor or hallway leading to other areas"
 *   2. "There is a doorway or opening that leads to another room"
 *   3. "This passage connects to multiple rooms or spaces"
 *
 * @author Zhi Yang
 */

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <vector>

#include <plan_env/sdf_map2d.h>
#include <plan_env/IGScore.h>

using Eigen::Vector2d;
using Eigen::Vector2i;
using std::vector;

namespace apexnav_planner {

// Forward declaration
class SDFMap2D;

/**
 * @brief Information Gain Value Map
 *
 * Stores exploration potential values for each grid cell based on
 * VLM-evaluated visual-semantic connectivity scores.
 */
class IGValueMap {
public:
  IGValueMap(SDFMap2D* sdf_map, ros::NodeHandle& nh);
  ~IGValueMap() = default;

  // ==================== Core Update Functions ====================

  /**
   * @brief Update IG value map with new observation
   * @param sensor_pos Current sensor position
   * @param sensor_yaw Current sensor yaw angle
   * @param free_grids List of free grid cells in current observation
   * @param ig_score IG score from VLM (0-1)
   */
  void updateIGMap(const Vector2d& sensor_pos, const double& sensor_yaw,
      const vector<Vector2i>& free_grids, const double& ig_score);

  /**
   * @brief Cache sensor state for use with ROS callback updates
   * @param sensor_pos Current sensor position
   * @param sensor_yaw Current sensor yaw angle
   * @param free_grids List of free grid cells in current observation
   */
  void updateSensorState(const Vector2d& sensor_pos, const double& sensor_yaw,
      const vector<Vector2i>& free_grids);

  // ==================== Query Functions ====================

  /**
   * @brief Get IG value at a position
   * @param pos Query position in world coordinates
   * @return IG value (0-1)
   */
  double getIGValue(const Vector2d& pos);
  double getIGValue(const Vector2i& idx);

  /**
   * @brief Get IG confidence at a position
   * @param pos Query position in world coordinates
   * @return Confidence score
   */
  double getIGConfidence(const Vector2d& pos);
  double getIGConfidence(const Vector2i& idx);

  // ==================== Reset and Utility ====================

  /**
   * @brief Reset IG map for new episode
   */
  void reset();

  /**
   * @brief Get current average IG score across observed cells (for debugging)
   */
  double getAverageIGScore() const;

  /**
   * @brief Get maximum IG value in the map
   */
  double getMaxIGValue() const;

private:
  // ==================== Internal Functions ====================

  /**
   * @brief Calculate FOV-based observation confidence
   * Uses cosine-squared model for stronger center weighting
   */
  double getFovConfidence(const Vector2d& sensor_pos, const double& sensor_yaw,
      const Vector2d& pt_pos);

  /**
   * @brief Normalize angle to [-π, π]
   */
  double normalizeAngle(double angle);

  /**
   * @brief Callback for IGScore messages from Habitat
   */
  void igScoreCallback(const plan_env::IGScore::ConstPtr& msg);

  // ==================== Data Members ====================

  // Data buffers
  vector<double> ig_value_buffer_;       ///< Grid-based IG value storage
  vector<double> ig_confidence_buffer_;  ///< Grid-based confidence storage

  // Map reference
  SDFMap2D* sdf_map_;

  // Parameters
  double fov_angle_;  ///< Field of view angle (radians)

  // ROS communication
  ros::Subscriber ig_score_sub_;  ///< Subscriber for IGScore messages

  // Cached sensor state for ROS callback updates
  Vector2d current_sensor_pos_;
  double current_sensor_yaw_;
  vector<Vector2i> current_free_grids_;
  bool has_sensor_state_;
};

// ==================== Inline Implementations ====================

inline double IGValueMap::normalizeAngle(double angle) {
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

inline double IGValueMap::getIGValue(const Vector2d& pos) {
  Vector2i idx;
  sdf_map_->posToIndex(pos, idx);
  return getIGValue(idx);
}

inline double IGValueMap::getIGValue(const Vector2i& idx) {
  if (!sdf_map_->isInMap(idx))
    return 0.0;
  int adr = sdf_map_->toAddress(idx);
  return ig_value_buffer_[adr];
}

inline double IGValueMap::getIGConfidence(const Vector2d& pos) {
  Vector2i idx;
  sdf_map_->posToIndex(pos, idx);
  return getIGConfidence(idx);
}

inline double IGValueMap::getIGConfidence(const Vector2i& idx) {
  if (!sdf_map_->isInMap(idx))
    return 0.0;
  int adr = sdf_map_->toAddress(idx);
  return ig_confidence_buffer_[adr];
}

}  // namespace apexnav_planner
#endif
