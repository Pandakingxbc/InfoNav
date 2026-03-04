#ifndef _MULTI_SOURCE_VALUE_MAP_H_
#define _MULTI_SOURCE_VALUE_MAP_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <vector>
#include <string>
#include <map>

#include <plan_env/sdf_map2d.h>
#include <plan_env/SemanticScores.h>
#include <std_msgs/Empty.h>

using Eigen::Vector2d;
using Eigen::Vector2i;
using Eigen::Vector3d;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;
using std::string;
using std::map;

namespace apexnav_planner {
class SDFMap2D;

/**
 * @brief Single semantic source value map
 *
 * Represents a value map for one semantic hypothesis (e.g., "kitchen", "dining table", etc.)
 */
struct SemanticSource {
  string prompt;                     ///< Semantic text prompt (e.g., "Is there a kitchen ahead?")
  double weight;                     ///< Fusion weight for this semantic source
  vector<double> value_buffer_;      ///< Semantic value for each grid cell
  vector<double> confidence_buffer_; ///< Observation confidence for each grid cell

  SemanticSource(const string& p, double w, int buffer_size)
    : prompt(p), weight(w),
      value_buffer_(buffer_size, 0.0),
      confidence_buffer_(buffer_size, 0.0) {}
};

/**
 * @brief Multi-source semantic value map with rotation sampling
 *
 * This class implements the value map construction and fusion strategy described in
 * the "Value Map Construction and Fusion" document. Key features:
 *
 * 1. Multi-source semantic value maps: Each semantic hypothesis gets its own value map
 * 2. Rotation sampling: VLM evaluates environment at multiple yaw angles before LLM reasoning
 * 3. Weighted fusion: Combines multiple semantic sources with confidence weighting
 * 4. Geometry value integration: Supports fusion with frontier-based geometric exploration
 *
 * Reference: Value_Map_Construction_and_Fusion.md
 */
class MultiSourceValueMap {
public:
  MultiSourceValueMap(SDFMap2D* sdf_map, ros::NodeHandle& nh);
  ~MultiSourceValueMap() = default;

  // ==================== Multi-Source Semantic Value Map ====================

  /**
   * @brief Add a new semantic source (semantic hypothesis)
   * @param prompt Semantic text prompt for VLM evaluation
   * @param weight Fusion weight (default 1.0, will be normalized)
   */
  void addSemanticSource(const string& prompt, double weight = 1.0);

  /**
   * @brief Clear all semantic sources (used when target changes)
   */
  void clearSemanticSources();

  /**
   * @brief Reset all Value Maps for new episode
   * Clears all semantic sources and resets initialization state
   */
  void resetForNewEpisode();

  /**
   * @brief Update a specific semantic source's value map
   * @param source_idx Index of the semantic source
   * @param sensor_pos Current sensor position
   * @param sensor_yaw Current sensor yaw angle
   * @param free_grids List of free grid cells in current observation
   * @param itm_score ITM score from VLM for this semantic source
   */
  void updateSemanticSource(int source_idx, const Vector2d& sensor_pos,
      const double& sensor_yaw, const vector<Vector2i>& free_grids,
      const double& itm_score);

  /**
   * @brief Update all semantic sources with rotation sampling
   *
   * This method performs VLM rotation understanding:
   * 1. Rotate camera through multiple yaw angles
   * 2. For each angle, query VLM with all semantic prompts
   * 3. Update corresponding semantic value maps
   *
   * @param sensor_pos Current sensor position
   * @param sensor_yaw Current sensor yaw angle
   * @param free_grids List of free grid cells
   * @param rgb_image RGB image for VLM evaluation
   * @param rotation_angles List of rotation angles to sample (relative to sensor_yaw)
   */
  void updateWithRotationSampling(const Vector2d& sensor_pos,
      const double& sensor_yaw, const vector<Vector2i>& free_grids,
      const cv::Mat& rgb_image, const vector<double>& rotation_angles);

  // ==================== Value Map Fusion ====================

  /**
   * @brief Update cached sensor state for semantic map updates
   * @param sensor_pos Current sensor position
   * @param sensor_yaw Current sensor yaw angle
   * @param free_grids List of free grid cells in current observation
   */
  void updateSensorState(const Vector2d& sensor_pos, const double& sensor_yaw,
      const vector<Vector2i>& free_grids);

  /**
   * @brief Get fused semantic value at a position
   *
   * Implements: V_semantic(p) = Σ w_k * V^(k)_sem(p)
   *
   * @param pos Query position in world coordinates
   * @return Fused semantic value
   */
  double getFusedSemanticValue(const Vector2d& pos);
  double getFusedSemanticValue(const Vector2i& idx);

  /**
   * @brief Get fused confidence at a position
   * @param pos Query position in world coordinates
   * @return Fused confidence score
   */
  double getFusedConfidence(const Vector2d& pos);
  double getFusedConfidence(const Vector2i& idx);

  /**
   * @brief Get total value combining semantic and geometric components
   *
   * Implements: V_total(p) = λ(t) * V_semantic(p) + (1-λ(t)) * V_geo(p)
   *
   * @param pos Query position
   * @param frontier_value Geometric frontier value at this position
   * @param lambda Semantic weight (0-1), dynamically estimated from reliability
   * @return Combined total value
   */
  double getTotalValue(const Vector2d& pos, double frontier_value, double lambda);
  double getTotalValue(const Vector2i& idx, double frontier_value, double lambda);

  // ==================== Semantic Reliability Estimation ====================

  /**
   * @brief Estimate semantic reliability for dynamic λ(t) calculation
   *
   * Analyzes semantic value map statistics to determine if semantic guidance
   * is reliable. High variance or clear semantic signal → high λ.
   * Low variance or weak signal → low λ (rely more on geometry).
   *
   * @return Estimated lambda value in [0, 1]
   */
  double estimateSemanticReliability();

  // ==================== Accessors ====================

  int getNumSemanticSources() const { return semantic_sources_.size(); }
  const SemanticSource& getSemanticSource(int idx) const { return semantic_sources_[idx]; }
  vector<string> getSemanticPrompts() const;

private:
  /**
   * @brief Calculate FOV-based observation confidence
   * @param sensor_pos Sensor position
   * @param sensor_yaw Sensor yaw angle
   * @param pt_pos Target point position
   * @return Confidence score based on FOV geometry
   */
  double getFovConfidence(const Vector2d& sensor_pos, const double& sensor_yaw,
      const Vector2d& pt_pos);

  /**
   * @brief Normalize angle to [-π, π]
   */
  double normalizeAngle(double angle);

  /**
   * @brief Normalize semantic source weights to sum to 1.0
   */
  void normalizeWeights();

  /**
   * @brief Callback for SemanticScores messages from Habitat
   * @param msg SemanticScores message containing multi-source semantic prompts and scores
   */
  void semanticScoresCallback(const plan_env::SemanticScores::ConstPtr& msg);

  /**
   * @brief Callback for episode reset signal
   * @param msg Empty message triggering reset
   */
  void episodeResetCallback(const std_msgs::Empty::ConstPtr& msg);

  // ==================== Data Members ====================

  SDFMap2D* sdf_map_;                          ///< Reference to SDF map for coordinate transforms
  vector<SemanticSource> semantic_sources_;    ///< List of semantic value maps

  // Parameters
  double fov_angle_;                           ///< Field of view angle (radians)
  bool enable_rotation_sampling_;              ///< Enable VLM rotation sampling
  int num_rotation_samples_;                   ///< Number of rotation angles to sample

  // Semantic reliability estimation
  double semantic_variance_threshold_;         ///< Threshold for semantic variance
  double min_lambda_;                          ///< Minimum lambda value
  double max_lambda_;                          ///< Maximum lambda value

  // ROS communication
  ros::Subscriber semantic_scores_sub_;        ///< Subscriber for SemanticScores messages
  ros::Subscriber episode_reset_sub_;          ///< Subscriber for episode reset signals
  string current_target_object_;               ///< Current navigation target
  bool sources_initialized_;                   ///< Whether semantic sources have been initialized

  // Cached sensor state for updating
  Vector2d current_sensor_pos_;                ///< Current sensor position
  double current_sensor_yaw_;                  ///< Current sensor yaw
  vector<Vector2i> current_free_grids_;        ///< Current free grids in view
};

// ==================== Inline Implementations ====================

inline double MultiSourceValueMap::normalizeAngle(double angle) {
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

inline double MultiSourceValueMap::getFusedSemanticValue(const Vector2d& pos) {
  Vector2i idx;
  sdf_map_->posToIndex(pos, idx);
  return getFusedSemanticValue(idx);
}

inline double MultiSourceValueMap::getFusedConfidence(const Vector2d& pos) {
  Vector2i idx;
  sdf_map_->posToIndex(pos, idx);
  return getFusedConfidence(idx);
}

inline double MultiSourceValueMap::getTotalValue(const Vector2d& pos,
    double frontier_value, double lambda) {
  Vector2i idx;
  sdf_map_->posToIndex(pos, idx);
  return getTotalValue(idx, frontier_value, lambda);
}

inline vector<string> MultiSourceValueMap::getSemanticPrompts() const {
  vector<string> prompts;
  for (const auto& source : semantic_sources_) {
    prompts.push_back(source.prompt);
  }
  return prompts;
}

}  // namespace apexnav_planner
#endif
