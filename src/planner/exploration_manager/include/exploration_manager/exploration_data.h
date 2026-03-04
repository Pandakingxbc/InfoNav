#ifndef _EXPL_DATA_H_
#define _EXPL_DATA_H_

#include <Eigen/Eigen>
#include <iostream>
#include <vector>
#include <limits>

using Eigen::Vector2d;
using Eigen::Vector3d;
using std::pair;
using std::vector;

namespace apexnav_planner {
struct FSMData {
  FSMData()
  {
    trigger_ = false;
    have_odom_ = false;
    have_confidence_ = false;
    have_finished_ = false;
    state_str_ = { "INIT", "WAIT_TRIGGER", "PLAN_ACTION", "WAIT_ACTION_FINISH", "PUB_ACTION",
      "FINISH" };

    odom_pos_ = Vector3d::Zero();
    odom_orient_ = Eigen::Quaterniond::Identity();
    odom_yaw_ = 0.0;
    start_pt_ = Vector3d::Zero();
    start_yaw_ = Vector3d::Zero();
    last_start_pos_ = Vector3d(-100, -100, -100);
    last_next_pos_ = Vector2d(-100, -100);
    newest_action_ = -1;
    init_action_count_ = 0;
    stucking_action_count_ = 0;
    stucking_next_pos_count_ = 0;
    traveled_path_.clear();

    final_result_ = -1;
    replan_flag_ = true;
    dormant_frontier_flag_ = false;
    escape_stucking_flag_ = false;
    escape_stucking_count_ = 0;
    stucking_points_.clear();

    local_pos_ = Vector2d(0, 0);

    // Patience-Aware Navigation (Paper Section 3.5)
    step_count_ = 0;
    target_category_ = "";

    // Suspicious target locking (prevent oscillation)
    suspicious_target_locked_ = false;
    locked_suspicious_pos_ = Vector2d(0, 0);

    // No-progress detection for frontier exploration
    tracked_frontier_pos_ = Vector2d(-1000, -1000);  // Invalid position
    initial_dist_to_frontier_ = -1.0;
    steps_toward_frontier_ = 0;
    best_dist_to_frontier_ = std::numeric_limits<double>::max();

    // Soft arrival detection for object navigation
    // Detects oscillation (forward-backward) when trying to reach target
    approach_best_dist_ = std::numeric_limits<double>::max();
    approach_attempt_count_ = 0;
    approach_no_improvement_count_ = 0;

    // VLM verification state
    vlm_waiting_ = false;
    vlm_target_object_id_ = -1;
    vlm_trigger_type_ = 0;
    vlm_disabled_this_episode_ = false;
    vlm_used_this_episode_ = false;
    vlm_verify_count_ = 0;
    vlm_consecutive_rejection_count_ = 0;

    // Spinning detection
    spinning_step_count_ = 0;
    spinning_start_pos_ = Vector2d(0, 0);
    spinning_boost_active_ = false;
    spinning_boost_frontier_ = Vector2d(0, 0);
  }
  // FSM data
  bool trigger_, have_odom_, have_confidence_;
  bool have_finished_;
  vector<string> state_str_;
  vector<Vector2d> traveled_path_;

  // odometry state
  Eigen::Vector3d odom_pos_;
  Eigen::Quaterniond odom_orient_;
  double odom_yaw_;

  Eigen::Vector3d start_pt_, start_yaw_;
  Eigen::Vector3d last_start_pos_;
  Eigen::Vector2d last_next_pos_;
  int newest_action_;
  int init_action_count_;
  int stucking_action_count_;
  int stucking_next_pos_count_;

  int final_result_;
  bool replan_flag_, dormant_frontier_flag_;
  bool escape_stucking_flag_;
  int escape_stucking_count_;
  Vector2d escape_stucking_pos_;
  double escape_stucking_yaw_;
  vector<Vector3d> stucking_points_;

  Vector2d local_pos_;

  // Universal stuck recovery (for all navigation modes)
  // Tracks consecutive stuck actions and triggers obstacle marking + replan
  int stuck_replan_count_;           ///< Consecutive stuck actions toward current target
  bool stuck_obstacle_marked_;       ///< Whether obstacle has been marked in front
  Vector2d stuck_target_pos_;        ///< Target position when stuck recovery started

  // Patience-Aware Navigation (Paper Section 3.5)
  int step_count_;              ///< Current exploration step count
  std::string target_category_; ///< Current target object category

  // Suspicious target locking (prevent oscillation when no frontier)
  bool suspicious_target_locked_;   ///< Whether a suspicious target is locked
  Vector2d locked_suspicious_pos_;  ///< Locked target position (path is replanned each step)

  // No-progress detection for frontier exploration
  // Detects when agent is moving but not making progress toward frontier
  Vector2d tracked_frontier_pos_;      ///< Currently tracked frontier position
  double initial_dist_to_frontier_;    ///< Distance to frontier when tracking started
  int steps_toward_frontier_;          ///< Steps taken toward current frontier
  double best_dist_to_frontier_;       ///< Best (minimum) distance achieved so far

  // Soft arrival detection for object navigation
  // Detects when agent oscillates (forward-backward) without getting closer
  // If best distance doesn't improve for several steps, consider arrived
  double approach_best_dist_;          ///< Best distance achieved during current approach
  int approach_attempt_count_;         ///< Total steps in current approach attempt
  int approach_no_improvement_count_;  ///< Consecutive steps without improvement

  // VLM verification state (synchronous blocking mode)
  // When vlm_waiting_ is true, FSM should not output any action
  bool vlm_waiting_;                   ///< Whether waiting for VLM verification result
  int vlm_target_object_id_;           ///< Object ID being verified by VLM
  int vlm_trigger_type_;               ///< Trigger type: 0=distance, 1=early, 2=rescue
  bool vlm_disabled_this_episode_;     ///< VLM disabled due to timeout (30s)
  bool vlm_used_this_episode_;         ///< Whether VLM was used this episode
  int vlm_verify_count_;               ///< Number of VLM verifications this episode
  int vlm_consecutive_rejection_count_; ///< Consecutive VLM rejections (for escape condition)

  // Spinning detection: detect when agent rotates in place without moving
  // If position barely changes but yaw keeps changing, agent is likely spinning
  int spinning_step_count_;            ///< Consecutive steps where position barely changes
  Vector2d spinning_start_pos_;        ///< Position when spinning detection started
  bool spinning_boost_active_;         ///< Whether frontier value boost is active
  Vector2d spinning_boost_frontier_;   ///< Frontier position being boosted
};

struct FSMParam {
  FSMParam()
  {
    vis_scale_ = 0.1;

    const double step_length = 0.25;
    const double angle_increment = M_PI / 6;
    action_steps_.clear();
    for (int i = 0; i < 12; ++i) {
      double angle = i * angle_increment;
      Vector2d step(step_length * cos(angle), step_length * sin(angle));
      action_steps_.push_back(step);
    }
  }
  double vis_scale_;
  vector<Vector2d> action_steps_;
};

struct ExplorationData {
  ExplorationData()
  {
    frontiers_.clear();
    frontier_averages_.clear();
    dormant_frontiers_.clear();
    dormant_frontier_averages_.clear();
    objects_.clear();
    object_averages_.clear();
    object_labels_.clear();
    next_pos_ = Vector2d(0, 0);
    next_best_path_.clear();
    tsp_tour_.clear();
  }
  vector<vector<Vector2d>> frontiers_, dormant_frontiers_;
  vector<Vector2d> frontier_averages_, dormant_frontier_averages_;
  vector<vector<Vector2d>> objects_;
  vector<Vector2d> object_averages_;
  vector<int> object_labels_;
  Vector2d next_pos_;
  vector<Vector2d> next_best_path_;
  vector<Vector2d> tsp_tour_;
};

struct ExplorationParam {
  enum POLICY_MODE { DISTANCE, SEMANTIC, HYBRID, TSP_DIST, WTRP };
  // params
  int policy_mode_;
  int top_k_value_;  // Number of top value frontiers for TSP/WTRP in HYBRID/WTRP mode
  std::string tsp_dir_;

  // Dual-Value Fusion (Paper Section III-E)
  // V_total(f_i) = alpha * V_sem(f_i) + (1-alpha) * V_ig(f_i)
  double fusion_alpha_;  ///< Weight for semantic value (0-1), default 0.5
  bool use_ig_fusion_;   ///< Enable IG fusion in frontier evaluation

  // Adaptive Alpha (Paper Equation: alpha(t) = alpha_base + (1-alpha_base) * max_fi(V_HSVM(fi)) / tau_conf)
  bool use_adaptive_alpha_;   ///< Enable adaptive alpha based on frontier semantic values
  double alpha_base_;         ///< Base alpha value when no strong semantic cues (default 0.8)
  double tau_conf_;           ///< Confidence threshold for alpha adaptation (default 0.8)

  // WTRP (Weighted Traveling Repairman Problem) parameters
  double wtrp_temperature_;    ///< Softmax temperature tau for weight computation (0.3-0.7)
  int wtrp_max_brute_force_;   ///< Max frontiers for brute-force permutation (default: 10)
  double wtrp_hysteresis_ratio_;  ///< Hysteresis ratio: only switch target if new cost < old * ratio
};

}  // namespace apexnav_planner

#endif