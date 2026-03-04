#ifndef _FAST_EXPLORATION_FSM_H_
#define _FAST_EXPLORATION_FSM_H_

// Third-party libraries
#include <Eigen/Eigen>

// Standard C++ libraries
#include <deque>
#include <memory>
#include <string>
#include <vector>
#include <utility>

// ROS core
#include <ros/ros.h>

// ROS message types
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>
#include <visualization_msgs/Marker.h>
#include <plan_env/MultipleMasksWithConfidence.h>
#include <plan_env/VLMVerificationRequest.h>
#include <plan_env/VLMVerificationResult.h>
#include <plan_env/ObjectUpdateInfo.h>

// Forward declaration for ObjectMap2D types
#include <plan_env/object_map2d.h>

using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;

namespace apexnav_planner {
// Centralized constants for ExplorationFSM (mirrors the style of FSMConstants in fsm2.h)
namespace FSMConstants {
// Timers (s)
constexpr double EXEC_TIMER_DURATION = 0.01;
constexpr double FRONTIER_TIMER_DURATION = 1.0;  // Increased from 0.25s to reduce CPU load

// Robot Action
constexpr double ACTION_DISTANCE = 0.25;
constexpr double ACTION_ANGLE = M_PI / 6.0;

// Distances (m)
constexpr double STUCKING_DISTANCE = 0.05;       // consider stuck if movement < this
constexpr double REACH_DISTANCE = 0.20;          // reach object distance
constexpr double SOFT_REACH_DISTANCE = 0.45;     // soft reach distance for object
constexpr double LOCAL_DISTANCE = 0.80;          // local target lookahead
constexpr double FORWARD_DISTANCE = 0.15;        // min clearance for marking obstacles
constexpr double FORCE_DORMANT_DISTANCE = 0.35;  // force dormant frontier if very close
constexpr double MIN_SAFE_DISTANCE = 0.15;       // min safe distance to obstacles

// Counters / thresholds
constexpr int MAX_STUCKING_COUNT = 25;           // max consecutive stuck actions -> try object map fallback
constexpr int MAX_STUCKING_NEXT_POS_COUNT = 14;  // times next_pos unchanged while stuck
constexpr int ESCAPE_SEQUENCE_FRONTIER = 10;     // escape sequence steps for frontier exploration
constexpr int ESCAPE_SEQUENCE_OBJECT = 20;       // escape sequence steps for object navigation (longer)

// No-progress detection thresholds (for unreachable frontiers)
// Agent may be moving but not making progress toward frontier due to depth defects or discrete actions
constexpr int NO_PROGRESS_MAX_STEPS = 50;        // max steps toward same frontier before giving up
constexpr double NO_PROGRESS_MIN_IMPROVEMENT = 0.3;  // min distance improvement required (meters)
constexpr double NO_PROGRESS_CHECK_INTERVAL = 15;    // check progress every N steps
constexpr double FRONTIER_CHANGE_THRESHOLD = 0.5;    // consider frontier changed if moved > this (meters)

// Universal stuck recovery thresholds (for all navigation modes)
// When agent is stuck, mark obstacle in front and trigger replan before marking target as unreachable
constexpr double TARGET_CHANGE_THRESHOLD = 0.5;      // consider target changed if moved > this (meters)
constexpr int STUCK_REPLAN_TRIGGER_COUNT = 12;       // consecutive stuck actions before marking obstacle and replanning
constexpr double STUCK_OBSTACLE_MARK_DISTANCE = 0.3; // distance in front of agent to mark as obstacle (meters)
constexpr double STUCK_OBSTACLE_MARK_WIDTH = 0.2;    // width of obstacle mark perpendicular to heading (meters)

// Soft arrival detection thresholds (for object navigation)
// Detects oscillation (forward-backward) when agent cannot get closer to target
// If best distance doesn't improve for several consecutive steps, consider arrived
constexpr double SOFT_ARRIVAL_DISTANCE = 0.50;       // trigger soft arrival check when dist < this
constexpr double SOFT_ARRIVAL_IMPROVEMENT_THRESH = 0.03;  // improvement must exceed this to count
constexpr int SOFT_ARRIVAL_MAX_NO_IMPROVE = 8;       // max steps without improvement before arrival

// Cost weights
constexpr double TARGET_WEIGHT = 150.0;
constexpr double TARGET_CLOSE_WEIGHT_1 = 2000.0;  // penalize moving away
constexpr double TARGET_CLOSE_WEIGHT_2 = 200.0;   // encourage moving closer
constexpr double SAFETY_WEIGHT = 1.0;
constexpr double SAMPLE_NUM = 10.0;  // samples along a step for safety cost

// Visualization / robot marker
constexpr double VIS_SCALE_FACTOR = 1.8;  // multiply by map resolution
constexpr double ROBOT_HEIGHT = 0.15;
constexpr double ROBOT_RADIUS = 0.18;

// ==================== VLM Sliding Window Constants ====================
// Distance thresholds for VLM validation phases
constexpr double VLM_TRIGGER_DISTANCE = 0.40;    ///< Trigger VLM validation (2 × REACH_DISTANCE)
constexpr double VLM_COLLECT_DISTANCE = 2.0;     ///< Start collecting candidate frames
// Sliding window parameters
constexpr int VLM_WINDOW_MAX_FRAMES = 3;         ///< Maximum frames in sliding window
constexpr double VLM_MIN_DETECTION_CONFIDENCE = 0.3;  ///< Minimum detector confidence
constexpr double VLM_FRAME_MAX_AGE = 5.0;        ///< Maximum age of cached frames (seconds)

// ==================== Approach Detection Check (Object Navigation) ====================
// Verify target visibility during final approach steps before reaching target
// Must detect target at least once in the last N steps before dist < NEAR_DISTANCE
constexpr double APPROACH_CHECK_NEAR_DISTANCE = 0.6;  ///< Trigger validation when dist < this (meters)
constexpr int APPROACH_CHECK_STEP_WINDOW = 8;         ///< Number of recent steps to check for detection
constexpr double APPROACH_CONFIDENCE_RATIO = 0.4;        ///< Min confidence ratio of tau_t for approach detection

// ==================== Final Approach Validation (Look-Around Scan) ====================
// When reaching target, rotate to face object and perform look-up/look-down scan
// Combined with sliding window detection using OR logic for more robust validation
constexpr double FINAL_APPROACH_YAW_THRESHOLD = M_PI / 12.0;  ///< ~15 degrees alignment threshold
constexpr double FINAL_APPROACH_SCAN_WAIT = 1.0;              ///< Wait time for detection after each scan action (seconds)
constexpr double FINAL_APPROACH_YAW_ADJUST = M_PI / 12.0;     ///< ~15 degrees for left/right micro-adjustment

// ==================== VLM Approach Verification (Synchronous Mode) ====================
// Trigger VLM verification when approaching target to validate detection
constexpr double VLM_APPROACH_TRIGGER_DISTANCE = 0.6;   ///< Distance to trigger VLM verification
constexpr double VLM_EARLY_TRIGGER_DISTANCE = 2.0;      ///< Max distance for early trigger
constexpr int VLM_EARLY_TRIGGER_MIN_FRAMES = 3;         ///< Min high-confidence frames for early trigger
constexpr double VLM_EARLY_TRIGGER_CONFIDENCE = 0.5;    ///< Min confidence for early trigger frames
constexpr double VLM_TIMEOUT_SECONDS = 30.0;            ///< VLM timeout (disable for episode if exceeded)

// ==================== VLM Supplementary Collection Mode ====================
// When VLM verification is triggered but not enough frames collected, robot rotates in place
// to collect more verification images before calling VLM API
constexpr int VLM_MIN_FRAMES_FOR_VERIFY = 3;           ///< Minimum frames required before VLM verification
constexpr int VLM_SUPPLEMENT_MAX_ROTATIONS = 12;       ///< Max rotation steps (360° / 30° = 12 steps)
constexpr double VLM_SUPPLEMENT_ROTATION_ANGLE = M_PI / 6.0;  ///< 30 degrees per rotation step
}  // namespace FSMConstants

class FastPlannerManager;
class ExplorationManager;
class PlanningVisualization;
struct FSMParam;
struct FSMData;

enum ROS_STATE { INIT, WAIT_TRIGGER, PLAN_ACTION, WAIT_ACTION_FINISH, PUB_ACTION, FINISH };
enum ACTION { STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, TURN_DOWN, TURN_UP };
enum HABITAT_STATE { READY, ACTION_EXEC, ACTION_FINISH, EPISODE_FINISH };
enum FINAL_RESULT { EXPLORE, SEARCH_OBJECT, STUCKING, NO_FRONTIER, REACH_OBJECT };

// ==================== Final Approach Validation State Machine ====================
// State machine for final approach validation with look-around scan
enum class FinalApproachState {
  IDLE,              ///< Not in final approach validation
  ROTATING_TO_OBJ,   ///< Rotating to face the target object
  LOOK_UP,           ///< Looking up to scan for object
  WAIT_LOOK_UP,      ///< Waiting for detection after looking up
  LOOK_DOWN,         ///< Looking down to scan for object
  WAIT_LOOK_DOWN,    ///< Waiting for detection after looking down
  LOOK_CENTER,       ///< Returning to center view
  WAIT_LOOK_CENTER,  ///< Waiting for detection after centering
  ADJUST_LEFT,       ///< Micro-adjusting yaw to the left
  WAIT_ADJUST_LEFT,  ///< Waiting for detection after left adjustment
  ADJUST_RIGHT,      ///< Micro-adjusting yaw to the right (from center, so +30° from left)
  WAIT_ADJUST_RIGHT, ///< Waiting for detection after right adjustment
  ADJUST_CENTER,     ///< Returning to center yaw after adjustments
  RESET_PITCH,       ///< Resetting pitch to center before completing (success case)
  WAIT_RESET_PITCH,  ///< Waiting for pitch reset to complete
  COMPLETED          ///< Validation completed (success or failure determined)
};
class ExplorationFSM {
private:
  /* Planning Utils */
  ros::NodeHandle nh_;
  shared_ptr<FastPlannerManager> planner_manager_;
  shared_ptr<ExplorationManager> expl_manager_;
  shared_ptr<PlanningVisualization> visualization_;

  shared_ptr<FSMParam> fp_;
  shared_ptr<FSMData> fd_;
  ROS_STATE state_;

  // Performance monitoring
  bool frontier_processing_;  // Prevent frontierCallback re-entrance

  /* ROS Utils */
  ros::NodeHandle node_;
  ros::Timer exec_timer_, vis_timer_, frontier_timer_;
  ros::Subscriber trigger_sub_, odom_sub_, habitat_state_sub_, confidence_threshold_sub_;
  ros::Subscriber episode_reset_sub_;   // Subscriber for episode reset signal
  ros::Subscriber detector_sub_;        // Subscriber for detector per-frame results
  ros::Subscriber target_category_sub_; // Subscriber for target category (Patience-Aware Navigation)
  ros::Publisher action_pub_, ros_state_pub_, expl_state_pub_, expl_result_pub_;
  ros::Publisher robot_marker_pub_;

  // VLM Approach Verification Publishers/Subscribers
  ros::Publisher vlm_request_pub_;           ///< Publish VLM verification request
  ros::Publisher object_update_pub_;         ///< Publish object update info for frame collection
  ros::Publisher vlm_target_switch_pub_;     ///< Publish target switch notification to Python
  ros::Subscriber vlm_result_sub_;           ///< Subscribe VLM verification result
  int last_published_target_id_;             ///< Track last published target ID to avoid duplicate notifications

  /* VLM Validation */
  ros::ServiceClient vlm_validation_client_;
  bool vlm_validation_enabled_;
  double vlm_validation_timeout_;
  int vlm_num_views_;          ///< Number of views for multi-view validation (default: 2)
  double vlm_camera_hfov_;     ///< Camera horizontal FOV in degrees (default: 79)

  // Multi-view validation state machine
  enum class VLMValidationState {
    IDLE,              ///< Not in validation mode
    NAVIGATING,        ///< Navigating to observation viewpoint
    ROTATING,          ///< Rotating to face object
    CAPTURING,         ///< Capturing image for VLM
    WAITING_RESPONSE,  ///< Waiting for VLM response
    COMPLETED          ///< Validation completed
  };
  VLMValidationState vlm_state_;
  int vlm_current_view_idx_;                    ///< Current view index (0-based)
  int vlm_views_confirmed_;                     ///< Number of views that confirmed object
  std::vector<std::pair<bool, double>> vlm_view_results_;  ///< Results for each view (is_valid, confidence)
  Vector2d vlm_target_viewpoint_;               ///< Current target viewpoint position
  double vlm_target_yaw_;                       ///< Current target yaw for facing object

  // Detector callback state
  bool last_frame_has_target_;                 ///< True if last detector message contained target
  ros::Time last_detection_time_;              ///< Timestamp of last detector message containing target
  boost::mutex detector_mutex_;                ///< Protects detector state

  // ==================== Approach Detection Check (Object Navigation) ====================
  // Track whether target was detected in recent steps using a sliding window
  // Validates that target is visible before confirming arrival
  std::deque<bool> approach_detection_history_;  ///< Sliding window of detection results (last N steps)

  /**
   * @brief Update approach detection check state
   * Called each step during SEARCH_OBJECT mode to track target visibility
   * @param dist_to_target Current distance to target object
   */
  void updateApproachDetectionCheck(double dist_to_target);

  /**
   * @brief Check if approach detection is valid (target was seen in recent steps)
   * @return true if target was detected at least once in the sliding window
   */
  bool isApproachDetectionValid() const;

  /**
   * @brief Reset approach detection state (for new target or episode)
   */
  void resetApproachDetectionCheck();

  /**
   * @brief Reset soft arrival tracking state
   * Called when target changes, mode changes, or distance exceeds soft arrival range
   */
  void resetSoftArrivalTracking();

  // ==================== Final Approach Validation (Look-Around Scan) ====================
  // When reaching target, performs: 1) rotate to face object, 2) look up/down scan,
  // 3) optional left/right micro-adjustment if no detection
  // Uses OR logic with sliding window: success if EITHER scan detects OR window has detection
  FinalApproachState final_approach_state_;         ///< Current state of final approach validation
  bool final_approach_scan_detected_;               ///< True if target detected during scan
  ros::Time final_approach_scan_start_time_;        ///< Start time of current scan wait
  double final_approach_target_yaw_;                ///< Target yaw to face the object
  int final_approach_adjustment_count_;             ///< Number of yaw adjustments made
  int final_approach_pitch_offset_;                 ///< Camera pitch offset: 0=center, +1=up, -1=down

  /**
   * @brief Process final approach validation state machine
   * Called when reaching target (dist < reach_distance) in SEARCH_OBJECT mode
   * Performs look-around scan and combines with sliding window using OR logic
   * @param current_yaw Current robot yaw angle
   * @return -1 if validation in progress, REACH_OBJECT if success, -2 if failed
   */
  int processFinalApproachValidation(double current_yaw);

  /**
   * @brief Start final approach validation
   * Computes target yaw to face object and initializes state machine
   * @return true if validation started successfully
   */
  bool startFinalApproachValidation();

  /**
   * @brief Reset final approach validation state
   */
  void resetFinalApproachValidation();

  /**
   * @brief Check if target was detected during scan (checks detector state)
   * @return true if target detected within timeout window
   */
  bool checkScanDetection() const;

  // ==================== VLM Sliding Window Validation ====================

  /// Candidate frame structure for sliding window
  struct VLMCandidateFrame {
    ros::Time timestamp;              ///< Timestamp for image retrieval from Python cache
    double detection_confidence;      ///< Detector confidence score
    double distance_to_target;        ///< Distance to target object when captured
    Eigen::Vector2d robot_position;   ///< Robot position when captured
    double robot_yaw;                 ///< Robot orientation when captured
    int detected_object_id;           ///< Object ID from object_map
  };

  /// Sliding window state
  std::deque<VLMCandidateFrame> vlm_candidate_frames_;  ///< Candidate frames buffer

  /// Collection phase lock state (locks target when entering collection distance)
  bool vlm_collection_locked_;                 ///< True when target is locked for collection
  int vlm_locked_object_id_;                   ///< Locked target object ID
  Eigen::Vector2d vlm_locked_target_pos_;      ///< Locked target position

  /// VLM validation result lock (locks target after VLM confirms)
  bool vlm_target_confirmed_;                  ///< True after VLM validation passes

  /// Sliding window management functions
  void collectVLMCandidateFrame();
  void clearVLMCandidateFrames();
  bool selectBestFramesForValidation(std::vector<VLMCandidateFrame>& selected);
  bool isDetectionFromTargetObject();
  bool performSlidingWindowVLMValidation(const std::vector<VLMCandidateFrame>& frames);

  /// Collection phase lock management
  void enterCollectionPhase(int object_id, const Eigen::Vector2d& target_pos);
  void exitCollectionPhase(bool success);
  bool isInCollectionPhase() const { return vlm_collection_locked_; }

  /**
   * @brief Call VLM validation service to verify target object (single view, legacy)
   * @return true if VLM confirms target object is present, false otherwise
   */
  bool callVLMValidation();

  /**
   * @brief Start multi-view VLM validation process
   * Computes observation viewpoints and initiates navigation to first viewpoint
   * @return true if validation process started successfully
   */
  bool startMultiViewValidation();

  /**
   * @brief Process one step of multi-view validation state machine
   * Called during SEARCH_OBJECT state when close to target
   * @return FINAL_RESULT indicating validation outcome or need to continue
   */
  int processMultiViewValidation();

  /**
   * @brief Capture current view and call VLM service
   * @return true if VLM confirms object in current view
   */
  bool captureAndValidateCurrentView();

  /**
   * @brief Compute action to rotate robot to face target yaw
   * @param current_yaw Current robot yaw
   * @param target_yaw Target yaw to face
   * @return ACTION to take (TURN_LEFT, TURN_RIGHT, or STOP if aligned)
   */
  int computeRotationAction(double current_yaw, double target_yaw);

  /* Action Planner */
  int callActionPlanner();
  int planNextBestAction(Vector2d current_pos, double current_yaw, const vector<Vector2d>& path,
      bool need_safety = true);
  Vector2d selectLocalTarget(
      const Vector2d& current_pos, const vector<Vector2d>& path, const double& local_distance);
  int decideNextAction(double current_yaw, double target_yaw);
  Vector2d computeBestStep(
      const Vector2d& current_pos, double current_yaw, const Vector2d& target_pos);
  double computeActionSafetyCost(const Vector2d& current_pos, const Vector2d& step);
  double computeActionTotalCost(const Vector2d& current_pos, double current_yaw,
      const Vector2d& target_pos, const Vector2d& step);

  /* Helper functions */
  bool updateFrontierAndObject();
  void transitState(ROS_STATE new_state, string pos_call);
  void wrapAngle(double& angle);
  void publishRobotMarker();
  void visualize();
  void clearVisMarker();

  /* ROS callbacks */
  void FSMCallback(const ros::TimerEvent& e);
  void frontierCallback(const ros::TimerEvent& e);
  void triggerCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  void odometryCallback(const nav_msgs::OdometryConstPtr& msg);
  void habitatStateCallback(const std_msgs::Int32ConstPtr& msg);
  void confidenceThresholdCallback(const std_msgs::Float64ConstPtr& msg);
  void episodeResetCallback(const std_msgs::EmptyConstPtr& msg);  // Reset frontiers for new episode
  void targetCategoryCallback(const std_msgs::StringConstPtr& msg);  // Set target category for Patience-Aware Navigation
  void detectorCallback(const plan_env::MultipleMasksWithConfidenceConstPtr& msg);  // Detector per-frame results

  // ==================== VLM Approach Verification ====================

  /**
   * @brief Callback for VLM verification result from Python
   * Applies VLM result to object confidence and releases FSM lock
   * @param msg VLM verification result message
   */
  void vlmVerificationResultCallback(const plan_env::VLMVerificationResultConstPtr& msg);

  /**
   * @brief Trigger VLM verification request to Python
   * Sets vlm_waiting_ to true and FSM will stop outputting actions
   * @param object_id Target object ID
   * @param trigger_type 0=distance, 1=early, 2=rescue
   * @param distance_to_target Current distance to target
   */
  void triggerVLMVerification(int object_id, int trigger_type, double distance_to_target);

  /**
   * @brief Publish ObjectUpdateInfo when detection updates target object
   * Python uses this to collect frames only from target object updates
   * @param object_id Updated object ID
   * @param confidence Detection confidence
   * @param distance Distance to object
   * @param is_current_target Whether this is the navigation target
   */
  void publishObjectUpdateInfo(int object_id, double confidence, double distance, bool is_current_target);

  /**
   * @brief Check and handle VLM trigger conditions during object navigation
   * Called in callActionPlanner during SEARCH_OBJECT mode
   * @param dist_to_target Current distance to target
   * @param object_id Current target object ID
   * @return true if VLM verification was triggered (FSM should stop)
   */
  bool checkVLMTriggerConditions(double dist_to_target, int object_id);

  /**
   * @brief Reset VLM verification state for new episode
   */
  void resetVLMVerificationState();

public:
  ExplorationFSM() = default;
  ~ExplorationFSM() = default;

  void init(ros::NodeHandle& nh);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline void ExplorationFSM::wrapAngle(double& angle)
{
  while (angle < -M_PI) angle += 2 * M_PI;
  while (angle > M_PI) angle -= 2 * M_PI;
}
}  // namespace apexnav_planner

#endif