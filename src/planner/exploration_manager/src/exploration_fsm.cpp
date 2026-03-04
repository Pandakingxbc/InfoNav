
#include <exploration_manager/exploration_manager.h>
#include <exploration_manager/exploration_fsm.h>
#include <exploration_manager/exploration_data.h>
#include <vis_utils/planning_visualization.h>
#include <plan_env/multi_source_value_map.h>
#include <plan_env/map_ros.h>

#include <limits>  // For std::numeric_limits in episodeResetCallback

// VLM Validation Service
#include <plan_env/ValidateObject.h>
#include <plan_env/MultipleMasksWithConfidence.h>

namespace apexnav_planner {
void ExplorationFSM::init(ros::NodeHandle& nh)
{
  nh_ = nh;
  fp_.reset(new FSMParam);
  fd_.reset(new FSMData);

  // Initialize performance monitoring
  frontier_processing_ = false;

  /* Initialize main modules */
  expl_manager_.reset(new ExplorationManager);
  expl_manager_->initialize(nh);
  visualization_.reset(new PlanningVisualization(nh));
  fp_->vis_scale_ = expl_manager_->sdf_map_->getResolution() * FSMConstants::VIS_SCALE_FACTOR;

  state_ = ROS_STATE::INIT;

  /* ROS Timer */
  exec_timer_ = nh.createTimer(
      ros::Duration(FSMConstants::EXEC_TIMER_DURATION), &ExplorationFSM::FSMCallback, this);
  frontier_timer_ = nh.createTimer(ros::Duration(FSMConstants::FRONTIER_TIMER_DURATION),
      &ExplorationFSM::frontierCallback, this);

  /* ROS Subscriber */
  trigger_sub_ = nh.subscribe("/move_base_simple/goal", 10, &ExplorationFSM::triggerCallback, this);
  odom_sub_ = nh.subscribe("/odom_world", 10, &ExplorationFSM::odometryCallback, this);
  habitat_state_sub_ =
      nh.subscribe("/habitat/state", 10, &ExplorationFSM::habitatStateCallback, this);
  confidence_threshold_sub_ = node_.subscribe(
      "/detector/confidence_threshold", 10, &ExplorationFSM::confidenceThresholdCallback, this);
    // Subscribe to detector per-frame outputs to know when the detector actually sees the object
    detector_sub_ = node_.subscribe(
      "/detector/clouds_with_scores", 5, &ExplorationFSM::detectorCallback, this);
  episode_reset_sub_ = nh.subscribe(
      "/habitat/episode_reset", 10, &ExplorationFSM::episodeResetCallback, this);

  /* ROS Publisher */
  ros_state_pub_ = nh.advertise<std_msgs::Int32>("/ros/state", 10);
  expl_state_pub_ = nh.advertise<std_msgs::Int32>("/ros/expl_state", 10);
  action_pub_ = nh.advertise<std_msgs::Int32>("/habitat/plan_action", 10);
  expl_result_pub_ = nh.advertise<std_msgs::Int32>("/ros/expl_result", 10);
  robot_marker_pub_ = nh.advertise<visualization_msgs::Marker>("/robot", 10);

  /* VLM Validation */
  nh.param("vlm_validation/enabled", vlm_validation_enabled_, false);
  nh.param("vlm_validation/timeout", vlm_validation_timeout_, 30.0);
  nh.param("vlm_validation/num_views", vlm_num_views_, 2);  // Default: 2 views for AND logic
  nh.param("vlm_validation/camera_hfov", vlm_camera_hfov_, 79.0);  // Default: 79 degrees

  // Initialize multi-view validation state
  vlm_state_ = VLMValidationState::IDLE;
  vlm_current_view_idx_ = 0;
  vlm_views_confirmed_ = 0;
  vlm_view_results_.clear();

  if (vlm_validation_enabled_) {
    vlm_validation_client_ = nh.serviceClient<plan_env::ValidateObject>("/vlm/validate_object");
    ROS_INFO("[FSM] VLM validation enabled: timeout=%.1fs, num_views=%d, hfov=%.1f°",
             vlm_validation_timeout_, vlm_num_views_, vlm_camera_hfov_);
  } else {
    ROS_INFO("[FSM] VLM validation disabled");
  }

  // Initialize detector flags
  last_frame_has_target_ = false;
  last_detection_time_ = ros::Time(0);

  // Initialize approach detection check state (sliding window)
  approach_detection_history_.clear();

  // Initialize final approach validation state
  final_approach_state_ = FinalApproachState::IDLE;
  final_approach_scan_detected_ = false;
  final_approach_scan_start_time_ = ros::Time(0);
  final_approach_target_yaw_ = 0.0;
  final_approach_adjustment_count_ = 0;

  // Initialize VLM sliding window state
  vlm_candidate_frames_.clear();
  vlm_collection_locked_ = false;
  vlm_locked_object_id_ = -1;
  vlm_locked_target_pos_ = Eigen::Vector2d::Zero();
  vlm_target_confirmed_ = false;

  // ==================== VLM Approach Verification ====================
  // Publishers for VLM verification request and object update info
  vlm_request_pub_ = nh.advertise<plan_env::VLMVerificationRequest>("/vlm/verification_request", 10);
  object_update_pub_ = nh.advertise<plan_env::ObjectUpdateInfo>("/vlm/object_update_info", 10);
  vlm_target_switch_pub_ = nh.advertise<std_msgs::Int32>("/vlm/target_switch", 10);
  last_published_target_id_ = -1;

  // Subscriber for VLM verification result from Python
  vlm_result_sub_ = nh.subscribe("/vlm/verification_result", 10,
                                  &ExplorationFSM::vlmVerificationResultCallback, this);

  ROS_INFO("[FSM] VLM Approach Verification initialized");
}

// FSM between ROS and Habitat for action planning and execution
void ExplorationFSM::FSMCallback(const ros::TimerEvent& e)
{
  exec_timer_.stop();
  std_msgs::Int32 ros_state_msg;
  ros_state_msg.data = state_;
  ros_state_pub_.publish(ros_state_msg);
  switch (state_) {
    case ROS_STATE::INIT: {
      // Wait for odometry and target confidence threshold
      if (!fd_->have_odom_ || !fd_->have_confidence_) {
        ROS_WARN_THROTTLE(1.0, "No odom || No target confidence threshold.");
        exec_timer_.start();
        return;
      }
      // Go to WAIT_TRIGGER when prerequisites are ready
      clearVisMarker();
      transitState(ROS_STATE::WAIT_TRIGGER, "FSM");
      break;
    }

    case ROS_STATE::WAIT_TRIGGER: {
      // Do nothing but wait for trigger
      ROS_WARN_THROTTLE(1.0, "Wait for trigger.");
      break;
    }

    case ROS_STATE::FINISH: {
      if (!fd_->have_finished_) {
        fd_->have_finished_ = true;
        clearVisMarker();
        std_msgs::Int32 action_msg;
        action_msg.data = ACTION::STOP;
        action_pub_.publish(action_msg);
      }
      ROS_WARN_THROTTLE(1.0, "Finish One Episode!!!");
      break;
    }

    case ROS_STATE::PLAN_ACTION: {
      // ==================== VLM Waiting State Check ====================
      // If waiting for VLM verification result, do not output any action
      if (fd_->vlm_waiting_) {
        ROS_DEBUG_THROTTLE(1.0, "[VLM] Waiting for verification result, no action output");
        exec_timer_.start();
        return;  // Do not publish any action, wait for VLM result
      }

      // Initial action sequence: perform orientation calibration turns
      if (fd_->init_action_count_ < 1 + 12 + 1 + 12) {
        if (fd_->init_action_count_ < 1)
          fd_->newest_action_ = ACTION::TURN_DOWN;
        else if (fd_->init_action_count_ < 1 + 12)
          fd_->newest_action_ = ACTION::TURN_LEFT;
        else if (fd_->init_action_count_ < 1 + 12 + 1)
          fd_->newest_action_ = ACTION::TURN_UP;
        else
          fd_->newest_action_ = ACTION::TURN_LEFT;
        ROS_WARN("Init Mode Process -----> (%d/26)", fd_->init_action_count_);
        fd_->init_action_count_++;
        transitState(ROS_STATE::PUB_ACTION, "FSM");
        updateFrontierAndObject();
      }
      else {
        // Main planning phase: determine robot pose and call action planner
        fd_->start_pt_ = fd_->odom_pos_;
        fd_->start_yaw_(0) = fd_->odom_yaw_;

        auto t1 = ros::Time::now();
        fd_->final_result_ = callActionPlanner();
        double call_action_planner_time = (ros::Time::now() - t1).toSec();
        ROS_INFO_THROTTLE(
            10.0, "[Calculating Time] Planning process time = %.3f s", call_action_planner_time);

        std_msgs::Int32 expl_state_msg;
        expl_state_msg.data = fd_->final_result_;
        expl_state_pub_.publish(expl_state_msg);
        if (fd_->final_result_ == FINAL_RESULT::EXPLORE ||
            fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT)
          transitState(ROS_STATE::PUB_ACTION, "FSM");
        else
          transitState(ROS_STATE::FINISH, "FSM");
      }
      visualize();
      break;
    }

    case ROS_STATE::PUB_ACTION: {
      std_msgs::Int32 action_msg;
      action_msg.data = fd_->newest_action_;
      action_pub_.publish(action_msg);
      transitState(ROS_STATE::WAIT_ACTION_FINISH, "FSM");
      break;
    }

    case ROS_STATE::WAIT_ACTION_FINISH: {
      exec_timer_.start();
      break;
    }
  }
  exec_timer_.start();
}

/**
 * @brief Plan the next action based on current state and environment
 * @return Final result indicating the planned action type and exploration state
 *
 * This is the core planning function that decides what action the robot should take next.
 * It handles obstacle avoidance, frontier exploration, object search, and stuck recovery.
 */
int ExplorationFSM::callActionPlanner()
{
  const double stucking_distance = FSMConstants::STUCKING_DISTANCE;
  const double reach_distance = FSMConstants::REACH_DISTANCE;
  const double soft_reach_distance = FSMConstants::SOFT_REACH_DISTANCE;
  const double vlm_trigger_distance = FSMConstants::VLM_TRIGGER_DISTANCE;
  const double vlm_collect_distance = FSMConstants::VLM_COLLECT_DISTANCE;

  bool frontier_change_flag = updateFrontierAndObject();

  // ==================== Sync current target to MapROS for VLM frame collection ====================
  // MapROS needs to know the current target object ID to publish ObjectUpdateInfo only for it
  {
    int current_obj_id = expl_manager_->object_map2d_->getCurrentTargetObjectId();
    if (expl_manager_->sdf_map_) {
      expl_manager_->sdf_map_->setCurrentNavTargetObjectId(current_obj_id);
    }

    // Notify Python VLM verifier about target switch to clear navigation cache
    // This prevents mixing frames from different targets
    if (current_obj_id != last_published_target_id_) {
      std_msgs::Int32 switch_msg;
      switch_msg.data = current_obj_id;  // -1 means frontier exploration (no specific target)
      vlm_target_switch_pub_.publish(switch_msg);

      if (current_obj_id == -1) {
        ROS_INFO("[VLM] Target switch notification: entering frontier exploration");
      } else if (last_published_target_id_ == -1) {
        ROS_INFO("[VLM] Target switch notification: new target object %d", current_obj_id);
      } else {
        ROS_INFO("[VLM] Target switch notification: %d -> %d", last_published_target_id_, current_obj_id);
      }
      last_published_target_id_ = current_obj_id;
    }
  }

  // ==================== DEBUG: Diagnose stuck agent ====================
  Eigen::Vector2d debug_current_pos = Eigen::Vector2d(fd_->start_pt_(0), fd_->start_pt_(1));
  double debug_current_yaw = fd_->start_yaw_(0);
  ROS_WARN_THROTTLE(5.0, "[DEBUG STUCK] ======== Agent Status ========");
  ROS_WARN_THROTTLE(5.0, "[DEBUG STUCK] Position: (%.2f, %.2f), Yaw: %.2f deg",
                    debug_current_pos.x(), debug_current_pos.y(), debug_current_yaw * 180.0 / M_PI);
  ROS_WARN_THROTTLE(5.0, "[DEBUG STUCK] VLM collection_locked: %s, locked_obj_id: %d",
                    vlm_collection_locked_ ? "TRUE" : "FALSE", vlm_locked_object_id_);
  if (vlm_collection_locked_) {
    double dist_to_locked = (debug_current_pos - vlm_locked_target_pos_).norm();
    ROS_WARN_THROTTLE(5.0, "[DEBUG STUCK] Locked target: (%.2f, %.2f), dist: %.2fm",
                      vlm_locked_target_pos_.x(), vlm_locked_target_pos_.y(), dist_to_locked);
  }
  ROS_WARN_THROTTLE(5.0, "[DEBUG STUCK] Last action: %d, final_result: %d, replan_flag: %s",
                    fd_->newest_action_, fd_->final_result_, fd_->replan_flag_ ? "TRUE" : "FALSE");
  ROS_WARN_THROTTLE(5.0, "[DEBUG STUCK] Next target pos: (%.2f, %.2f), dist: %.2fm",
                    expl_manager_->ed_->next_pos_.x(), expl_manager_->ed_->next_pos_.y(),
                    (debug_current_pos - expl_manager_->ed_->next_pos_).norm());
  ROS_WARN_THROTTLE(5.0, "[DEBUG STUCK] Path size: %zu, escape_stucking: %s, stucking_count: %d",
                    expl_manager_->ed_->next_best_path_.size(),
                    fd_->escape_stucking_flag_ ? "TRUE" : "FALSE", fd_->stucking_action_count_);
  ROS_WARN_THROTTLE(5.0, "[DEBUG STUCK] ==============================");

  int expl_res, final_res;
  Eigen::Vector2d current_pos = Eigen::Vector2d(fd_->start_pt_(0), fd_->start_pt_(1));
  Eigen::Vector2d last_pos = Eigen::Vector2d(fd_->last_start_pos_(0), fd_->last_start_pos_(1));
  double current_yaw = fd_->start_yaw_(0);
  fd_->last_start_pos_ = fd_->start_pt_;

  // ==================== VLM Sliding Window Logic ====================
  // Handle collection phase lock: once locked, use locked target position
  if (vlm_collection_locked_) {
    double dist_to_locked = (current_pos - vlm_locked_target_pos_).norm();

    // Case 1: Reached the locked target (within reach distance)
    if (dist_to_locked < reach_distance) {
      ROS_INFO("[VLM Sliding] Reached locked target, triggering validation (dist=%.2fm)", dist_to_locked);

      // Perform VLM validation with collected frames
      std::vector<VLMCandidateFrame> selected_frames;
      bool has_frames = selectBestFramesForValidation(selected_frames);

      bool vlm_result = false;
      if (has_frames) {
        vlm_result = performSlidingWindowVLMValidation(selected_frames);
      } else {
        ROS_WARN("[VLM Sliding] No candidate frames, using current frame");
        vlm_result = callVLMValidation();
      }

      if (vlm_result) {
        // VLM passed - exit collection phase successfully
        exitCollectionPhase(true);
        if (expl_manager_->isSuspiciousTargetLocked()) {
          expl_manager_->setSuspiciousTargetLock(false);
        }
        ROS_INFO("[VLM Sliding] Validation PASSED! Target confirmed.");
        return FINAL_RESULT::REACH_OBJECT;
      } else {
        // VLM failed - exit collection phase, mark invalid, replan
        exitCollectionPhase(false);
        if (expl_manager_->isSuspiciousTargetLocked()) {
          expl_manager_->setSuspiciousTargetLock(false);
        }
        fd_->replan_flag_ = true;
        ROS_WARN("[VLM Sliding] Validation FAILED! Replanning...");
        // Continue to normal planning below
      }
    }
    // Case 2: Within trigger distance but not yet reached
    else if (dist_to_locked <= vlm_trigger_distance) {
      ROS_INFO("[VLM Sliding] Near trigger distance (%.2fm), collecting frames", dist_to_locked);
      collectVLMCandidateFrame();
      // Force use locked target position, skip normal planning for target selection
      expl_manager_->ed_->next_pos_ = vlm_locked_target_pos_;
    }
    // Case 3: Still in collection phase (between collect and trigger distance)
    else if (dist_to_locked <= vlm_collect_distance) {
      collectVLMCandidateFrame();
      // Force use locked target position
      expl_manager_->ed_->next_pos_ = vlm_locked_target_pos_;
    }
    // Case 4: Somehow got farther than collect distance while locked (shouldn't happen normally)
    else {
      ROS_WARN("[VLM Sliding] Distance increased beyond collect range (%.2fm > %.2fm), maintaining lock",
               dist_to_locked, vlm_collect_distance);
      // Still maintain lock but don't collect
      expl_manager_->ed_->next_pos_ = vlm_locked_target_pos_;
    }
  }
  // Note: entering collection phase is handled AFTER planNextBestPoint (see below)
  // because we need the current expl_res to know if we're in SEARCH_OBJECT state

  // ==================== Approach Detection Check (Object Navigation) ====================
  // Track target detection during approach phase (0.6m - 3.0m from target)
  // This ensures we can actually see the target before confirming success
  double dist_to_target = (current_pos - expl_manager_->ed_->next_pos_).norm();
  if (fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT) {
    updateApproachDetectionCheck(dist_to_target);
  }

  // ==================== Soft Arrival Detection (Object Navigation) ====================
  // Detects oscillation (forward-backward) when agent cannot get closer to target
  // If within soft arrival range and best distance doesn't improve for several steps, consider arrived
  const double soft_arrival_distance = FSMConstants::SOFT_ARRIVAL_DISTANCE;
  const double soft_arrival_improvement_thresh = FSMConstants::SOFT_ARRIVAL_IMPROVEMENT_THRESH;
  const int soft_arrival_max_no_improve = FSMConstants::SOFT_ARRIVAL_MAX_NO_IMPROVE;

  if (!vlm_collection_locked_ &&
      fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
      dist_to_target < soft_arrival_distance &&
      dist_to_target >= reach_distance) {
    // Within soft arrival range - track progress
    fd_->approach_attempt_count_++;

    if (dist_to_target < fd_->approach_best_dist_ - soft_arrival_improvement_thresh) {
      // Made meaningful progress - update best distance and reset no-improvement counter
      fd_->approach_best_dist_ = dist_to_target;
      fd_->approach_no_improvement_count_ = 0;
      ROS_INFO_THROTTLE(2.0, "[Soft Arrival] Progress! dist=%.2fm (best=%.2fm), attempts=%d",
                        dist_to_target, fd_->approach_best_dist_, fd_->approach_attempt_count_);
    } else {
      // No meaningful progress
      fd_->approach_no_improvement_count_++;
      ROS_WARN_THROTTLE(2.0, "[Soft Arrival] No progress: dist=%.2fm, best=%.2fm, no_improve=%d/%d",
                        dist_to_target, fd_->approach_best_dist_,
                        fd_->approach_no_improvement_count_, soft_arrival_max_no_improve);

      // Check if we should trigger soft arrival
      if (fd_->approach_no_improvement_count_ >= soft_arrival_max_no_improve) {
        ROS_WARN("[Soft Arrival] Triggered! Agent oscillating at dist=%.2fm (best=%.2fm) after %d attempts",
                 dist_to_target, fd_->approach_best_dist_, fd_->approach_attempt_count_);

        // Check approach detection before confirming (same as hard arrival)
        if (!isApproachDetectionValid()) {
          ROS_WARN("[Soft Arrival] Approach detection FAILED - target not seen during approach");
          int current_obj_id = expl_manager_->object_map2d_->getCurrentTargetObjectId();
          if (current_obj_id >= 0) {
            expl_manager_->object_map2d_->addFailedApproachPoint(
                current_obj_id, expl_manager_->ed_->next_pos_);
          }
          if (expl_manager_->isSuspiciousTargetLocked()) {
            expl_manager_->setSuspiciousTargetLock(false);
          }
          if (expl_manager_->isObjectApproachLocked()) {
            expl_manager_->setObjectApproachLock(false);
          }
          resetApproachDetectionCheck();
          resetSoftArrivalTracking();
          fd_->replan_flag_ = true;
        } else {
          // Use VLM Approach Verifier for soft arrival validation
          int current_obj_id = expl_manager_->object_map2d_->getCurrentTargetObjectId();
          if (current_obj_id >= 0 && !fd_->vlm_disabled_this_episode_) {
            // Check if VLM already verified this object
            bool vlm_verified = expl_manager_->sdf_map_->object_map2d_->isObjectVLMVerified(current_obj_id);
            if (vlm_verified) {
              bool vlm_passed = expl_manager_->sdf_map_->object_map2d_->getObjectVLMResult(current_obj_id);
              bool vlm_uncertain = expl_manager_->sdf_map_->object_map2d_->isObjectVLMUncertain(current_obj_id);
              if (vlm_passed || vlm_uncertain) {
                ROS_INFO("[Soft Arrival] VLM already verified PASSED/UNCERTAIN at dist=%.2fm!", dist_to_target);
                if (expl_manager_->isSuspiciousTargetLocked()) {
                  expl_manager_->setSuspiciousTargetLock(false);
                }
                if (expl_manager_->isObjectApproachLocked()) {
                  expl_manager_->setObjectApproachLock(false);
                }
                resetApproachDetectionCheck();
                resetSoftArrivalTracking();
                return FINAL_RESULT::REACH_OBJECT;
              } else {
                // VLM rejected - replan
                ROS_WARN("[Soft Arrival] VLM already REJECTED at dist=%.2fm, replanning...", dist_to_target);
                expl_manager_->object_map2d_->markObjectAsInvalid(current_obj_id);
                if (expl_manager_->isSuspiciousTargetLocked()) {
                  expl_manager_->setSuspiciousTargetLock(false);
                }
                if (expl_manager_->isObjectApproachLocked()) {
                  expl_manager_->setObjectApproachLock(false);
                }
                resetApproachDetectionCheck();
                resetSoftArrivalTracking();
                fd_->replan_flag_ = true;
              }
            } else {
              // VLM not verified yet - trigger verification
              if (!fd_->vlm_waiting_) {
                ROS_INFO("[Soft Arrival] Triggering VLM verification at dist=%.2fm", dist_to_target);
                triggerVLMVerification(current_obj_id, 0, dist_to_target);  // type=0: distance trigger
              }
              // Wait for VLM result
              return fd_->final_result_;
            }
          } else if (fd_->vlm_disabled_this_episode_) {
            // VLM disabled this episode - trust detector
            ROS_WARN("[Soft Arrival] VLM disabled, trusting detector at dist=%.2fm", dist_to_target);
            if (expl_manager_->isSuspiciousTargetLocked()) {
              expl_manager_->setSuspiciousTargetLock(false);
            }
            if (expl_manager_->isObjectApproachLocked()) {
              expl_manager_->setObjectApproachLock(false);
            }
            resetApproachDetectionCheck();
            resetSoftArrivalTracking();
            return FINAL_RESULT::REACH_OBJECT;
          }
        }
      }
    }
  } else if (fd_->final_result_ != FINAL_RESULT::SEARCH_OBJECT ||
             dist_to_target >= soft_arrival_distance) {
    // Not in object search mode or outside soft arrival range - reset tracking
    if (fd_->approach_attempt_count_ > 0) {
      ROS_INFO("[Soft Arrival] Reset tracking (mode changed or dist=%.2fm >= %.2fm)",
               dist_to_target, soft_arrival_distance);
    }
    resetSoftArrivalTracking();
  }

  // ==================== VLM Approach Verification Trigger ====================
  // Trigger VLM verification when approaching target object (synchronous mode)
  // This happens BEFORE reach check, so VLM can validate while agent is still approaching
  if (!vlm_collection_locked_ &&
      fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
      dist_to_target < FSMConstants::VLM_APPROACH_TRIGGER_DISTANCE &&
      dist_to_target >= reach_distance) {
    int current_obj_id = expl_manager_->object_map2d_->getCurrentTargetObjectId();
    if (current_obj_id >= 0 && checkVLMTriggerConditions(dist_to_target, current_obj_id)) {
      // Trigger VLM verification - type=0: normal distance trigger
      ROS_INFO("[VLM Approach] Triggered at dist=%.2fm, waiting for result...", dist_to_target);
      triggerVLMVerification(current_obj_id, 0, dist_to_target);
      return fd_->final_result_;  // Continue in SEARCH_OBJECT state, but FSM will block
    }
  }

  // Reach the object - check if close enough to target object
  // Only use VLM verification result, no sliding window detection
  int current_obj_id_for_final = expl_manager_->object_map2d_->getCurrentTargetObjectId();

  // FIX: Only enter Final Approach if we have a valid target object
  // When obj_id=-1, skip this block entirely and let planNextBestPoint() choose a new target
  if (!vlm_collection_locked_ &&
      fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
      dist_to_target < reach_distance &&
      current_obj_id_for_final >= 0) {  // Added: require valid object ID to enter

    int current_obj_id = current_obj_id_for_final;

    // ========== VLM-Only Validation ==========
    // Logic:
    // 1. If VLM verified this object and passed (CONFIRM) -> SUCCESS
    // 2. If VLM verified this object and UNCERTAIN -> treat as SUCCESS (trust detector)
    // 3. If VLM verified this object and REJECTED -> replan to other targets
    // 4. If VLM not verified yet -> TRIGGER VLM verification first (don't auto-succeed)
    // 5. If VLM disabled -> SUCCESS (fallback, trust detector)

    bool vlm_verified = false;
    bool vlm_passed = false;  // Default to false - require VLM verification
    bool vlm_uncertain = false;

    if (current_obj_id >= 0 && !fd_->vlm_disabled_this_episode_) {
      vlm_verified = expl_manager_->sdf_map_->object_map2d_->isObjectVLMVerified(current_obj_id);
      if (vlm_verified) {
        vlm_passed = expl_manager_->sdf_map_->object_map2d_->getObjectVLMResult(current_obj_id);
        vlm_uncertain = expl_manager_->sdf_map_->object_map2d_->isObjectVLMUncertain(current_obj_id);
      } else {
        // VLM not verified yet - trigger verification now (rescue trigger at reach distance)
        if (!fd_->vlm_waiting_ && checkVLMTriggerConditions(dist_to_target, current_obj_id)) {
          ROS_INFO("[Final Approach] VLM not verified, triggering rescue verification at dist=%.2fm", dist_to_target);
          triggerVLMVerification(current_obj_id, 2, dist_to_target);  // type=2: rescue trigger
          return fd_->final_result_;  // Wait for VLM result
        }
        // If already waiting, continue waiting
        if (fd_->vlm_waiting_) {
          ROS_DEBUG_THROTTLE(1.0, "[Final Approach] Waiting for VLM result...");
          return fd_->final_result_;
        }
      }
    } else if (fd_->vlm_disabled_this_episode_) {
      // VLM disabled - trust detector as fallback
      vlm_passed = true;
      ROS_WARN("[Final Approach] VLM disabled, trusting detector");
    }

    ROS_INFO("[Final Approach] VLM validation: obj_id=%d, verified=%s, passed=%s, uncertain=%s",
             current_obj_id, vlm_verified ? "YES" : "NO", vlm_passed ? "YES" : "NO",
             vlm_uncertain ? "YES" : "NO");

    // UNCERTAIN should be treated as success - VLM couldn't determine, trust the detector
    if (vlm_passed || vlm_uncertain) {
      if (vlm_uncertain) {
        ROS_INFO("[Final Approach] SUCCESS! VLM returned UNCERTAIN, trusting detector.");
      } else {
        ROS_INFO("[Final Approach] SUCCESS! VLM validation passed (or not required).");
      }
      if (expl_manager_->isSuspiciousTargetLocked()) {
        expl_manager_->setSuspiciousTargetLock(false);
      }
      if (expl_manager_->isObjectApproachLocked()) {
        expl_manager_->setObjectApproachLock(false);
      }
      resetFinalApproachValidation();
      resetSoftArrivalTracking();
      fd_->vlm_consecutive_rejection_count_ = 0;  // Reset on success
      return FINAL_RESULT::REACH_OBJECT;
    }
    else {
      // VLM explicitly REJECTED this object (not UNCERTAIN)
      fd_->vlm_consecutive_rejection_count_++;
      ROS_WARN("[Final Approach] VLM REJECTED object %d. Consecutive rejections: %d. Replanning...",
               current_obj_id, fd_->vlm_consecutive_rejection_count_);

      if (current_obj_id >= 0) {
        expl_manager_->object_map2d_->markObjectAsInvalid(current_obj_id);
      }

      // Unlock targets
      if (expl_manager_->isSuspiciousTargetLocked()) {
        expl_manager_->setSuspiciousTargetLock(false);
      }
      if (expl_manager_->isObjectApproachLocked()) {
        expl_manager_->setObjectApproachLock(false);
      }

      resetFinalApproachValidation();
      resetSoftArrivalTracking();

      // Fix 3: After N consecutive VLM rejections, force exploration mode
      // This prevents infinite loops when all nearby objects are rejected
      const int MAX_CONSECUTIVE_REJECTIONS = 5;
      if (fd_->vlm_consecutive_rejection_count_ >= MAX_CONSECUTIVE_REJECTIONS) {
        ROS_WARN("[Final Approach] %d consecutive VLM rejections! Marking all target objects as invalid to force frontier exploration.",
                 fd_->vlm_consecutive_rejection_count_);
        fd_->vlm_consecutive_rejection_count_ = 0;  // Reset counter
        // Mark all current top confidence objects as invalid to force frontier exploration
        std::vector<int> object_ids;
        expl_manager_->object_map2d_->getTopConfidenceObjectIds(object_ids, false);
        for (int obj_id : object_ids) {
          expl_manager_->object_map2d_->markObjectAsInvalid(obj_id);
        }
      }

      fd_->replan_flag_ = true;
      return fd_->final_result_;
    }
  }

  // Also unlock suspicious target if we reached the locked position (even if not final destination)
  if (expl_manager_->isSuspiciousTargetLocked()) {
    Vector2d locked_pos = expl_manager_->getLockedSuspiciousPos();
    if ((current_pos - locked_pos).norm() < reach_distance) {
      ROS_WARN("[Target Lock] Reached locked suspicious target, unlocking");
      expl_manager_->setSuspiciousTargetLock(false);
    }
  }

  /*******  Escape-from-stuck logic START *******/
  // Detect if robot is stuck and initiate escape sequence
  int last_action = fd_->newest_action_;
  if (!fd_->escape_stucking_flag_ && (current_pos - last_pos).norm() < stucking_distance &&
      last_action == ACTION::MOVE_FORWARD) {
    if (fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
        (current_pos - expl_manager_->ed_->next_pos_).norm() < soft_reach_distance) {
      // Stuck near target - require VLM verification before confirming success
      int current_obj_id = expl_manager_->object_map2d_->getCurrentTargetObjectId();
      if (current_obj_id >= 0 && !fd_->vlm_disabled_this_episode_) {
        bool vlm_verified = expl_manager_->sdf_map_->object_map2d_->isObjectVLMVerified(current_obj_id);
        if (vlm_verified) {
          bool vlm_passed = expl_manager_->sdf_map_->object_map2d_->getObjectVLMResult(current_obj_id);
          bool vlm_uncertain = expl_manager_->sdf_map_->object_map2d_->isObjectVLMUncertain(current_obj_id);
          if (vlm_passed || vlm_uncertain) {
            ROS_INFO("[Stuck Near Target] VLM verified PASSED/UNCERTAIN, confirming success!");
            resetApproachDetectionCheck();
            resetSoftArrivalTracking();
            final_res = FINAL_RESULT::REACH_OBJECT;
            return final_res;
          } else {
            // VLM rejected - don't confirm, continue escape sequence
            ROS_WARN("[Stuck Near Target] VLM REJECTED, not confirming success");
          }
        } else {
          // VLM not verified yet - trigger verification and wait
          if (!fd_->vlm_waiting_) {
            double dist_to_target = (current_pos - expl_manager_->ed_->next_pos_).norm();
            ROS_INFO("[Stuck Near Target] Triggering VLM verification at dist=%.2fm", dist_to_target);
            triggerVLMVerification(current_obj_id, 0, dist_to_target);
          }
          // Don't return REACH_OBJECT, wait for VLM result
        }
      } else if (fd_->vlm_disabled_this_episode_) {
        // VLM disabled - trust detector
        ROS_WARN("[Stuck Near Target] VLM disabled, trusting detector");
        resetApproachDetectionCheck();
        resetSoftArrivalTracking();
        final_res = FINAL_RESULT::REACH_OBJECT;
        return final_res;
      }
    }

    bool past_stucking_flag = false;
    for (auto stucking_point : fd_->stucking_points_) {
      Vector2d stucking_pos = Vector2d(stucking_point(0), stucking_point(1));
      double stucking_yaw = stucking_point(2);
      if ((stucking_pos - current_pos).norm() < stucking_distance &&
          fabs(stucking_yaw - current_yaw) < FSMConstants::ACTION_ANGLE) {
        past_stucking_flag = true;
        ROS_ERROR("Still stuck at the same place");
        break;
      }
    }
    if (!past_stucking_flag) {
      fd_->escape_stucking_flag_ = true;
      fd_->escape_stucking_count_ = 0;
      fd_->escape_stucking_pos_ = current_pos;
      fd_->escape_stucking_yaw_ = current_yaw;
    }
  }

  if (fd_->escape_stucking_flag_ && (current_pos - last_pos).norm() >= stucking_distance) {
    ROS_ERROR("Escaped from stuck state.");
    fd_->escape_stucking_flag_ = false;
  }

  if (fd_->escape_stucking_flag_) {
    // Determine max escape steps based on current mode
    // Object navigation gets longer escape sequence to avoid premature obstacle marking
    bool is_object_navigation = (fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT);
    int max_escape_steps = is_object_navigation ?
        FSMConstants::ESCAPE_SEQUENCE_OBJECT : FSMConstants::ESCAPE_SEQUENCE_FRONTIER;

    ROS_ERROR("Escaping stuck... (step %d/%d, mode: %s)",
              fd_->escape_stucking_count_, max_escape_steps,
              is_object_navigation ? "OBJECT" : "FRONTIER");

    // Extended escape sequence pattern (cycles for object navigation)
    int step = fd_->escape_stucking_count_ % 10;  // Cycle through pattern
    if (step == 0)
      fd_->newest_action_ = ACTION::TURN_RIGHT;
    else if (step == 1)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else if (step == 2)
      fd_->newest_action_ = ACTION::TURN_RIGHT;
    else if (step == 3)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else if (step == 4)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (step == 5)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (step == 6)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (step == 7)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else if (step == 8)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (step == 9)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;

    // Check if escape sequence exhausted
    if (fd_->escape_stucking_count_ >= max_escape_steps - 1) {
      // Failed to escape after max steps
      ROS_ERROR("Cannot escape stuck state after %d steps (mode: %s).",
                max_escape_steps, is_object_navigation ? "OBJECT" : "FRONTIER");
      fd_->escape_stucking_flag_ = false;

      // Mark obstacles and record stucking point
      expl_manager_->sdf_map_->setForceOccGrid(current_pos);
      double forward_distance = FSMConstants::FORWARD_DISTANCE;
      Eigen::Vector2d forward_pos = fd_->escape_stucking_pos_;
      forward_pos(0) += forward_distance * cos(fd_->escape_stucking_yaw_);
      forward_pos(1) += forward_distance * sin(fd_->escape_stucking_yaw_);
      expl_manager_->sdf_map_->setForceOccGrid(forward_pos);
      forward_distance = FSMConstants::FORWARD_DISTANCE * 2.0;
      forward_pos = fd_->escape_stucking_pos_;
      forward_pos(0) += forward_distance * cos(fd_->escape_stucking_yaw_);
      forward_pos(1) += forward_distance * sin(fd_->escape_stucking_yaw_);
      expl_manager_->sdf_map_->setForceOccGrid(forward_pos);

      fd_->dormant_frontier_flag_ = true;
      Vector3d stucking_point(
          fd_->escape_stucking_pos_(0), fd_->escape_stucking_pos_(1), fd_->escape_stucking_yaw_);
      fd_->stucking_points_.push_back(stucking_point);
    }

    if (fd_->escape_stucking_flag_) {
      fd_->escape_stucking_count_++;
      return fd_->final_result_;
    }
  }

  /*******  Decide whether to replan path (stability heuristic) START *******/
  // Use path stability to reduce oscillation between different frontier targets
  vector<Vector2d> last_next_best_path = expl_manager_->ed_->next_best_path_;
  Vector2d last_next_pos = expl_manager_->ed_->next_pos_;
  if (fd_->dormant_frontier_flag_) {
    fd_->replan_flag_ = true;
    fd_->dormant_frontier_flag_ = false;
  }
  // FIX: Only suppress replan if replan_flag_ is not already set by other sources (e.g., VLM REJECT)
  // Previously this would override replan_flag_=true from VLM rejection, causing the agent to get stuck
  else if (fd_->final_result_ == FINAL_RESULT::EXPLORE && !frontier_change_flag && !fd_->replan_flag_) {
    // Keep replan_flag_ as false only if no other component requested a replan
  }

  // VLM Sliding Window: Skip target replanning if in collection phase
  // Only replan path, not target selection
  if (vlm_collection_locked_) {
    // Force next_pos to be the locked target (in case it was changed)
    Vector2d locked_pos_backup = vlm_locked_target_pos_;

    // Still call planNextBestPoint to get path, but we'll override the target
    expl_res = expl_manager_->planNextBestPoint(fd_->start_pt_, fd_->start_yaw_(0));

    // Restore locked target position - this is the key to preventing target switching
    expl_manager_->ed_->next_pos_ = locked_pos_backup;

    // Force SEARCH_BEST_OBJECT result since we have a locked target
    if (expl_res != EXPL_RESULT::SEARCH_BEST_OBJECT &&
        expl_res != EXPL_RESULT::SEARCH_SUSPICIOUS_OBJECT &&
        expl_res != EXPL_RESULT::SEARCH_EXTREME) {
      // Recalculate path to locked target if needed
      ROS_INFO("[VLM Sliding] Locked target: forcing path recalculation to (%.2f, %.2f)",
               locked_pos_backup.x(), locked_pos_backup.y());
    }
    expl_res = EXPL_RESULT::SEARCH_BEST_OBJECT;  // Override result

    ROS_DEBUG("[VLM Sliding] Collection locked: target fixed at (%.2f, %.2f)",
              vlm_locked_target_pos_.x(), vlm_locked_target_pos_.y());
  } else {
    expl_res = expl_manager_->planNextBestPoint(fd_->start_pt_, fd_->start_yaw_(0));
  }

  if (expl_res != EXPL_RESULT::EXPLORATION) {
    fd_->replan_flag_ = true;
  }
  if (expl_res == EXPL_RESULT::EXPLORATION && !fd_->replan_flag_) {
    expl_manager_->ed_->next_best_path_ = last_next_best_path;
    expl_manager_->ed_->next_pos_ = last_next_pos;
    fd_->replan_flag_ = true;
  }
  /*******  Decide whether to replan path (stability heuristic) END *******/

  // Publish exploration result to monitor
  std_msgs::Int32 expl_result_msg;
  expl_result_msg.data = expl_res;
  expl_result_pub_.publish(expl_result_msg);

  // Determine current high-level state based on exploration results
  if (expl_res == EXPL_RESULT::EXPLORATION)
    final_res = FINAL_RESULT::EXPLORE;
  else if (expl_res == EXPL_RESULT::NO_COVERABLE_FRONTIER ||
           expl_res == EXPL_RESULT::NO_PASSABLE_FRONTIER)
    final_res = FINAL_RESULT::NO_FRONTIER;
  else if (expl_res == EXPL_RESULT::SEARCH_SIMILAR_OBJECT) {
    // Similar object search: reuse SEARCH_OBJECT handling for re-detection attempts
    ROS_WARN("[FSM] Navigating to similar object for target re-detection");
    final_res = FINAL_RESULT::SEARCH_OBJECT;
  }
  else
    final_res = FINAL_RESULT::SEARCH_OBJECT;

  // ==================== Approach Detection: Reset on Mode/Target Change ====================
  // Reset approach detection state when switching from SEARCH_OBJECT to EXPLORE
  // or when target position changes significantly
  if (fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT && final_res != FINAL_RESULT::SEARCH_OBJECT) {
    // Mode changed from SEARCH_OBJECT to something else - reset detection state
    resetApproachDetectionCheck();
    ROS_DEBUG("[Approach Check] Reset due to mode change (SEARCH_OBJECT -> %d)", final_res);
  }
  else if (final_res == FINAL_RESULT::SEARCH_OBJECT &&
           (expl_manager_->ed_->next_pos_ - fd_->last_next_pos_).norm() > 1.0) {
    // Target changed significantly (>1m) - reset detection state for new target
    resetApproachDetectionCheck();
    ROS_DEBUG("[Approach Check] Reset due to target change (dist=%.2fm)",
              (expl_manager_->ed_->next_pos_ - fd_->last_next_pos_).norm());
  }

  // ==================== VLM Sliding Window: Enter Collection Phase ====================
  // This check is done AFTER planNextBestPoint so we have the correct final_res
  if (!vlm_collection_locked_ && vlm_validation_enabled_ &&
      final_res == FINAL_RESULT::SEARCH_OBJECT) {
    double dist_to_target = (current_pos - expl_manager_->ed_->next_pos_).norm();

    if (dist_to_target <= vlm_collect_distance && dist_to_target > reach_distance) {
      // Enter collection phase - lock the current target
      int current_obj_id = expl_manager_->object_map2d_->getCurrentTargetObjectId();
      enterCollectionPhase(current_obj_id, expl_manager_->ed_->next_pos_);
      ROS_INFO("[VLM Sliding] Entered collection phase at dist=%.2fm", dist_to_target);
    }
  }

  if (final_res == FINAL_RESULT::NO_FRONTIER || expl_manager_->ed_->next_best_path_.empty()) {
    ROS_WARN("No (passable) frontier");
    return final_res;
  }

  Eigen::Vector2d end_pos = expl_manager_->ed_->next_pos_;
  Eigen::Vector2d last_end_pos = fd_->last_next_pos_;
  fd_->last_next_pos_ = end_pos;
  double min_dist = (current_pos - end_pos).norm();
  ROS_WARN("To the next point (%.2fm %.2fm), distance = %.2f m", end_pos(0), end_pos(1), min_dist);

  // Handling being stuck while exploring toward a specific frontier
  if (final_res == FINAL_RESULT::EXPLORE) {
    // Force dormant if very close to target but still exploring
    if (min_dist < FSMConstants::FORCE_DORMANT_DISTANCE) {
      ROS_ERROR("Force set dormant frontier.");
      expl_manager_->frontier_map2d_->setForceDormantFrontier(end_pos);
      fd_->dormant_frontier_flag_ = true;
      expl_manager_->resetWTRPHysteresis();  // Allow selecting new frontier
    }

    // Count consecutive times with same target position while stuck
    if ((end_pos - last_end_pos).norm() < 1e-3 &&
        (current_pos - last_pos).norm() < stucking_distance) {
      fd_->stucking_next_pos_count_++;
      ROS_ERROR_COND(fd_->stucking_next_pos_count_ > 8, "stucking_next_pos_count_ = %d",
          fd_->stucking_next_pos_count_);
    }
    else
      fd_->stucking_next_pos_count_ = 0;

    // Mark frontier as dormant if stuck too long with same target
    if (fd_->stucking_next_pos_count_ >= FSMConstants::MAX_STUCKING_NEXT_POS_COUNT) {
      ROS_ERROR("Set dormant frontier.");
      fd_->stucking_action_count_ = 0;
      fd_->stucking_next_pos_count_ = 0;
      expl_manager_->frontier_map2d_->setForceDormantFrontier(end_pos);
      fd_->dormant_frontier_flag_ = true;
      expl_manager_->resetWTRPHysteresis();  // Allow selecting new frontier
    }

    // ==================== No-Progress Detection ====================
    // Detect when agent is moving but not making progress toward frontier
    // This handles cases where depth defects or discrete actions prevent reaching the frontier
    bool frontier_changed = (end_pos - fd_->tracked_frontier_pos_).norm() > FSMConstants::FRONTIER_CHANGE_THRESHOLD;

    if (frontier_changed) {
      // New frontier target - reset tracking
      fd_->tracked_frontier_pos_ = end_pos;
      fd_->initial_dist_to_frontier_ = min_dist;
      fd_->best_dist_to_frontier_ = min_dist;
      fd_->steps_toward_frontier_ = 0;
      ROS_INFO("[No-Progress] New frontier target at (%.2f, %.2f), initial dist = %.2f m",
               end_pos(0), end_pos(1), min_dist);
    } else {
      // Same frontier - update tracking
      fd_->steps_toward_frontier_++;

      // Update best distance achieved
      if (min_dist < fd_->best_dist_to_frontier_) {
        fd_->best_dist_to_frontier_ = min_dist;
      }

      // Periodic progress check
      if (fd_->steps_toward_frontier_ % static_cast<int>(FSMConstants::NO_PROGRESS_CHECK_INTERVAL) == 0) {
        double improvement = fd_->initial_dist_to_frontier_ - fd_->best_dist_to_frontier_;
        ROS_WARN("[No-Progress] Step %d toward frontier, initial=%.2f, best=%.2f, current=%.2f, improvement=%.2f",
                 fd_->steps_toward_frontier_, fd_->initial_dist_to_frontier_,
                 fd_->best_dist_to_frontier_, min_dist, improvement);
      }

      // Check if max steps exceeded without sufficient progress
      if (fd_->steps_toward_frontier_ >= FSMConstants::NO_PROGRESS_MAX_STEPS) {
        double improvement = fd_->initial_dist_to_frontier_ - fd_->best_dist_to_frontier_;

        if (improvement < FSMConstants::NO_PROGRESS_MIN_IMPROVEMENT) {
          ROS_ERROR("[No-Progress] Abandoning frontier after %d steps with only %.2fm improvement (threshold: %.2fm)",
                    fd_->steps_toward_frontier_, improvement, FSMConstants::NO_PROGRESS_MIN_IMPROVEMENT);
          ROS_ERROR("[No-Progress] Frontier at (%.2f, %.2f), initial dist=%.2f, best dist=%.2f, current dist=%.2f",
                    end_pos(0), end_pos(1), fd_->initial_dist_to_frontier_,
                    fd_->best_dist_to_frontier_, min_dist);

          // Mark frontier as dormant and reset tracking
          expl_manager_->frontier_map2d_->setForceDormantFrontier(end_pos);
          fd_->dormant_frontier_flag_ = true;
          fd_->tracked_frontier_pos_ = Vector2d(-1000, -1000);
          fd_->initial_dist_to_frontier_ = -1.0;
          fd_->steps_toward_frontier_ = 0;
          fd_->best_dist_to_frontier_ = std::numeric_limits<double>::max();
          expl_manager_->resetWTRPHysteresis();  // Allow selecting new frontier
        } else {
          // Made progress, reset step counter but keep tracking same frontier
          ROS_INFO("[No-Progress] Made %.2fm progress, resetting step counter", improvement);
          fd_->initial_dist_to_frontier_ = fd_->best_dist_to_frontier_;
          fd_->steps_toward_frontier_ = 0;
        }
      }
    }
  }

  // Track consecutive stuck actions globally
  if ((current_pos - last_pos).norm() < stucking_distance) {
    fd_->stucking_action_count_++;
    ROS_ERROR_COND(fd_->stucking_action_count_ > 15, "Stucking action count = %d",
        fd_->stucking_action_count_);
  }
  else
    fd_->stucking_action_count_ = 0;

  // ==================== Spinning Detection (Agent rotating in place) ====================
  // Detect when agent keeps rotating without making spatial progress
  // This happens when value map keeps updating due to rotation, causing indecision
  constexpr double SPINNING_POSITION_THRESHOLD = 0.5;   // Position change threshold (meters)
  constexpr int SPINNING_TRIGGER_COUNT = 15;            // Steps before triggering boost
  constexpr double SPINNING_BOOST_VALUE = 0.2;          // Temporary value boost for nearest frontier

  if ((current_pos - last_pos).norm() < SPINNING_POSITION_THRESHOLD) {
    // Position barely changed
    if (fd_->spinning_step_count_ == 0) {
      fd_->spinning_start_pos_ = current_pos;
    }
    fd_->spinning_step_count_++;

    // Check if we've been spinning too long
    if (fd_->spinning_step_count_ >= SPINNING_TRIGGER_COUNT && !fd_->spinning_boost_active_) {
      // Mark current target frontier as dormant to force selection of a different one
      Vector2d current_target = expl_manager_->ed_->next_pos_;
      expl_manager_->frontier_map2d_->setForceDormantFrontier(current_target);

      fd_->spinning_boost_active_ = true;
      fd_->spinning_boost_frontier_ = current_target;
      fd_->dormant_frontier_flag_ = true;
      fd_->replan_flag_ = true;

      // Reset WTRP hysteresis to allow selecting a new frontier
      expl_manager_->resetWTRPHysteresis();

      ROS_WARN("[Spinning Detection] Agent spinning in place for %d steps, marking current target (%.2f, %.2f) as dormant to force replan",
               fd_->spinning_step_count_, current_target.x(), current_target.y());
    }
  } else {
    // Agent moved - reset spinning detection
    if (fd_->spinning_step_count_ > 0) {
      ROS_INFO("[Spinning Detection] Agent moved, resetting spinning counter (was %d steps)",
               fd_->spinning_step_count_);
    }
    fd_->spinning_step_count_ = 0;
    fd_->spinning_boost_active_ = false;
  }

  // ==================== Universal Stuck Recovery (All Navigation Modes) ====================
  // When stuck for moderate duration, mark obstacle in front and trigger replan
  // This applies to ALL modes (EXPLORE, SEARCH_OBJECT, etc.) to handle depth map issues
  Eigen::Vector2d current_target = expl_manager_->ed_->next_pos_;
  bool target_changed = (current_target - fd_->stuck_target_pos_).norm() > FSMConstants::TARGET_CHANGE_THRESHOLD;

  if (target_changed) {
    // New target - reset stuck recovery tracking
    fd_->stuck_replan_count_ = 0;
    fd_->stuck_obstacle_marked_ = false;
    fd_->stuck_target_pos_ = current_target;
  }

  // Count stuck actions toward current target
  if ((current_pos - last_pos).norm() < stucking_distance) {
    fd_->stuck_replan_count_++;

    // Trigger obstacle marking and replan when threshold reached
    if (fd_->stuck_replan_count_ >= FSMConstants::STUCK_REPLAN_TRIGGER_COUNT &&
        !fd_->stuck_obstacle_marked_) {
      ROS_WARN("[Stuck Recovery] Stuck count = %d toward target (%.2f, %.2f), marking obstacle and replanning",
               fd_->stuck_replan_count_, current_target.x(), current_target.y());

      // Mark obstacle in front of agent based on current heading
      double yaw = fd_->start_yaw_(0);
      Eigen::Vector2d forward_dir(cos(yaw), sin(yaw));
      Eigen::Vector2d obstacle_center = current_pos + forward_dir * FSMConstants::STUCK_OBSTACLE_MARK_DISTANCE;

      // Mark a small area as PERMANENT obstacle to prevent depth updates from clearing it
      // This ensures the agent won't repeatedly try the same blocked path
      Eigen::Vector2d perpendicular(-sin(yaw), cos(yaw));
      double half_width = FSMConstants::STUCK_OBSTACLE_MARK_WIDTH / 2.0;
      for (double offset = -half_width; offset <= half_width; offset += 0.05) {
        Eigen::Vector2d mark_pos = obstacle_center + perpendicular * offset;
        if (expl_manager_->sdf_map_->isInMap(mark_pos)) {
          expl_manager_->sdf_map_->setPermanentObstacle(mark_pos);
        }
      }

      fd_->stuck_obstacle_marked_ = true;
      fd_->replan_flag_ = true;  // Trigger replan to find alternative path

      ROS_WARN("[Stuck Recovery] Marked obstacle at (%.2f, %.2f), triggered replan",
               obstacle_center.x(), obstacle_center.y());
    }
  } else {
    // Made progress - reset stuck recovery
    fd_->stuck_replan_count_ = 0;
    fd_->stuck_obstacle_marked_ = false;
  }

  // If stuck for too long globally, try object map fallback before terminating
  if (fd_->stucking_action_count_ >= FSMConstants::MAX_STUCKING_COUNT) {
    ROS_ERROR("[Stucking Recovery] Stuck count = %d, trying object map fallback before terminating...",
              fd_->stucking_action_count_);

    // ==================== Stucking Recovery: Object Map Fallback ====================
    // When stuck count reaches terminal threshold (25), try to navigate to any object in object map
    // This considers ALL objects (not just high-confidence ones) sorted by confidence
    vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> all_object_clouds;
    expl_manager_->object_map2d_->getTopConfidenceObjectCloud(all_object_clouds, false);

    bool fallback_success = false;
    if (!all_object_clouds.empty()) {
      ROS_WARN("[Stucking Recovery] Found %zu objects in object map, trying to navigate...",
               all_object_clouds.size());

      // Try each object in order of confidence (highest first)
      for (size_t i = 0; i < all_object_clouds.size(); ++i) {
        Eigen::Vector2d fallback_pos;
        std::vector<Eigen::Vector2d> fallback_path;

        if (expl_manager_->searchObjectPath(fd_->start_pt_, all_object_clouds[i],
                                            fallback_pos, fallback_path)) {
          // Found a reachable object - switch to it
          expl_manager_->ed_->next_pos_ = fallback_pos;
          expl_manager_->ed_->next_best_path_ = fallback_path;
          final_res = FINAL_RESULT::SEARCH_OBJECT;
          fd_->stucking_action_count_ = 0;  // Reset stuck counter
          fd_->stucking_next_pos_count_ = 0;
          fallback_success = true;

          ROS_ERROR("[Stucking Recovery] SUCCESS! Switched to object #%zu at (%.2f, %.2f)",
                    i, fallback_pos.x(), fallback_pos.y());
          break;
        } else {
          ROS_WARN("[Stucking Recovery] Object #%zu not reachable, trying next...", i);
        }
      }
    } else {
      ROS_WARN("[Stucking Recovery] Object map is empty, no fallback available");
    }

    // If fallback failed, terminate episode
    if (!fallback_success) {
      ROS_ERROR("Stuck for too long and no reachable object, stopping episode.");
      final_res = FINAL_RESULT::STUCKING;
      return final_res;
    }
  }

  // Plan specific action based on exploration result
  if (expl_res == EXPL_RESULT::SEARCH_EXTREME)
    fd_->newest_action_ =
        planNextBestAction(current_pos, current_yaw, expl_manager_->ed_->next_best_path_, false);
  else
    fd_->newest_action_ =
        planNextBestAction(current_pos, current_yaw, expl_manager_->ed_->next_best_path_);

  // ==================== DEBUG: Final action decision ====================
  static int last_action_for_debug = -1;
  static int same_action_count = 0;
  if (fd_->newest_action_ == last_action_for_debug) {
    same_action_count++;
  } else {
    same_action_count = 0;
    last_action_for_debug = fd_->newest_action_;
  }

  const char* action_names[] = {"STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "TURN_UP", "TURN_DOWN"};
  const char* action_name = (fd_->newest_action_ >= 0 && fd_->newest_action_ <= 5)
                            ? action_names[fd_->newest_action_] : "UNKNOWN";

  ROS_WARN_THROTTLE(3.0, "[DEBUG ACTION] Decided action: %s (%d), repeated %d times, expl_res: %d, final_res: %d",
                    action_name, fd_->newest_action_, same_action_count, expl_res, final_res);

  // Alert if same action repeats too many times (possible stuck)
  if (same_action_count > 20) {
    ROS_ERROR("[DEBUG STUCK] Same action '%s' repeated %d times! Agent may be stuck!",
              action_name, same_action_count);
    ROS_ERROR("[DEBUG STUCK] Current pos: (%.2f, %.2f), Target: (%.2f, %.2f), Dist: %.2fm",
              current_pos.x(), current_pos.y(),
              expl_manager_->ed_->next_pos_.x(), expl_manager_->ed_->next_pos_.y(),
              (current_pos - expl_manager_->ed_->next_pos_).norm());
  }

  return final_res;
}

int ExplorationFSM::planNextBestAction(
    Vector2d current_pos, double current_yaw, const vector<Vector2d>& path, bool need_safety)
{
  const double local_distance = FSMConstants::LOCAL_DISTANCE;

  // Update target position based on path and local distance
  Vector2d local_pos = selectLocalTarget(current_pos, path, local_distance);
  fd_->local_pos_ = local_pos;

  // Compute the best step considering obstacles and safety
  Vector2d best_step;
  if ((current_pos - path.back()).norm() > FSMConstants::ACTION_DISTANCE && need_safety)
    best_step = computeBestStep(current_pos, current_yaw, local_pos);
  else
    best_step = local_pos;

  // Calculate target orientation from best step direction
  double target_yaw = std::atan2(best_step(1) - current_pos(1), best_step(0) - current_pos(0));
  return decideNextAction(current_yaw, target_yaw);
}

Vector2d ExplorationFSM::selectLocalTarget(
    const Vector2d& current_pos, const vector<Vector2d>& path, const double& local_distance)
{
  Vector2d target_pos = path.back();

  // Find the closest path point to current position as starting search index
  int start_path_id = 0;
  double min_dist = std::numeric_limits<double>::max();
  for (int i = 0; i < (int)path.size() - 1; i++) {
    Eigen::Vector2d pos = path[i];
    if ((pos - current_pos).norm() < min_dist) {
      min_dist = (pos - current_pos).norm();
      start_path_id = i + 1;
    }
  }

  // Select a local target position within the specified distance
  double len = (path[start_path_id] - current_pos).norm();
  for (int i = start_path_id + 1; i < (int)path.size(); i++) {
    len += (path[i] - path[i - 1]).norm();
    if (len > local_distance && (current_pos - path[i - 1]).norm() > 0.30) {
      target_pos = path[i - 1];
      break;
    }
  }

  return target_pos;
}

Vector2d ExplorationFSM::computeBestStep(
    const Vector2d& current_pos, double current_yaw, const Vector2d& target_pos)
{
  Vector2d best_step = target_pos;

  double min_cost = std::numeric_limits<double>::max();
  for (auto step : fp_->action_steps_) {
    double cost = computeActionTotalCost(current_pos, current_yaw, target_pos, step);
    if (cost < min_cost) {
      best_step = current_pos + step;
      min_cost = cost;
    }
  }

  return best_step;
}

// Compute total cost of taking a step towards target
// Considers distance-to-target, movement efficiency, and collision safety
double ExplorationFSM::computeActionTotalCost(const Vector2d& current_pos, double current_yaw,
    const Vector2d& target_pos, const Vector2d& step)
{
  const double traget_weight = FSMConstants::TARGET_WEIGHT;
  const double traget_close_weight1 = FSMConstants::TARGET_CLOSE_WEIGHT_1;
  const double traget_close_weight2 = FSMConstants::TARGET_CLOSE_WEIGHT_2;
  const double safety_weight = FSMConstants::SAFETY_WEIGHT;
  double cost = 0.0;

  // Distance-to-target cost
  Vector2d step_pos = current_pos + step;
  double target_cost = traget_weight * (step_pos - target_pos).norm();

  // Change-in-distance cost (negative if moving closer)
  double target_close_cost = (step_pos - target_pos).norm() - (current_pos - target_pos).norm();
  if (target_close_cost > 0)
    target_close_cost *= traget_close_weight1;
  else
    target_close_cost *= traget_close_weight2;

  // Safety distance cost
  double safety_cost = safety_weight * computeActionSafetyCost(current_pos, step);

  cost += target_cost + target_close_cost + safety_cost;
  return cost;
}

// Compute safety cost along the step using SDF distance to obstacles
// Returns higher cost for paths that go too close to obstacles
double ExplorationFSM::computeActionSafetyCost(const Vector2d& current_pos, const Vector2d& step)
{
  const double min_safe_distance = FSMConstants::MIN_SAFE_DISTANCE;
  const double sample_num = FSMConstants::SAMPLE_NUM;

  Vector2d dir = step;
  double len = dir.norm();
  dir.normalize();

  double safety_cost = 0.0;
  for (double l = len / sample_num; l < len; l += len / sample_num) {
    Vector2d ckpt = current_pos + l * dir;
    Vector2d grad;
    double dist_to_occ = expl_manager_->sdf_map_->getDistWithGrad(ckpt, grad);
    if (dist_to_occ < min_safe_distance)
      safety_cost += 1 / (dist_to_occ + 1e-2);
  }

  return safety_cost;
}

// Decide whether to turn or move forward based on yaw difference
// Uses action angle threshold to determine if orientation adjustment is needed
int ExplorationFSM::decideNextAction(double current_yaw, double target_yaw)
{
  double orig_current = current_yaw;
  double orig_target = target_yaw;

  wrapAngle(target_yaw);
  wrapAngle(current_yaw);
  double yaw_diff = target_yaw - current_yaw;
  wrapAngle(yaw_diff);

  int next_action;
  double threshold = FSMConstants::ACTION_ANGLE / 1.9;  // ~15.8 degrees

  if (std::fabs(yaw_diff) > threshold) {
    if (yaw_diff > 0)
      next_action = ACTION::TURN_LEFT;
    else
      next_action = ACTION::TURN_RIGHT;
  }
  else
    next_action = ACTION::MOVE_FORWARD;

  // DEBUG: Log decision details
  ROS_WARN_THROTTLE(3.0, "[DEBUG decideNextAction] current_yaw=%.1f deg, target_yaw=%.1f deg, yaw_diff=%.1f deg, threshold=%.1f deg -> action=%d",
                    current_yaw * 180.0 / M_PI, target_yaw * 180.0 / M_PI,
                    yaw_diff * 180.0 / M_PI, threshold * 180.0 / M_PI, next_action);

  return next_action;
}

void ExplorationFSM::visualize()
{
  auto ed_ptr = expl_manager_->ed_;

  // Lambda function to convert 2D vectors to 3D for visualization
  auto vec2dTo3d = [](const vector<Eigen::Vector2d>& vec2d, double z = 0.15) {
    vector<Eigen::Vector3d> vec3d;
    for (auto v : vec2d) vec3d.push_back(Vector3d(v(0), v(1), z));
    return vec3d;
  };

  // Draw frontier
  static int last_ftr2d_num = 0;
  for (int i = 0; i < (int)ed_ptr->frontiers_.size(); ++i) {
    visualization_->drawCubes(vec2dTo3d(ed_ptr->frontiers_[i]), fp_->vis_scale_,
        visualization_->getColor(double(i) / ed_ptr->frontiers_.size(), 1.0), "frontier", i, 4);
  }
  for (int i = ed_ptr->frontiers_.size(); i < last_ftr2d_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "frontier", i, 4);
  }
  last_ftr2d_num = ed_ptr->frontiers_.size();

  // Draw dormant frontier
  static int last_dftr2d_num = 0;
  for (int i = 0; i < (int)ed_ptr->dormant_frontiers_.size(); ++i) {
    visualization_->drawCubes(vec2dTo3d(ed_ptr->dormant_frontiers_[i]), fp_->vis_scale_,
        Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
  }
  for (int i = ed_ptr->dormant_frontiers_.size(); i < last_dftr2d_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
  }
  last_dftr2d_num = ed_ptr->dormant_frontiers_.size();

  // Draw object
  // static int last_obj_num = 0;
  // for (int i = 0; i < (int)ed_ptr->objects_.size(); ++i) {
  //   visualization_->drawCubes(vec2dTo3d(ed_ptr->objects_[i]), fp_->vis_scale_,
  //       visualization_->getColor(double(i) / ed_ptr->objects_.size(), 1.0), "object", i, 4);
  // }
  // for (int i = ed_ptr->objects_.size(); i < last_obj_num; ++i) {
  //   visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "object", i, 4);
  // }
  // last_obj_num = ed_ptr->objects_.size();

  static int last_obj_num = 0;
  for (int i = 0; i < (int)ed_ptr->objects_.size(); ++i) {
    int label = ed_ptr->object_labels_[i];
    visualization_->drawCubes(vec2dTo3d(ed_ptr->objects_[i]), fp_->vis_scale_,
        visualization_->getColor(double(label) / 5.0, 1.0), "object", i, 4);
  }
  for (int i = ed_ptr->objects_.size(); i < last_obj_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "object", i, 4);
  }
  last_obj_num = ed_ptr->objects_.size();

  // Draw next best path
  visualization_->drawLines(vec2dTo3d(ed_ptr->next_best_path_), fp_->vis_scale_,
      Vector4d(1, 0.2, 0.2, 1), "next_path", 1, 6);

  // Draw next local point
  vector<Vector2d> local_points;
  local_points.push_back(fd_->local_pos_);
  visualization_->drawSpheres(vec2dTo3d(local_points), fp_->vis_scale_ * 3,
      Vector4d(0.2, 0.2, 1.0, 1), "local_point", 1, 6);

  visualization_->drawLines(vec2dTo3d(ed_ptr->tsp_tour_), fp_->vis_scale_ / 1.25,
      Vector4d(0.2, 1, 0.2, 1), "tsp_tour", 0, 6);

  visualization_->drawSpheres(vec2dTo3d(fd_->traveled_path_), fp_->vis_scale_ * 1.5,
      Vector4d(2.0 / 255.0, 111.0 / 255.0, 197.0 / 255.0, 1), "traveled_path", 1, 6);
}

void ExplorationFSM::clearVisMarker()
{
  auto ed_ptr = expl_manager_->ed_;
  for (int i = 0; i < 500; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "frontier", i, 4);
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "object", i, 4);
  }

  visualization_->drawLines({}, fp_->vis_scale_, Vector4d(0, 0, 1, 1), "next_path", 1, 6);
}

bool ExplorationFSM::updateFrontierAndObject()
{
  // ========== ROS PERFORMANCE PROFILING START ==========
  auto t_total_start = ros::Time::now();

  bool change_flag = false;
  auto frt_map = expl_manager_->frontier_map2d_;
  auto obj_map = expl_manager_->object_map2d_;
  auto ed = expl_manager_->ed_;
  Eigen::Vector2d start_pos2d = Eigen::Vector2d(fd_->start_pt_(0), fd_->start_pt_(1));

  // ====== Patience-Aware Navigation: Update step count ======
  fd_->step_count_++;
  obj_map->setCurrentStep(fd_->step_count_);

  auto t0 = ros::Time::now();
  change_flag = frt_map->isAnyFrontierChanged();
  double t_check_change = (ros::Time::now() - t0).toSec();

  t0 = ros::Time::now();
  frt_map->searchFrontiers();
  double t_search = (ros::Time::now() - t0).toSec();

  t0 = ros::Time::now();
  change_flag |= frt_map->dormantSeenFrontiers(start_pos2d, fd_->start_yaw_(0));
  double t_dormant = (ros::Time::now() - t0).toSec();

  t0 = ros::Time::now();
  frt_map->getFrontiers(ed->frontiers_, ed->frontier_averages_);
  frt_map->getDormantFrontiers(ed->dormant_frontiers_, ed->dormant_frontier_averages_);
  double t_get_frontiers = (ros::Time::now() - t0).toSec();

  t0 = ros::Time::now();
  obj_map->getObjects(ed->objects_, ed->object_averages_, ed->object_labels_);
  double t_get_objects = (ros::Time::now() - t0).toSec();

  // ====== Patience-Aware Navigation: Check for confirmed target ======
  if (obj_map->hasConfirmedTarget()) {
    double tau_t = obj_map->getCurrentConfidenceThreshold();
    ROS_INFO("[Patience] Found confirmed target at step %d (tau=%.3f, target='%s')",
             fd_->step_count_, tau_t, obj_map->getTargetCategory().c_str());
  }

  double t_total = (ros::Time::now() - t_total_start).toSec();

  // Print ROS-side timing breakdown
  ROS_INFO("\n[ROS TIMING] updateFrontierAndObject:");
  ROS_INFO("  Total: %.3fs", t_total);
  ROS_INFO("  - Check Changes:    %.3fs (%.1f%%) [%zu frontiers]",
           t_check_change, t_check_change/t_total*100, ed->frontiers_.size() + ed->dormant_frontiers_.size());
  ROS_INFO("  - Search Frontiers: %.3fs (%.1f%%)", t_search, t_search/t_total*100);
  ROS_INFO("  - Dormant Check:    %.3fs (%.1f%%)", t_dormant, t_dormant/t_total*100);
  ROS_INFO("  - Get Frontiers:    %.3fs (%.1f%%)", t_get_frontiers, t_get_frontiers/t_total*100);
  ROS_INFO("  - Get Objects:      %.3fs (%.1f%%) [%zu objects]",
           t_get_objects, t_get_objects/t_total*100, ed->objects_.size());
  ROS_INFO("  - Patience Step:    %d, tau(t)=%.3f", fd_->step_count_, obj_map->getCurrentConfidenceThreshold());
  // ========== ROS PERFORMANCE PROFILING END ==========

  return change_flag;
}

// Receive Habitat state messages
void ExplorationFSM::habitatStateCallback(const std_msgs::Int32ConstPtr& msg)
{
  if (msg->data == HABITAT_STATE::ACTION_FINISH && state_ == ROS_STATE::WAIT_ACTION_FINISH)
    transitState(PLAN_ACTION, "Habitat Finish Action");
  if (msg->data == HABITAT_STATE::EPISODE_FINISH) {
    // Note: Episode reset is now handled by episodeResetCallback via /habitat/episode_reset topic
    // This prevents duplicate init() calls and state conflicts
    // Just transition to INIT state and wait for reset signal
    ROS_INFO("[FSM] Received EPISODE_FINISH, transitioning to INIT state");
    state_ = ROS_STATE::INIT;
    fd_->have_finished_ = false;
  }
  return;
}

// Periodically update frontiers and visualize in idle states
void ExplorationFSM::frontierCallback(const ros::TimerEvent& e)
{
  if (state_ != ROS_STATE::WAIT_TRIGGER && state_ != ROS_STATE::FINISH)
    return;

  // Prevent re-entrance if previous call still processing
  if (frontier_processing_) {
    ROS_WARN_THROTTLE(2.0, "[frontierCallback] Still processing, skipping this call (CPU overload!)");
    return;
  }

  frontier_processing_ = true;
  auto t_start = ros::Time::now();

  updateFrontierAndObject();
  visualize();

  auto t_end = ros::Time::now();
  double duration = (t_end - t_start).toSec();

  // Log if processing takes too long
  if (duration > 0.5) {
    ROS_WARN("[frontierCallback] Processing took %.3fs (exceeds timer interval!)", duration);
  }

  // Log periodically for monitoring
  static int call_count = 0;
  static double max_duration = 0.0;
  call_count++;
  if (duration > max_duration) max_duration = duration;

  if (call_count % 20 == 0) {
    ROS_INFO("[frontierCallback] Called %d times, last=%.3fs, max=%.3fs",
             call_count, duration, max_duration);
  }

  frontier_processing_ = false;
}

// Receive user trigger to start exploration
void ExplorationFSM::triggerCallback(const geometry_msgs::PoseStampedConstPtr& msg)
{
  if (state_ != ROS_STATE::WAIT_TRIGGER)
    return;
  fd_->trigger_ = true;
  cout << "Triggered!" << endl;
  transitState(PLAN_ACTION, "triggerCallback");
}

// Receive robot odometry and update traveled path + marker
void ExplorationFSM::odometryCallback(const nav_msgs::OdometryConstPtr& msg)
{
  fd_->odom_pos_(0) = msg->pose.pose.position.x;
  fd_->odom_pos_(1) = msg->pose.pose.position.y;
  fd_->odom_pos_(2) = msg->pose.pose.position.z;

  fd_->odom_orient_.w() = msg->pose.pose.orientation.w;
  fd_->odom_orient_.x() = msg->pose.pose.orientation.x;
  fd_->odom_orient_.y() = msg->pose.pose.orientation.y;
  fd_->odom_orient_.z() = msg->pose.pose.orientation.z;

  Eigen::Vector3d rot_x = fd_->odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
  fd_->odom_yaw_ = atan2(rot_x(1), rot_x(0));

  fd_->have_odom_ = true;

  Vector2d odom_pos2d = Vector2d(fd_->odom_pos_(0), fd_->odom_pos_(1));
  if (fd_->traveled_path_.empty())
    fd_->traveled_path_.push_back(odom_pos2d);
  else if ((fd_->traveled_path_.back() - odom_pos2d).norm() > 1e-2)
    fd_->traveled_path_.push_back(odom_pos2d);

  publishRobotMarker();
}

// Callback for detector per-frame outputs
void ExplorationFSM::detectorCallback(const plan_env::MultipleMasksWithConfidenceConstPtr& msg)
{
  boost::mutex::scoped_lock lock(detector_mutex_);
  bool has_target = false;

  // Get minimum confidence threshold for approach detection check
  // Use fixed confidence threshold (not dependent on dynamic tau_t)
  double min_approach_confidence = FSMConstants::APPROACH_CONFIDENCE_RATIO;

  // Check each detection for target class with sufficient confidence
  for (size_t i = 0; i < msg->label_indices.size(); ++i) {
    if (msg->label_indices[i] == 0) {  // label 0 corresponds to target class
      // Check confidence threshold
      if (i < msg->confidence_scores.size() &&
          msg->confidence_scores[i] >= min_approach_confidence) {
        has_target = true;
        ROS_DEBUG("[Approach Detection] Target detected with confidence=%.3f >= threshold=%.3f",
                  msg->confidence_scores[i], min_approach_confidence);
        break;
      }
    }
  }
  last_frame_has_target_ = has_target;
  last_detection_time_ = ros::Time::now();
}

void ExplorationFSM::publishRobotMarker()
{
  const double robot_height = FSMConstants::ROBOT_HEIGHT;
  const double robot_radius = FSMConstants::ROBOT_RADIUS;

  // Create robot body cylinder marker
  visualization_msgs::Marker robot_marker;
  robot_marker.header.frame_id = "world";
  robot_marker.header.stamp = ros::Time::now();
  robot_marker.ns = "robot_position";
  robot_marker.id = 0;
  robot_marker.type = visualization_msgs::Marker::CYLINDER;
  robot_marker.action = visualization_msgs::Marker::ADD;

  // Set cylinder position
  robot_marker.pose.position.x = fd_->odom_pos_(0);
  robot_marker.pose.position.y = fd_->odom_pos_(1);
  robot_marker.pose.position.z = fd_->odom_pos_(2) + robot_height / 2.0;

  // Set cylinder orientation
  robot_marker.pose.orientation.x = fd_->odom_orient_.x();
  robot_marker.pose.orientation.y = fd_->odom_orient_.y();
  robot_marker.pose.orientation.z = fd_->odom_orient_.z();
  robot_marker.pose.orientation.w = fd_->odom_orient_.w();

  // Set cylinder dimensions
  robot_marker.scale.x = robot_radius * 2;  // Diameter
  robot_marker.scale.y = robot_radius * 2;  // Diameter
  robot_marker.scale.z = robot_height;      // Height

  // Set cylinder color (blue)
  robot_marker.color.r = 50.0 / 255.0;
  robot_marker.color.g = 50.0 / 255.0;
  robot_marker.color.b = 255.0 / 255.0;
  robot_marker.color.a = 1.0;

  // Create direction arrow marker
  visualization_msgs::Marker arrow_marker;
  arrow_marker.header.frame_id = "world";
  arrow_marker.header.stamp = ros::Time::now();
  arrow_marker.ns = "robot_direction";
  arrow_marker.id = 1;
  arrow_marker.type = visualization_msgs::Marker::ARROW;
  arrow_marker.action = visualization_msgs::Marker::ADD;

  // Set arrow position
  arrow_marker.pose.position.x = fd_->odom_pos_(0);
  arrow_marker.pose.position.y = fd_->odom_pos_(1);
  arrow_marker.pose.position.z = fd_->odom_pos_(2) + robot_height;

  // Set arrow orientation
  arrow_marker.pose.orientation.x = fd_->odom_orient_.x();
  arrow_marker.pose.orientation.y = fd_->odom_orient_.y();
  arrow_marker.pose.orientation.z = fd_->odom_orient_.z();
  arrow_marker.pose.orientation.w = fd_->odom_orient_.w();

  // Set arrow dimensions
  arrow_marker.scale.x = robot_radius + 0.13;  // Arrow length
  arrow_marker.scale.y = 0.08;                 // Arrow width
  arrow_marker.scale.z = 0.08;                 // Arrow thickness

  // Set arrow color (green)
  arrow_marker.color.r = 10.0 / 255.0;
  arrow_marker.color.g = 255.0 / 255.0;
  arrow_marker.color.b = 10.0 / 255.0;
  arrow_marker.color.a = 1.0;

  // Publish both markers
  robot_marker_pub_.publish(robot_marker);
  robot_marker_pub_.publish(arrow_marker);
}

void ExplorationFSM::confidenceThresholdCallback(const std_msgs::Float64ConstPtr& msg)
{
  if (fd_->have_confidence_)
    return;
  fd_->have_confidence_ = true;
  expl_manager_->sdf_map_->object_map2d_->setConfidenceThreshold(msg->data);
}

void ExplorationFSM::episodeResetCallback(const std_msgs::EmptyConstPtr& msg)
{
  ROS_INFO("[ExplorationFSM] Received episode reset signal");

  // Reset FSM state to INIT to stop processing old episode data
  state_ = ROS_STATE::INIT;

  // Reset VLM multi-view validation state
  vlm_state_ = VLMValidationState::IDLE;
  vlm_current_view_idx_ = 0;
  vlm_views_confirmed_ = 0;
  vlm_view_results_.clear();

  // Reset VLM sliding window state
  vlm_candidate_frames_.clear();
  vlm_collection_locked_ = false;
  vlm_locked_object_id_ = -1;
  vlm_locked_target_pos_ = Eigen::Vector2d::Zero();
  vlm_target_confirmed_ = false;

  // Reset final approach validation state
  resetFinalApproachValidation();

  // Reset FSM flags
  if (fd_) {
    fd_->have_odom_ = false;
    fd_->have_confidence_ = false;
    fd_->have_finished_ = false;
    fd_->trigger_ = false;

    // CRITICAL: Reset init_action_count_ to enable 26-step initial scanning
    // Without this, the robot skips the initial environment scan and
    // immediately enters exploration mode with an empty frontier map
    fd_->init_action_count_ = 0;

    // Reset action and planning state
    fd_->newest_action_ = -1;
    fd_->final_result_ = -1;
    fd_->replan_flag_ = true;
    fd_->dormant_frontier_flag_ = false;

    // Reset stuck detection state
    fd_->stucking_action_count_ = 0;
    fd_->stucking_next_pos_count_ = 0;
    fd_->escape_stucking_flag_ = false;
    fd_->escape_stucking_count_ = 0;

    // Reset universal stuck recovery state
    fd_->stuck_replan_count_ = 0;
    fd_->stuck_obstacle_marked_ = false;
    fd_->stuck_target_pos_ = Eigen::Vector2d(-1000, -1000);

    // Reset spinning detection state
    fd_->spinning_step_count_ = 0;
    fd_->spinning_boost_active_ = false;
    fd_->spinning_start_pos_ = Eigen::Vector2d(0, 0);
    fd_->spinning_boost_frontier_ = Eigen::Vector2d(0, 0);

    // Reset position tracking
    fd_->last_start_pos_ = Eigen::Vector3d(-100, -100, -100);
    fd_->last_next_pos_ = Eigen::Vector2d(-100, -100);
    fd_->local_pos_ = Eigen::Vector2d(0, 0);

    // Clear accumulated path and stucking points to prevent memory growth
    size_t path_size = fd_->traveled_path_.size();
    size_t stuck_size = fd_->stucking_points_.size();
    fd_->traveled_path_.clear();
    fd_->stucking_points_.clear();
    ROS_INFO("[ExplorationFSM] Cleared %zu traveled_path points and %zu stucking_points",
             path_size, stuck_size);

    // Reset Patience-Aware Navigation state
    fd_->step_count_ = 0;
    fd_->target_category_ = "";

    // Reset suspicious target locking
    fd_->suspicious_target_locked_ = false;
    fd_->locked_suspicious_pos_ = Eigen::Vector2d(0, 0);

    // Reset no-progress detection for frontier exploration
    fd_->tracked_frontier_pos_ = Eigen::Vector2d(-1000, -1000);
    fd_->initial_dist_to_frontier_ = -1.0;
    fd_->steps_toward_frontier_ = 0;
    fd_->best_dist_to_frontier_ = std::numeric_limits<double>::max();

    // Reset soft arrival detection for object navigation
    fd_->approach_best_dist_ = std::numeric_limits<double>::max();
    fd_->approach_attempt_count_ = 0;
    fd_->approach_no_improvement_count_ = 0;

    // Log VLM approach verification statistics before resetting
    ROS_INFO("[VLM Stats] Episode Summary: used=%s, disabled=%s, verify_count=%d",
             fd_->vlm_used_this_episode_ ? "true" : "false",
             fd_->vlm_disabled_this_episode_ ? "true" : "false",
             fd_->vlm_verify_count_);

    // Reset VLM approach verification state
    fd_->vlm_waiting_ = false;
    fd_->vlm_target_object_id_ = -1;
    fd_->vlm_trigger_type_ = 0;
    fd_->vlm_disabled_this_episode_ = false;
    fd_->vlm_used_this_episode_ = false;
    fd_->vlm_verify_count_ = 0;

    ROS_INFO("[ExplorationFSM] Reset init_action_count_ to 0 for 26-step initial scan");
  }

  // Clear all frontiers for new episode
  if (expl_manager_ && expl_manager_->frontier_map2d_) {
    expl_manager_->frontier_map2d_->clearFrontiers();
  }

  // Reset all map buffers (occupancy, distance, inflation, etc.)
  if (expl_manager_ && expl_manager_->sdf_map_) {
    expl_manager_->sdf_map_->resetAllBuffers();
    ROS_INFO("[ExplorationFSM] Reset all SDF map buffers");
  }

  // Clear exploration data vectors to prevent accumulation across episodes
  if (expl_manager_ && expl_manager_->ed_) {
    size_t tsp_size = expl_manager_->ed_->tsp_tour_.size();
    size_t path_size = expl_manager_->ed_->next_best_path_.size();
    expl_manager_->ed_->tsp_tour_.clear();
    expl_manager_->ed_->next_best_path_.clear();
    ROS_INFO("[ExplorationFSM] Cleared %zu tsp_tour points and %zu next_best_path points",
             tsp_size, path_size);
  }

  // Reset suspicious target lock for new episode
  if (expl_manager_) {
    expl_manager_->setSuspiciousTargetLock(false);
    ROS_INFO("[ExplorationFSM] Reset suspicious target lock");
  }

  // Reset object approach point lock for new episode
  if (expl_manager_) {
    expl_manager_->setObjectApproachLock(false);
    ROS_INFO("[ExplorationFSM] Reset object approach point lock");
  }

  // Reset approach detection and soft arrival tracking for new episode
  resetApproachDetectionCheck();
  resetSoftArrivalTracking();
  ROS_INFO("[ExplorationFSM] Reset approach detection and soft arrival tracking");

  // Clear object detection map
  if (expl_manager_ && expl_manager_->sdf_map_ && expl_manager_->sdf_map_->object_map2d_) {
    expl_manager_->sdf_map_->object_map2d_->clearObjects();
    ROS_INFO("[ExplorationFSM] Cleared object map");
  }

  // Reset multi-source value map (semantic fusion)
  if (expl_manager_ && expl_manager_->sdf_map_ &&
      expl_manager_->sdf_map_->multi_source_value_map_) {
    expl_manager_->sdf_map_->multi_source_value_map_->resetForNewEpisode();
    ROS_INFO("[ExplorationFSM] Reset multi-source value map");
  }

  ROS_INFO("[ExplorationFSM] Episode reset complete - all resources cleaned");
}

// Transition FSM state and log the change
void ExplorationFSM::transitState(ROS_STATE new_state, string pos_call)
{
  int pre_s = int(state_);
  state_ = new_state;
  cout << "[ " + pos_call + "]: from " + fd_->state_str_[pre_s] + " to " +
              fd_->state_str_[int(new_state)]
       << endl;
}

// ==================== VLM Sliding Window Functions ====================

void ExplorationFSM::enterCollectionPhase(int object_id, const Eigen::Vector2d& target_pos)
{
  if (vlm_collection_locked_) {
    return;  // Already in collection phase
  }

  vlm_collection_locked_ = true;
  vlm_locked_object_id_ = object_id;
  vlm_locked_target_pos_ = target_pos;
  vlm_candidate_frames_.clear();

  ROS_INFO("[VLM Sliding] Entering collection phase: locked object_id=%d at (%.2f, %.2f)",
           object_id, target_pos.x(), target_pos.y());
}

void ExplorationFSM::exitCollectionPhase(bool success)
{
  if (!vlm_collection_locked_) {
    return;  // Not in collection phase
  }

  ROS_INFO("[VLM Sliding] Exiting collection phase: %s, object_id=%d",
           success ? "SUCCESS" : "FAILED", vlm_locked_object_id_);

  if (!success) {
    // Mark object as invalid if validation failed
    if (vlm_locked_object_id_ >= 0) {
      expl_manager_->object_map2d_->markObjectAsInvalid(vlm_locked_object_id_);
      ROS_WARN("[VLM Sliding] Object id=%d added to blacklist", vlm_locked_object_id_);
    }
  }

  vlm_collection_locked_ = false;
  vlm_locked_object_id_ = -1;
  vlm_locked_target_pos_ = Eigen::Vector2d::Zero();
  vlm_candidate_frames_.clear();
  vlm_target_confirmed_ = success;
}

void ExplorationFSM::clearVLMCandidateFrames()
{
  vlm_candidate_frames_.clear();
}

void ExplorationFSM::collectVLMCandidateFrame()
{
  // Check if detector has seen target recently
  boost::mutex::scoped_lock lock(detector_mutex_);
  if (!last_frame_has_target_) {
    return;
  }

  // Check detection freshness
  double time_since_detection = (ros::Time::now() - last_detection_time_).toSec();
  if (time_since_detection > 0.5) {
    return;
  }

  // Check if detection is from our locked target object
  if (!isDetectionFromTargetObject()) {
    ROS_DEBUG("[VLM Sliding] Detection not from locked target object, skipping");
    return;
  }

  // Create candidate frame
  VLMCandidateFrame frame;
  frame.timestamp = last_detection_time_;
  frame.detection_confidence = 0.5;  // TODO: Get actual confidence from detector message
  frame.distance_to_target = (fd_->start_pt_.head<2>() - vlm_locked_target_pos_).norm();
  frame.robot_position = fd_->start_pt_.head<2>();
  frame.robot_yaw = fd_->start_yaw_(0);
  frame.detected_object_id = vlm_locked_object_id_;

  // Add to sliding window
  vlm_candidate_frames_.push_back(frame);

  // Remove oldest frames if window exceeds max size
  while (vlm_candidate_frames_.size() > static_cast<size_t>(FSMConstants::VLM_WINDOW_MAX_FRAMES)) {
    vlm_candidate_frames_.pop_front();
  }

  // Also remove frames that are too old
  ros::Time now = ros::Time::now();
  while (!vlm_candidate_frames_.empty() &&
         (now - vlm_candidate_frames_.front().timestamp).toSec() > FSMConstants::VLM_FRAME_MAX_AGE) {
    vlm_candidate_frames_.pop_front();
  }

  ROS_INFO("[VLM Sliding] Collected frame: window_size=%zu, dist=%.2fm, conf=%.2f",
           vlm_candidate_frames_.size(), frame.distance_to_target, frame.detection_confidence);
}

bool ExplorationFSM::isDetectionFromTargetObject()
{
  // If not in collection phase, no target to match
  if (!vlm_collection_locked_ || vlm_locked_object_id_ < 0) {
    return false;
  }

  // Simple approach: if we're within collection distance of the locked target,
  // assume the detection is from that target.
  // More precise matching would require 3D position from detection point cloud.
  double dist_to_locked = (fd_->start_pt_.head<2>() - vlm_locked_target_pos_).norm();

  // Accept detections when robot is heading towards and near the locked target
  return dist_to_locked <= FSMConstants::VLM_COLLECT_DISTANCE;
}

bool ExplorationFSM::selectBestFramesForValidation(std::vector<VLMCandidateFrame>& selected)
{
  selected.clear();

  if (vlm_candidate_frames_.empty()) {
    return false;
  }

  // Sort frames by confidence (descending) and distance (ascending for tie-breaking)
  std::vector<VLMCandidateFrame> sorted_frames(
      vlm_candidate_frames_.begin(), vlm_candidate_frames_.end());

  std::sort(sorted_frames.begin(), sorted_frames.end(),
            [](const VLMCandidateFrame& a, const VLMCandidateFrame& b) {
              // Primary: higher confidence is better
              if (std::abs(a.detection_confidence - b.detection_confidence) > 0.01) {
                return a.detection_confidence > b.detection_confidence;
              }
              // Secondary: closer distance is better
              return a.distance_to_target < b.distance_to_target;
            });

  // Select top 1-2 frames for validation
  int num_to_select = std::min(2, static_cast<int>(sorted_frames.size()));
  for (int i = 0; i < num_to_select; i++) {
    selected.push_back(sorted_frames[i]);
  }

  ROS_INFO("[VLM Sliding] Selected %d best frames for validation", num_to_select);
  return !selected.empty();
}

bool ExplorationFSM::performSlidingWindowVLMValidation(const std::vector<VLMCandidateFrame>& frames)
{
  if (frames.empty()) {
    ROS_WARN("[VLM Sliding] No candidate frames for validation");
    return false;
  }

  // For now, use the existing single-frame VLM validation
  // The Python side will use its cached images based on timing
  // Future enhancement: pass timestamps to Python for exact frame retrieval

  ROS_INFO("[VLM Sliding] Performing validation with %zu candidate frames", frames.size());

  // Call standard VLM validation - Python node uses its latest cached image
  bool result = callVLMValidation();

  ROS_INFO("[VLM Sliding] Validation result: %s", result ? "PASS" : "FAIL");
  return result;
}

// ==================== VLM Validation ====================

bool ExplorationFSM::callVLMValidation()
{
  if (!vlm_validation_enabled_) {
    ROS_WARN("[VLM] Validation disabled, assuming valid");
    return true;
  }

  // Check if service is available
  if (!vlm_validation_client_.exists()) {
    ROS_WARN("[VLM] Validation service not available, assuming valid");
    return true;
  }

  // Get target object category
  std::string target_category = expl_manager_->object_map2d_->getTargetCategory();
  if (target_category.empty()) {
    ROS_WARN("[VLM] No target category set, assuming valid");
    return true;
  }

  // Create service request
  plan_env::ValidateObject srv;
  srv.request.target_object = target_category;

  ROS_INFO("[VLM] Calling validation service for target: '%s'", target_category.c_str());

  // Call service with timeout
  auto start_time = ros::Time::now();
  bool service_success = vlm_validation_client_.call(srv);
  double call_duration = (ros::Time::now() - start_time).toSec();

  if (!service_success) {
    ROS_ERROR("[VLM] Service call failed after %.2fs, assuming valid", call_duration);
    return true;
  }

  ROS_INFO("[VLM] Validation result: is_valid=%s, confidence=%.2f, response='%s' (took %.2fs)",
           srv.response.is_valid ? "true" : "false",
           srv.response.confidence,
           srv.response.raw_response.c_str(),
           call_duration);

  return srv.response.is_valid;
}

// ==================== Multi-View VLM Validation ====================

bool ExplorationFSM::startMultiViewValidation()
{
  int current_obj_id = expl_manager_->object_map2d_->getCurrentTargetObjectId();
  if (current_obj_id < 0) {
    ROS_WARN("[VLM Multi-View] No target object ID set");
    return false;
  }

  Eigen::Vector2d robot_pos(fd_->start_pt_(0), fd_->start_pt_(1));
  double robot_yaw = fd_->start_yaw_(0);

  // Compute observation viewpoints for the target object
  std::vector<ObjectMap2D::ObservationViewpoint> viewpoints;
  bool found = expl_manager_->object_map2d_->computeObservationViewpoints(
      current_obj_id, robot_pos, robot_yaw, vlm_camera_hfov_, viewpoints);

  if (!found || viewpoints.empty()) {
    ROS_WARN("[VLM Multi-View] No valid observation viewpoints found, using current position");
    // Fall back to current position validation
    vlm_state_ = VLMValidationState::ROTATING;

    // Get object center to compute facing direction
    Vector2d object_center;
    if (expl_manager_->object_map2d_->getObjectCenter(current_obj_id, object_center)) {
      vlm_target_yaw_ = std::atan2(object_center.y() - robot_pos.y(),
                                   object_center.x() - robot_pos.x());
    } else {
      vlm_target_yaw_ = robot_yaw;  // Keep current orientation
    }
    vlm_target_viewpoint_ = robot_pos;
  } else {
    // Use the first (best) viewpoint
    vlm_target_viewpoint_ = viewpoints[0].position;
    vlm_target_yaw_ = viewpoints[0].yaw;

    // Check if we're already close to the viewpoint
    double dist_to_viewpoint = (robot_pos - vlm_target_viewpoint_).norm();
    if (dist_to_viewpoint < 0.3) {
      // Already at viewpoint, just need to rotate
      vlm_state_ = VLMValidationState::ROTATING;
      ROS_INFO("[VLM Multi-View] Already at viewpoint, rotating to face object");
    } else {
      // Need to navigate to viewpoint
      vlm_state_ = VLMValidationState::NAVIGATING;
      ROS_INFO("[VLM Multi-View] Navigating to viewpoint (%.2f, %.2f), dist=%.2f",
               vlm_target_viewpoint_.x(), vlm_target_viewpoint_.y(), dist_to_viewpoint);
    }
  }

  // Initialize validation tracking
  vlm_current_view_idx_ = 0;
  vlm_views_confirmed_ = 0;
  vlm_view_results_.clear();

  return true;
}

int ExplorationFSM::processMultiViewValidation()
{
  Eigen::Vector2d robot_pos(fd_->start_pt_(0), fd_->start_pt_(1));
  double robot_yaw = fd_->start_yaw_(0);

  // State machine for multi-view validation
  switch (vlm_state_) {
    case VLMValidationState::IDLE: {
      // Start multi-view validation
      if (!startMultiViewValidation()) {
        // Failed to start, fall back to single-view validation
        bool single_valid = callVLMValidation();
        return single_valid ? FINAL_RESULT::REACH_OBJECT : -2;
      }
      return -1;  // Continue processing
    }

    case VLMValidationState::NAVIGATING: {
      // Navigate to viewpoint
      double dist_to_viewpoint = (robot_pos - vlm_target_viewpoint_).norm();

      if (dist_to_viewpoint < 0.3) {
        // Reached viewpoint, start rotating
        vlm_state_ = VLMValidationState::ROTATING;
        ROS_INFO("[VLM Multi-View] Reached viewpoint, rotating to face object");
        return -1;
      }

      // Plan action to navigate to viewpoint
      double target_angle = std::atan2(vlm_target_viewpoint_.y() - robot_pos.y(),
                                       vlm_target_viewpoint_.x() - robot_pos.x());
      fd_->newest_action_ = decideNextAction(robot_yaw, target_angle);

      // If facing the right direction, move forward
      if (fd_->newest_action_ == ACTION::MOVE_FORWARD) {
        // Already set, just return
      }

      return -1;  // Continue processing
    }

    case VLMValidationState::ROTATING: {
      // Rotate to face object
      int rotation_action = computeRotationAction(robot_yaw, vlm_target_yaw_);

      if (rotation_action == ACTION::STOP) {
        // Aligned, start capturing
        vlm_state_ = VLMValidationState::CAPTURING;
        ROS_INFO("[VLM Multi-View] Aligned with object, capturing view %d/%d",
                 vlm_current_view_idx_ + 1, vlm_num_views_);
        return -1;
      }

      fd_->newest_action_ = rotation_action;
      return -1;  // Continue processing
    }

    case VLMValidationState::CAPTURING: {
      // Capture and validate current view
      bool view_valid = captureAndValidateCurrentView();
      vlm_view_results_.push_back({view_valid, 0.0});  // TODO: get confidence

      if (view_valid) {
        vlm_views_confirmed_++;
        ROS_INFO("[VLM Multi-View] View %d/%d: CONFIRMED",
                 vlm_current_view_idx_ + 1, vlm_num_views_);
      } else {
        ROS_WARN("[VLM Multi-View] View %d/%d: REJECTED",
                 vlm_current_view_idx_ + 1, vlm_num_views_);
        // AND logic: If any view rejects, validation fails immediately
        vlm_state_ = VLMValidationState::COMPLETED;
        return -2;  // Validation failed
      }

      vlm_current_view_idx_++;

      // Check if we have enough confirmed views
      if (vlm_current_view_idx_ >= vlm_num_views_) {
        // All views completed and confirmed (due to AND logic, we only get here if all passed)
        vlm_state_ = VLMValidationState::COMPLETED;
        return FINAL_RESULT::REACH_OBJECT;  // Success!
      }

      // Move to next viewpoint for additional view
      // For simplicity, just rotate 45 degrees for next view
      vlm_target_yaw_ += M_PI / 4;
      while (vlm_target_yaw_ > M_PI) vlm_target_yaw_ -= 2 * M_PI;
      vlm_state_ = VLMValidationState::ROTATING;

      ROS_INFO("[VLM Multi-View] Moving to next view angle: %.1f°",
               vlm_target_yaw_ * 180 / M_PI);
      return -1;  // Continue processing
    }

    case VLMValidationState::WAITING_RESPONSE: {
      // This state is handled synchronously in captureAndValidateCurrentView
      // But we include it for completeness
      return -1;
    }

    case VLMValidationState::COMPLETED: {
      // Validation completed, check results
      if (vlm_views_confirmed_ >= vlm_num_views_) {
        return FINAL_RESULT::REACH_OBJECT;  // Success
      } else {
        return -2;  // Failed
      }
    }

    default:
      ROS_ERROR("[VLM Multi-View] Unknown state: %d", static_cast<int>(vlm_state_));
      return -2;
  }
}

bool ExplorationFSM::captureAndValidateCurrentView()
{
  // Wait a short time to ensure camera has stabilized
  ros::Duration(0.1).sleep();

  // Call the VLM validation service for current view
  return callVLMValidation();
}

int ExplorationFSM::computeRotationAction(double current_yaw, double target_yaw)
{
  // Normalize angles
  while (target_yaw > M_PI) target_yaw -= 2 * M_PI;
  while (target_yaw < -M_PI) target_yaw += 2 * M_PI;
  while (current_yaw > M_PI) current_yaw -= 2 * M_PI;
  while (current_yaw < -M_PI) current_yaw += 2 * M_PI;

  double yaw_diff = target_yaw - current_yaw;

  // Normalize difference to [-PI, PI]
  while (yaw_diff > M_PI) yaw_diff -= 2 * M_PI;
  while (yaw_diff < -M_PI) yaw_diff += 2 * M_PI;

  // Threshold for "aligned" (about 10 degrees)
  const double align_threshold = M_PI / 18.0;

  if (std::fabs(yaw_diff) < align_threshold) {
    return ACTION::STOP;  // Aligned
  } else if (yaw_diff > 0) {
    return ACTION::TURN_LEFT;
  } else {
    return ACTION::TURN_RIGHT;
  }
}

// ==================== Approach Detection Check Implementation ====================

void ExplorationFSM::updateApproachDetectionCheck(double dist_to_target)
{
  const int window_size = FSMConstants::APPROACH_CHECK_STEP_WINDOW;  // 4 steps

  // Check if detector currently sees the target
  bool current_detection = false;
  {
    boost::mutex::scoped_lock lock(detector_mutex_);
    // Consider detection valid if it happened within last 0.5 seconds
    double elapsed = (ros::Time::now() - last_detection_time_).toSec();
    current_detection = last_frame_has_target_ && elapsed < 0.5;
  }

  // Add current detection result to sliding window
  approach_detection_history_.push_back(current_detection);

  // Keep only the last N steps
  while (static_cast<int>(approach_detection_history_.size()) > window_size) {
    approach_detection_history_.pop_front();
  }

  // Log detection status periodically
  if (current_detection) {
    ROS_INFO("[Approach Check] Target DETECTED at dist=%.2fm (window: %zu/%d)",
             dist_to_target, approach_detection_history_.size(), window_size);
  }
}

bool ExplorationFSM::isApproachDetectionValid() const
{
  // Check if target was detected at least once in the sliding window
  for (bool detected : approach_detection_history_) {
    if (detected) {
      return true;
    }
  }
  return false;
}

void ExplorationFSM::resetApproachDetectionCheck()
{
  approach_detection_history_.clear();
  ROS_DEBUG("[Approach Check] State reset");
}

void ExplorationFSM::resetSoftArrivalTracking()
{
  fd_->approach_best_dist_ = std::numeric_limits<double>::max();
  fd_->approach_attempt_count_ = 0;
  fd_->approach_no_improvement_count_ = 0;
}

// ==================== Final Approach Validation Implementation ====================

bool ExplorationFSM::startFinalApproachValidation()
{
  Vector2d current_pos(fd_->odom_pos_(0), fd_->odom_pos_(1));

  // Get the ACTUAL object center, not the planning target position
  // ed_->next_pos_ is where the robot should stand, not the object location
  int current_obj_id = expl_manager_->object_map2d_->getCurrentTargetObjectId();
  Vector2d object_center;

  if (current_obj_id >= 0 &&
      expl_manager_->object_map2d_->getObjectCenter(current_obj_id, object_center)) {
    // Use object center to compute facing direction
    final_approach_target_yaw_ = std::atan2(object_center.y() - current_pos.y(),
                                             object_center.x() - current_pos.x());
    ROS_INFO("[Final Approach] Started validation. Facing object center (%.2f, %.2f), target yaw: %.2f deg",
             object_center.x(), object_center.y(), final_approach_target_yaw_ * 180.0 / M_PI);
  } else {
    // Fallback: keep current orientation if object center unavailable
    final_approach_target_yaw_ = fd_->odom_yaw_;
    ROS_WARN("[Final Approach] Could not get object center (obj_id=%d), keeping current yaw: %.2f deg",
             current_obj_id, final_approach_target_yaw_ * 180.0 / M_PI);
  }

  // Initialize state machine
  final_approach_state_ = FinalApproachState::ROTATING_TO_OBJ;
  final_approach_scan_detected_ = false;
  final_approach_adjustment_count_ = 0;
  final_approach_pitch_offset_ = 0;  // Start at center pitch

  ROS_INFO("[Final Approach] Current pos: (%.2f, %.2f), planning target: (%.2f, %.2f)",
           current_pos.x(), current_pos.y(),
           expl_manager_->ed_->next_pos_.x(), expl_manager_->ed_->next_pos_.y());

  return true;
}

void ExplorationFSM::resetFinalApproachValidation()
{
  final_approach_state_ = FinalApproachState::IDLE;
  final_approach_scan_detected_ = false;
  final_approach_adjustment_count_ = 0;
  final_approach_pitch_offset_ = 0;
  ROS_DEBUG("[Final Approach] State reset");
}

bool ExplorationFSM::checkScanDetection() const
{
  boost::mutex::scoped_lock lock(const_cast<boost::mutex&>(detector_mutex_));
  double elapsed = (ros::Time::now() - last_detection_time_).toSec();
  // Consider detection valid if it happened within the scan wait time
  return last_frame_has_target_ && elapsed < FSMConstants::FINAL_APPROACH_SCAN_WAIT + 0.1;
}

int ExplorationFSM::processFinalApproachValidation(double current_yaw)
{
  const double scan_wait = FSMConstants::FINAL_APPROACH_SCAN_WAIT;
  // Note: yaw_threshold and yaw_adjust are defined in FSMConstants but not used currently
  // They are kept for potential future use in more precise yaw control

  // Check for detection at any point during the process
  if (checkScanDetection()) {
    final_approach_scan_detected_ = true;
    ROS_INFO("[Final Approach] Target DETECTED during scan!");
  }

  switch (final_approach_state_) {
    case FinalApproachState::IDLE: {
      // Start validation
      startFinalApproachValidation();
      return -1;  // In progress
    }

    case FinalApproachState::ROTATING_TO_OBJ: {
      // Rotate to face the target object
      int action = computeRotationAction(current_yaw, final_approach_target_yaw_);
      if (action == ACTION::STOP) {
        // Aligned with target, start look-DOWN scan first (objects are usually on ground)
        ROS_INFO("[Final Approach] Aligned with target. Starting look-DOWN scan first (objects usually on ground).");
        // Set up for LOOK_DOWN
        final_approach_pitch_offset_ = 0;
        // Execute LOOK_DOWN action immediately
        fd_->newest_action_ = ACTION::TURN_DOWN;
        final_approach_pitch_offset_--;
        final_approach_state_ = FinalApproachState::WAIT_LOOK_DOWN;
        final_approach_scan_start_time_ = ros::Time::now();
        ROS_INFO("[Final Approach] Looking DOWN... (pitch_offset: %d)", final_approach_pitch_offset_);
      } else {
        fd_->newest_action_ = action;
      }
      return -1;  // In progress
    }

    case FinalApproachState::LOOK_DOWN: {
      // Execute look down action - look down once from center
      // (This state is only reached if we need additional look-down actions)
      fd_->newest_action_ = ACTION::TURN_DOWN;
      final_approach_pitch_offset_--;
      final_approach_state_ = FinalApproachState::WAIT_LOOK_DOWN;
      final_approach_scan_start_time_ = ros::Time::now();
      ROS_INFO("[Final Approach] Looking DOWN... (pitch_offset: %d)", final_approach_pitch_offset_);
      return -1;
    }

    case FinalApproachState::WAIT_LOOK_DOWN: {
      // Wait for detection after looking down
      double elapsed = (ros::Time::now() - final_approach_scan_start_time_).toSec();
      if (elapsed >= scan_wait || final_approach_scan_detected_) {
        // Now look up: need to go up twice (down->center->up)
        final_approach_state_ = FinalApproachState::LOOK_UP;
        ROS_INFO("[Final Approach] Look-down done (detected: %s). Looking UP...",
                 final_approach_scan_detected_ ? "YES" : "NO");
      }
      return -1;
    }

    case FinalApproachState::LOOK_UP: {
      // Execute look up action
      fd_->newest_action_ = ACTION::TURN_UP;
      final_approach_pitch_offset_++;
      final_approach_state_ = FinalApproachState::WAIT_LOOK_UP;
      final_approach_scan_start_time_ = ros::Time::now();
      ROS_INFO("[Final Approach] Looking UP... (pitch_offset: %d)", final_approach_pitch_offset_);
      return -1;
    }

    case FinalApproachState::WAIT_LOOK_UP: {
      double elapsed = (ros::Time::now() - final_approach_scan_start_time_).toSec();
      if (elapsed >= scan_wait || final_approach_scan_detected_) {
        // Check if we need to continue looking up (from down position, need 2 up actions)
        // pitch_offset: -1 (down) -> 0 (center) -> +1 (up)
        if (final_approach_pitch_offset_ < 1) {
          // Need one more LOOK_UP to reach up position
          final_approach_state_ = FinalApproachState::LOOK_UP;
          ROS_INFO("[Final Approach] Continuing to look UP (pitch_offset: %d)...", final_approach_pitch_offset_);
        } else {
          // Reached up position, return to center
          final_approach_state_ = FinalApproachState::LOOK_CENTER;
          ROS_INFO("[Final Approach] Look-up done (detected: %s). Returning to center...",
                   final_approach_scan_detected_ ? "YES" : "NO");
        }
      }
      return -1;
    }

    case FinalApproachState::LOOK_CENTER: {
      // Return to center view (look down once from up position)
      fd_->newest_action_ = ACTION::TURN_DOWN;
      final_approach_pitch_offset_--;
      final_approach_state_ = FinalApproachState::WAIT_LOOK_CENTER;
      final_approach_scan_start_time_ = ros::Time::now();
      ROS_INFO("[Final Approach] Returning to center... (pitch_offset: %d)", final_approach_pitch_offset_);
      return -1;
    }

    case FinalApproachState::WAIT_LOOK_CENTER: {
      double elapsed = (ros::Time::now() - final_approach_scan_start_time_).toSec();
      if (elapsed >= scan_wait || final_approach_scan_detected_) {
        // Check if we detected anything so far or in sliding window
        if (final_approach_scan_detected_ || isApproachDetectionValid()) {
          ROS_INFO("[Final Approach] SUCCESS! Scan detected: %s, Window detected: %s",
                   final_approach_scan_detected_ ? "YES" : "NO",
                   isApproachDetectionValid() ? "YES" : "NO");
          // Check if pitch needs to be reset before completing
          if (final_approach_pitch_offset_ != 0) {
            ROS_INFO("[Final Approach] Resetting pitch from %d to center before completing...",
                     final_approach_pitch_offset_);
            final_approach_state_ = FinalApproachState::RESET_PITCH;
          } else {
            final_approach_state_ = FinalApproachState::COMPLETED;
            return FINAL_RESULT::REACH_OBJECT;
          }
        } else {
          // No detection yet, try left/right micro-adjustment
          ROS_INFO("[Final Approach] No detection in up/down scan. Trying left adjustment...");
          final_approach_state_ = FinalApproachState::ADJUST_LEFT;
        }
      }
      return -1;
    }

    case FinalApproachState::ADJUST_LEFT: {
      // Micro-adjust yaw to the left
      fd_->newest_action_ = ACTION::TURN_LEFT;
      final_approach_state_ = FinalApproachState::WAIT_ADJUST_LEFT;
      final_approach_scan_start_time_ = ros::Time::now();
      final_approach_adjustment_count_++;
      ROS_INFO("[Final Approach] Adjusting LEFT (attempt %d)...", final_approach_adjustment_count_);
      return -1;
    }

    case FinalApproachState::WAIT_ADJUST_LEFT: {
      double elapsed = (ros::Time::now() - final_approach_scan_start_time_).toSec();
      if (elapsed >= scan_wait || final_approach_scan_detected_) {
        if (final_approach_scan_detected_ || isApproachDetectionValid()) {
          ROS_INFO("[Final Approach] SUCCESS after left adjustment!");
          // Check if pitch needs to be reset before completing
          if (final_approach_pitch_offset_ != 0) {
            ROS_INFO("[Final Approach] Resetting pitch from %d to center before completing...",
                     final_approach_pitch_offset_);
            final_approach_state_ = FinalApproachState::RESET_PITCH;
          } else {
            final_approach_state_ = FinalApproachState::COMPLETED;
            return FINAL_RESULT::REACH_OBJECT;
          }
        } else {
          // Try right adjustment (turn right twice to go from left to right of center)
          final_approach_state_ = FinalApproachState::ADJUST_RIGHT;
          ROS_INFO("[Final Approach] No detection. Trying right adjustment...");
        }
      }
      return -1;
    }

    case FinalApproachState::ADJUST_RIGHT: {
      // Micro-adjust yaw to the right (need to turn right twice: left->center->right)
      fd_->newest_action_ = ACTION::TURN_RIGHT;
      final_approach_state_ = FinalApproachState::WAIT_ADJUST_RIGHT;
      final_approach_scan_start_time_ = ros::Time::now();
      final_approach_adjustment_count_++;
      ROS_INFO("[Final Approach] Adjusting RIGHT (attempt %d)...", final_approach_adjustment_count_);
      return -1;
    }

    case FinalApproachState::WAIT_ADJUST_RIGHT: {
      double elapsed = (ros::Time::now() - final_approach_scan_start_time_).toSec();
      if (elapsed >= scan_wait || final_approach_scan_detected_) {
        if (final_approach_scan_detected_ || isApproachDetectionValid()) {
          ROS_INFO("[Final Approach] SUCCESS after right adjustment!");
          // Check if pitch needs to be reset before completing
          if (final_approach_pitch_offset_ != 0) {
            ROS_INFO("[Final Approach] Resetting pitch from %d to center before completing...",
                     final_approach_pitch_offset_);
            final_approach_state_ = FinalApproachState::RESET_PITCH;
          } else {
            final_approach_state_ = FinalApproachState::COMPLETED;
            return FINAL_RESULT::REACH_OBJECT;
          }
        } else {
          // Check if we've done enough right adjustments (left->center->right = 2 turns)
          if (final_approach_adjustment_count_ < 3) {
            // Need one more right turn to complete the sweep
            final_approach_state_ = FinalApproachState::ADJUST_RIGHT;
          } else {
            // Return to center
            final_approach_state_ = FinalApproachState::ADJUST_CENTER;
            ROS_INFO("[Final Approach] No detection after left/right adjustments. Returning to center...");
          }
        }
      }
      return -1;
    }

    case FinalApproachState::ADJUST_CENTER: {
      // Return to center yaw (turn left once from right position)
      fd_->newest_action_ = ACTION::TURN_LEFT;
      // Final check after returning to center
      if (final_approach_scan_detected_ || isApproachDetectionValid()) {
        ROS_INFO("[Final Approach] SUCCESS! (final check)");
        // Check if pitch needs to be reset before completing
        if (final_approach_pitch_offset_ != 0) {
          ROS_INFO("[Final Approach] Resetting pitch from %d to center before completing...",
                   final_approach_pitch_offset_);
          final_approach_state_ = FinalApproachState::RESET_PITCH;
        } else {
          final_approach_state_ = FinalApproachState::COMPLETED;
          return FINAL_RESULT::REACH_OBJECT;
        }
      } else {
        // All scans failed
        ROS_WARN("[Final Approach] FAILED! No detection during entire scan sequence.");
        final_approach_state_ = FinalApproachState::COMPLETED;
        return -2;  // Validation failed
      }
      return -1;
    }

    case FinalApproachState::RESET_PITCH: {
      // Reset pitch to center before completing successfully
      if (final_approach_pitch_offset_ > 0) {
        // Looking up, need to look down
        fd_->newest_action_ = ACTION::TURN_DOWN;
        final_approach_pitch_offset_--;
        ROS_INFO("[Final Approach] Resetting pitch: TURN_DOWN (pitch_offset: %d)", final_approach_pitch_offset_);
      } else if (final_approach_pitch_offset_ < 0) {
        // Looking down, need to look up
        fd_->newest_action_ = ACTION::TURN_UP;
        final_approach_pitch_offset_++;
        ROS_INFO("[Final Approach] Resetting pitch: TURN_UP (pitch_offset: %d)", final_approach_pitch_offset_);
      }
      final_approach_state_ = FinalApproachState::WAIT_RESET_PITCH;
      final_approach_scan_start_time_ = ros::Time::now();
      return -1;
    }

    case FinalApproachState::WAIT_RESET_PITCH: {
      double elapsed = (ros::Time::now() - final_approach_scan_start_time_).toSec();
      if (elapsed >= scan_wait) {
        if (final_approach_pitch_offset_ == 0) {
          // Pitch is now centered, complete successfully
          ROS_INFO("[Final Approach] Pitch reset complete. SUCCESS!");
          final_approach_state_ = FinalApproachState::COMPLETED;
          return FINAL_RESULT::REACH_OBJECT;
        } else {
          // Need more pitch adjustments
          final_approach_state_ = FinalApproachState::RESET_PITCH;
        }
      }
      return -1;
    }

    case FinalApproachState::COMPLETED: {
      // Should not reach here normally
      if (final_approach_scan_detected_ || isApproachDetectionValid()) {
        return FINAL_RESULT::REACH_OBJECT;
      }
      return -2;
    }

    default:
      ROS_ERROR("[Final Approach] Unknown state: %d", static_cast<int>(final_approach_state_));
      return -2;
  }
}

// ==================== VLM Approach Verification Implementation ====================

void ExplorationFSM::vlmVerificationResultCallback(const plan_env::VLMVerificationResultConstPtr& msg)
{
  // Check if we are waiting for VLM result
  if (!fd_->vlm_waiting_) {
    ROS_WARN("[VLM] Received result but not waiting, ignoring (object_id=%d)", msg->object_id);
    return;
  }

  // Check if this is for the correct object
  if (msg->object_id != fd_->vlm_target_object_id_) {
    ROS_WARN("[VLM] Received result for wrong object (expected %d, got %d)",
             fd_->vlm_target_object_id_, msg->object_id);
    return;
  }

  // 3-level decision: 1=CONFIRM, 0=UNCERTAIN, -1=REJECT
  const char* decision_names[] = {"REJECT", "UNCERTAIN", "CONFIRM"};
  int decision_idx = msg->decision_level + 1;  // Map -1,0,1 to 0,1,2
  if (decision_idx < 0 || decision_idx > 2) decision_idx = 1;  // Default to UNCERTAIN

  ROS_INFO("[VLM] Received verification result: object_id=%d, decision=%s (%d), conf=%.2f, reason=%s",
           msg->object_id, decision_names[decision_idx], msg->decision_level,
           msg->vlm_confidence, msg->reason.c_str());

  // Release the waiting lock
  fd_->vlm_waiting_ = false;

  // Check for timeout
  if (msg->timeout) {
    ROS_ERROR("[VLM] Verification timed out (%.1fs)! Disabling VLM for this episode.",
              msg->duration);
    fd_->vlm_disabled_this_episode_ = true;
    return;
  }

  // Apply VLM result to ObjectMap with 3-level decision
  if (expl_manager_ && expl_manager_->sdf_map_ && expl_manager_->sdf_map_->object_map2d_) {
    expl_manager_->sdf_map_->object_map2d_->applyVLMVerificationResult(
        msg->object_id, msg->decision_level, msg->vlm_confidence);
  }

  // Handle decision-based actions
  if (msg->decision_level == -1) {
    // REJECT: VLM rejected the object as false positive, trigger replan
    ROS_WARN("[VLM] Object %d REJECTED as false positive, triggering replan", msg->object_id);

    // Release all target locks to allow re-planning to find new targets
    if (expl_manager_->isSuspiciousTargetLocked()) {
      ROS_INFO("[VLM] Releasing suspicious target lock after REJECT");
      expl_manager_->setSuspiciousTargetLock(false);
    }
    if (expl_manager_->isObjectApproachLocked()) {
      ROS_INFO("[VLM] Releasing object approach lock after REJECT");
      expl_manager_->setObjectApproachLock(false);
    }

    // Exit VLM collection phase if active
    if (vlm_collection_locked_) {
      ROS_INFO("[VLM] Exiting collection phase after REJECT");
      exitCollectionPhase(false);  // This will also mark object as invalid
    }

    fd_->replan_flag_ = true;
  } else if (msg->decision_level == 0) {
    // UNCERTAIN: Keep original confidence, mark as verified to prevent re-triggering
    ROS_INFO("[VLM] Object %d UNCERTAIN, marking as verified (uncertain) to prevent re-verification loop", msg->object_id);
    // Object is now marked vlm_verified_=true, vlm_uncertain_=true in applyVLMVerificationResult
    // This prevents infinite re-verification while allowing navigation to continue
  } else {
    // CONFIRM: Object confirmed, continue approaching
    ROS_INFO("[VLM] Object %d CONFIRMED as real target, continuing approach", msg->object_id);
  }
}

void ExplorationFSM::triggerVLMVerification(int object_id, int trigger_type, double distance_to_target)
{
  // Check if VLM is disabled for this episode
  if (fd_->vlm_disabled_this_episode_) {
    ROS_WARN("[VLM] VLM disabled for this episode (previous timeout), skipping verification");
    return;
  }

  // Check if object is already VLM verified
  if (expl_manager_ && expl_manager_->sdf_map_ && expl_manager_->sdf_map_->object_map2d_) {
    if (expl_manager_->sdf_map_->object_map2d_->isObjectVLMVerified(object_id)) {
      ROS_INFO("[VLM] Object %d already VLM verified, skipping", object_id);
      return;
    }
  }

  ROS_INFO("[VLM] Triggering verification: object_id=%d, trigger_type=%d, distance=%.2f",
           object_id, trigger_type, distance_to_target);

  // Set waiting state - FSM will stop outputting actions
  fd_->vlm_waiting_ = true;
  fd_->vlm_target_object_id_ = object_id;
  fd_->vlm_trigger_type_ = trigger_type;
  fd_->vlm_used_this_episode_ = true;
  fd_->vlm_verify_count_++;

  // Mark object as VLM pending
  if (expl_manager_ && expl_manager_->sdf_map_ && expl_manager_->sdf_map_->object_map2d_) {
    expl_manager_->sdf_map_->object_map2d_->setObjectVLMPending(object_id, true);
  }

  // Publish verification request to Python
  plan_env::VLMVerificationRequest req_msg;
  req_msg.object_id = object_id;
  // Get target category from ObjectMap2D (correctly subscribed to /habitat/semantic_scores)
  req_msg.target_category = expl_manager_->sdf_map_->object_map2d_->getTargetCategory();
  req_msg.trigger_type = trigger_type;
  req_msg.distance_to_target = distance_to_target;

  // Fill in object map confidence info for VLM prompt
  double fused_conf = 0.0;
  int obs_count = 0;
  double threshold = 0.0;
  if (expl_manager_->sdf_map_->object_map2d_->getObjectInfoForVLM(
          object_id, fused_conf, obs_count, threshold)) {
    req_msg.target_fused_confidence = fused_conf;
    req_msg.target_observation_count = obs_count;
    req_msg.current_threshold = threshold;
    req_msg.similar_objects_info = expl_manager_->sdf_map_->object_map2d_->getSimilarObjectsInfoForVLM(object_id);
  } else {
    req_msg.target_fused_confidence = 0.0;
    req_msg.target_observation_count = 0;
    req_msg.current_threshold = 0.0;
    req_msg.similar_objects_info = "";
  }

  req_msg.header.stamp = ros::Time::now();
  vlm_request_pub_.publish(req_msg);

  ROS_INFO("[VLM] Verification request sent: object_id=%d, target='%s', distance=%.2f, "
           "fused_conf=%.3f, obs_count=%d, threshold=%.3f",
           object_id, req_msg.target_category.c_str(), distance_to_target,
           fused_conf, obs_count, threshold);
}

void ExplorationFSM::publishObjectUpdateInfo(int object_id, double confidence,
                                              double distance, bool is_current_target)
{
  plan_env::ObjectUpdateInfo msg;
  msg.object_id = object_id;
  msg.detection_confidence = confidence;
  msg.distance_to_object = distance;
  msg.is_current_target = is_current_target;
  msg.header.stamp = ros::Time::now();
  object_update_pub_.publish(msg);
}

bool ExplorationFSM::checkVLMTriggerConditions(double dist_to_target, int object_id)
{
  // Skip if VLM disabled or already waiting
  if (fd_->vlm_disabled_this_episode_ || fd_->vlm_waiting_) {
    return false;
  }

  // Skip if object already verified
  if (expl_manager_ && expl_manager_->sdf_map_ && expl_manager_->sdf_map_->object_map2d_) {
    if (expl_manager_->sdf_map_->object_map2d_->isObjectVLMVerified(object_id)) {
      return false;
    }
  }

  // Always allow trigger if object not verified (for rescue mode at any distance)
  return true;
}

void ExplorationFSM::resetVLMVerificationState()
{
  fd_->vlm_waiting_ = false;
  fd_->vlm_target_object_id_ = -1;
  fd_->vlm_trigger_type_ = 0;
  fd_->vlm_disabled_this_episode_ = false;
  fd_->vlm_used_this_episode_ = false;
  fd_->vlm_verify_count_ = 0;
  ROS_DEBUG("[VLM] Verification state reset");
}

}  // namespace apexnav_planner
