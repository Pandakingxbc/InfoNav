/**
 * @file exploration_manager.cpp
 * @brief Implementation of exploration manager for autonomous semantic navigation
 * @author Zager-Zhang
 *
 * This file implements the ExplorationManager class that handles various
 * exploration strategies including distance-based, semantic-based, hybrid,
 * and TSP-optimized frontier selection for autonomous robot exploration.
 */

#include <exploration_manager/exploration_manager.h>
#include <exploration_manager/exploration_data.h>
#include <lkh_mtsp_solver/SolveMTSP.h>
#include <plan_env/map_ros.h>

#include <algorithm>
#include <numeric>

using namespace Eigen;

namespace apexnav_planner {

ExplorationManager::~ExplorationManager() = default;

void ExplorationManager::initialize(ros::NodeHandle& nh)
{
  // Initialize SDF map and get object map reference
  sdf_map_.reset(new SDFMap2D);
  sdf_map_->initMap(nh);
  object_map2d_ = sdf_map_->object_map2d_;

  // Initialize frontier map and path finder
  frontier_map2d_.reset(new FrontierMap2D(sdf_map_, nh));
  path_finder_.reset(new Astar2D);
  path_finder_->init(nh, sdf_map_);

  // Initialize exploration data and parameter containers
  ed_.reset(new ExplorationData);
  ep_.reset(new ExplorationParam);

  // Load exploration parameters from ROS parameter server
  nh.param("exploration/policy", ep_->policy_mode_, 0);
  nh.param("exploration/top_k_value", ep_->top_k_value_, 5);  // Default: top 5 value frontiers for TSP
  nh.param("exploration/tsp_dir", ep_->tsp_dir_, string("null"));

  // Dual-Value Fusion parameters (Paper Section III-E)
  // V_total(f_i) = alpha * V_sem(f_i) + (1-alpha) * V_ig(f_i)
  nh.param("exploration/fusion_alpha", ep_->fusion_alpha_, 0.5);  // Default: equal weight
  nh.param("exploration/use_ig_fusion", ep_->use_ig_fusion_, true);  // Enable IG fusion by default

  // Adaptive Alpha parameters (Paper Equation)
  // alpha(t) = alpha_base + (1 - alpha_base) * max_fi(V_HSVM(fi)) / tau_conf
  nh.param("exploration/use_adaptive_alpha", ep_->use_adaptive_alpha_, true);  // Enable by default
  nh.param("exploration/alpha_base", ep_->alpha_base_, 0.8);  // Base alpha when no strong semantic cues
  nh.param("exploration/tau_conf", ep_->tau_conf_, 0.8);  // Confidence threshold

  ROS_INFO("[ExplorationManager] Dual-Value Fusion: alpha=%.2f, use_ig=%s, adaptive=%s (base=%.2f, tau=%.2f)",
      ep_->fusion_alpha_, ep_->use_ig_fusion_ ? "true" : "false",
      ep_->use_adaptive_alpha_ ? "true" : "false", ep_->alpha_base_, ep_->tau_conf_);

  // WTRP (Weighted Traveling Repairman Problem) parameters
  nh.param("exploration/wtrp_temperature", ep_->wtrp_temperature_, 0.5);
  nh.param("exploration/wtrp_max_brute_force", ep_->wtrp_max_brute_force_, 10);
  nh.param("exploration/wtrp_hysteresis_ratio", ep_->wtrp_hysteresis_ratio_, 0.8);
  ROS_INFO("[ExplorationManager] WTRP: temperature=%.2f, max_brute_force=%d, hysteresis=%.2f",
      ep_->wtrp_temperature_, ep_->wtrp_max_brute_force_, ep_->wtrp_hysteresis_ratio_);

  // Get map parameters for ray casting initialization
  double resolution = sdf_map_->getResolution();
  Eigen::Vector2d origin, size;
  sdf_map_->getRegion(origin, size);

  // Initialize ray caster for collision checking and TSP service client
  ray_caster2d_.reset(new RayCaster2D);
  ray_caster2d_->setParams(resolution, origin);
  tsp_client_ = nh.serviceClient<lkh_mtsp_solver::SolveMTSP>("/solve_tsp", true);

  // Initialize suspicious target locking state
  suspicious_target_locked_ = false;
  locked_suspicious_pos_ = Vector2d(0, 0);
  locked_suspicious_object_id_ = -1;

  // Initialize object approach point locking state
  object_approach_locked_ = false;
  locked_object_approach_pos_ = Vector2d(0, 0);
  locked_object_id_ = -1;

  // Initialize WTRP target hysteresis state
  has_last_wtrp_target_ = false;
  last_wtrp_target_ = Vector2d(0, 0);
  last_wtrp_target_value_ = 0.0;
}

void ExplorationManager::setSuspiciousTargetLock(bool locked, const Vector2d& pos, int object_id)
{
  suspicious_target_locked_ = locked;
  if (locked) {
    locked_suspicious_pos_ = pos;
    locked_suspicious_object_id_ = object_id;
    // Set current target for VLM frame collection (if object_id is valid)
    if (object_id >= 0 && sdf_map_) {
      sdf_map_->object_map2d_->setCurrentTargetObjectId(object_id);
    }
    ROS_WARN("[Target Lock] Locked suspicious target at (%.2f, %.2f), object_id=%d", pos(0), pos(1), object_id);
  } else {
    locked_suspicious_object_id_ = -1;
    ROS_WARN("[Target Lock] Unlocked suspicious target");
  }
}

void ExplorationManager::setObjectApproachLock(bool locked, const Vector2d& pos, int object_id)
{
  object_approach_locked_ = locked;
  if (locked) {
    locked_object_approach_pos_ = pos;
    locked_object_id_ = object_id;
    ROS_WARN("[Approach Lock] Locked approach point at (%.2f, %.2f) for object id=%d",
             pos(0), pos(1), object_id);
  } else {
    locked_object_approach_pos_ = Vector2d(0, 0);
    locked_object_id_ = -1;
    ROS_WARN("[Approach Lock] Unlocked approach point");
  }
}

void ExplorationManager::resetWTRPHysteresis()
{
  if (has_last_wtrp_target_) {
    ROS_INFO("[WTRP Hysteresis] Reset: clearing locked target (%.2f, %.2f)",
             last_wtrp_target_.x(), last_wtrp_target_.y());
  }
  has_last_wtrp_target_ = false;
  last_wtrp_target_ = Vector2d(0, 0);
  last_wtrp_target_value_ = 0.0;
}

int ExplorationManager::planNextBestPoint(const Vector3d& pos, const double& yaw)
{
  Vector2d pos2d = Vector2d(pos(0), pos(1));
  ros::Time t1 = ros::Time::now();
  auto t2 = t1;

  // Clear previous planning results
  ed_->tsp_tour_.clear();
  ed_->next_best_path_.clear();
  vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> object_clouds;
  sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds);

  // Get object IDs for VLM validation tracking (same order as object_clouds)
  std::vector<int> object_ids;
  sdf_map_->object_map2d_->getTopConfidenceObjectIds(object_ids);

  // ==================== Navigation Mode: High-Confidence Objects ====================
  // High-confidence object has highest priority, unlocks any suspicious target
  if (!object_clouds.empty()) {
    ROS_WARN("[Navigation Mode] Get object_cloud num = %ld", object_clouds.size());

    // Unlock suspicious target when high-confidence object is found
    if (suspicious_target_locked_) {
      setSuspiciousTargetLock(false);
    }

    // ==================== Approach Point Locking: Prevent Oscillation ====================
    // If we have a locked approach point, try to continue navigating to it first
    // This prevents oscillation between different candidate points of the same object
    if (object_approach_locked_) {
      // Check if the locked object is still in the high-confidence list AND not marked unreachable
      bool locked_object_still_valid = false;
      for (size_t i = 0; i < object_ids.size(); ++i) {
        if (object_ids[i] == locked_object_id_) {
          locked_object_still_valid = true;
          break;
        }
      }

      // Also check if the locked object has been marked as unreachable
      if (locked_object_still_valid && object_map2d_->isObjectUnreachable(locked_object_id_)) {
        ROS_WARN("[Approach Lock] Locked object id=%d is now UNREACHABLE, unlocking", locked_object_id_);
        locked_object_still_valid = false;
        setObjectApproachLock(false);
      }

      if (locked_object_still_valid) {
        // Try to find path to the locked approach point
        path_finder_->reset();
        double dist_to_locked = (pos2d - locked_object_approach_pos_).norm();

        // Use appropriate safety distance based on current distance
        double safety_dist = (dist_to_locked < 0.8) ? 0.2 : 0.5;

        if (path_finder_->astarSearch(pos2d, locked_object_approach_pos_, safety_dist, 0.2) ==
            Astar2D::REACH_END) {
          ed_->next_pos_ = locked_object_approach_pos_;
          ed_->next_best_path_ = path_finder_->getPath();
          sdf_map_->object_map2d_->setCurrentTargetObjectId(locked_object_id_);
          ROS_WARN("[Approach Lock] Continuing to locked point (%.2f, %.2f), dist=%.2fm",
                   locked_object_approach_pos_(0), locked_object_approach_pos_(1), dist_to_locked);
          return SEARCH_BEST_OBJECT;
        } else {
          // Cannot reach locked point anymore, unlock and try other candidates
          ROS_WARN("[Approach Lock] Cannot reach locked point, unlocking and re-selecting");
          setObjectApproachLock(false);
        }
      } else {
        // Locked object no longer in high-confidence list, unlock
        ROS_WARN("[Approach Lock] Locked object id=%d no longer valid, unlocking", locked_object_id_);
        setObjectApproachLock(false);
      }
    }

    // Try to find path to each detected object in order of confidence
    for (size_t i = 0; i < object_clouds.size(); ++i) {
      int obj_id = (i < object_ids.size()) ? object_ids[i] : -1;

      // Skip objects that have been marked as unreachable (too many failed approach attempts)
      if (obj_id >= 0 && object_map2d_->isObjectUnreachable(obj_id)) {
        ROS_WARN_THROTTLE(5.0, "[Navigation Mode] Skipping UNREACHABLE object id=%d, trying next candidate", obj_id);
        continue;
      }

      if (searchObjectPath(pos, object_clouds[i], ed_->next_pos_, ed_->next_best_path_, obj_id)) {
        // Record the current target object ID for VLM validation
        if (obj_id >= 0) {
          sdf_map_->object_map2d_->setCurrentTargetObjectId(obj_id);
          ROS_INFO("[Navigation Mode] Navigating to object id=%d", obj_id);
        }

        // Lock the selected approach point to prevent oscillation
        if (!object_approach_locked_) {
          setObjectApproachLock(true, ed_->next_pos_, obj_id);
        }

        return SEARCH_BEST_OBJECT;
      }
    }
  }

  // ==================== Navigation Mode: Over-Depth Objects ====================
  if (!object_map2d_->over_depth_object_cloud_->points.empty()) {
    ROS_WARN("[Navigation Mode (Over Depth)] Get over depth object cloud");

    // Unlock suspicious target when over-depth object is found
    if (suspicious_target_locked_) {
      setSuspiciousTargetLock(false);
    }

    if (searchObjectPath(
            pos, object_map2d_->over_depth_object_cloud_, ed_->next_pos_, ed_->next_best_path_))
      return SEARCH_OVER_DEPTH_OBJECT;
  }

  // ==================== Locked Suspicious Target Mode ====================
  // If we have a locked suspicious target, try to navigate to it (replan path only)
  if (suspicious_target_locked_) {
    // Try to find path to locked position
    path_finder_->reset();
    if (path_finder_->astarSearch(pos2d, locked_suspicious_pos_, 0.5, 0.2) == Astar2D::REACH_END) {
      ed_->next_pos_ = locked_suspicious_pos_;
      ed_->next_best_path_ = path_finder_->getPath();
      // Update current target for VLM frame collection
      if (locked_suspicious_object_id_ >= 0 && sdf_map_) {
        sdf_map_->object_map2d_->setCurrentTargetObjectId(locked_suspicious_object_id_);
      }
      ROS_WARN("[Target Lock] Navigating to locked target at (%.2f, %.2f)",
               locked_suspicious_pos_(0), locked_suspicious_pos_(1));
      return SEARCH_SUSPICIOUS_OBJECT;
    } else {
      // Cannot reach locked target anymore, unlock it
      ROS_WARN("[Target Lock] Cannot reach locked target, unlocking");
      setSuspiciousTargetLock(false);
    }
  }

  // ==================== Exploration Mode: Frontier-Based Planning ====================
  sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds, false);
  std::vector<int> top_object_ids;
  sdf_map_->object_map2d_->getTopConfidenceObjectIds(top_object_ids, false);
  pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> top_object_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  int top_object_id = -1;
  if (object_clouds.size() >= 1) {
    top_object_cloud = object_clouds[0];
    if (top_object_ids.size() >= 1) {
      top_object_id = top_object_ids[0];
    }
  }

  // Apply selected exploration policy to choose next frontier
  Eigen::Vector2d next_best_pos;
  std::vector<Eigen::Vector2d> next_best_path;
  chooseExplorationPolicy(pos2d, ed_->frontier_averages_, next_best_pos, next_best_path);

  // Handle case when no passable frontiers are found
  if (next_best_path.empty()) {
    ROS_WARN("Maybe no passable frontier.");

    // Try suspicious objects as backup - and LOCK the target
    if (!top_object_cloud->points.empty() &&
        searchObjectPath(pos, top_object_cloud, ed_->next_pos_, ed_->next_best_path_)) {
      // Lock this suspicious target to prevent oscillation (pass object_id for VLM frame collection)
      setSuspiciousTargetLock(true, ed_->next_pos_, top_object_id);
      return SEARCH_SUSPICIOUS_OBJECT;
    }
    else
      // Try dormant frontiers as last resort
      chooseExplorationPolicy(
          pos2d, ed_->dormant_frontier_averages_, next_best_pos, next_best_path);

    // Extreme search mode when all normal options fail
    if (next_best_path.empty()) {
      ROS_ERROR("search exterme case!!!");

      // Try extreme object search with relaxed constraints
      for (auto object_cloud : object_clouds) {
        if (!object_cloud->points.empty() &&
            searchObjectPathExtreme(pos, object_cloud, ed_->next_pos_, ed_->next_best_path_))
          return SEARCH_EXTREME;
      }

      // Include lower confidence objects in extreme search
      sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds, false, true);
      for (auto object_cloud : object_clouds) {
        if (!object_cloud->points.empty() &&
            searchObjectPathExtreme(pos, object_cloud, ed_->next_pos_, ed_->next_best_path_))
          return SEARCH_EXTREME;
      }

      // Try cached over-depth objects as final option
      static auto last_over_depth_object_cloud = object_map2d_->over_depth_object_cloud_;
      if (!object_map2d_->over_depth_object_cloud_->points.empty())
        last_over_depth_object_cloud = object_map2d_->over_depth_object_cloud_;

      if (!last_over_depth_object_cloud->points.empty() &&
          searchObjectPathExtreme(
              pos, last_over_depth_object_cloud, ed_->next_pos_, ed_->next_best_path_)) {
        return SEARCH_EXTREME;
      }
    }

    // ==================== Similar Objects Fallback ====================
    // If extreme search still failed, try visiting similar objects (best_label != 0)
    // to attempt re-detection of target object from different angles
    if (next_best_path.empty()) {
      ROS_WARN("[Similar Object] Extreme search failed, trying similar objects...");

      std::vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> similar_clouds;
      std::vector<int> similar_ids;
      sdf_map_->object_map2d_->getSimilarObjectClouds(similar_clouds, similar_ids);

      for (size_t i = 0; i < similar_clouds.size(); ++i) {
        if (similar_clouds[i]->empty()) continue;

        int obj_id = (i < similar_ids.size()) ? similar_ids[i] : -1;

        // Try normal path search first
        if (searchObjectPath(pos, similar_clouds[i], ed_->next_pos_, ed_->next_best_path_, obj_id)) {
          ROS_WARN("[Similar Object] Navigating to similar object id=%d for re-detection", obj_id);
          sdf_map_->object_map2d_->setCurrentTargetObjectId(obj_id);
          return SEARCH_SIMILAR_OBJECT;
        }

        // Try extreme mode if normal search fails
        if (searchObjectPathExtreme(pos, similar_clouds[i], ed_->next_pos_, ed_->next_best_path_)) {
          ROS_WARN("[Similar Object Extreme] Navigating to similar object id=%d", obj_id);
          sdf_map_->object_map2d_->setCurrentTargetObjectId(obj_id);
          return SEARCH_SIMILAR_OBJECT;
        }
      }
    }

    // Final error handling when no valid targets exist
    if (next_best_path.empty()) {
      if (ed_->frontiers_.empty()) {
        ROS_ERROR("No coverable frontier!!");
        return NO_COVERABLE_FRONTIER;
      }
      else {
        ROS_ERROR("No passable frontier!!");
        return NO_PASSABLE_FRONTIER;
      }
    }
  }

  // Found valid frontier - unlock any target locks since we're switching to exploration mode
  if (suspicious_target_locked_) {
    setSuspiciousTargetLock(false);
  }
  if (object_approach_locked_) {
    ROS_WARN("[Approach Lock] Unlocking due to switch to Frontier exploration");
    setObjectApproachLock(false);
  }

  // Store successful planning results
  ed_->next_pos_ = next_best_pos;
  ed_->next_best_path_ = next_best_path;

  // Performance monitoring
  double total_time = (ros::Time::now() - t2).toSec();
  ROS_ERROR_COND(total_time > 0.25, "[Plan NBV] Total time %.2lf s too long!!!", total_time);

  return EXPLORATION;
}

void ExplorationManager::chooseExplorationPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  switch (ep_->policy_mode_) {
    case ExplorationParam::DISTANCE:
      ROS_WARN("[Exploration Mode] Distance (Greedy Closest)");
      findClosestFrontierPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    case ExplorationParam::SEMANTIC:
      ROS_WARN("[Exploration Mode] Value (Greedy Highest)");
      findHighestSemanticsFrontierPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    case ExplorationParam::HYBRID:
      // Log is inside hybridExplorePolicy with top-k count
      hybridExplorePolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    case ExplorationParam::TSP_DIST:
      ROS_WARN("[Exploration Mode] Distance (TSP All Frontiers)");
      findTSPTourPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    case ExplorationParam::WTRP:
      wtrpExplorePolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    default:
      ROS_WARN("[Exploration Mode] Unknown Mode");
      break;
  }
}

void ExplorationManager::hybridExplorePolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  // Get frontiers sorted by value (descending)
  vector<SemanticFrontier> sem_frontiers;
  getSortedSemanticFrontiers(cur_pos, frontiers, sem_frontiers);
  if (sem_frontiers.empty())
    return;

  // Select top-k high value frontiers for TSP optimization
  int k = ep_->top_k_value_;
  vector<Vector2d> top_k_frontiers;
  for (int i = 0; i < min(k, (int)sem_frontiers.size()); i++) {
    top_k_frontiers.push_back(sem_frontiers[i].position);
  }

  ROS_WARN("[Exploration Mode] TSP with top-%d value frontiers", (int)top_k_frontiers.size());
  findTSPTourPolicy(cur_pos, top_k_frontiers, next_best_pos, next_best_path);
}

void ExplorationManager::findHighestSemanticsFrontierPolicy(Vector2d cur_pos,
    vector<Vector2d> frontiers, Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  next_best_path.clear();

  // Container for frontier-value pairs for sorting
  vector<pair<Vector2d, double>> frontier_values;

  // Dual-Value Fusion with Adaptive Alpha (Paper Section III-E)
  // alpha(t) = alpha_base + (1 - alpha_base) * max_fi(V_HSVM(fi)) / tau_conf
  double alpha = computeAdaptiveAlpha(frontiers);

  // Check if IG value map is available and enabled
  bool use_ig = ep_->use_ig_fusion_ &&
                sdf_map_->use_ig_value_map_ &&
                sdf_map_->ig_value_map_;

  // Compute total value for each frontier
  for (auto frontier : frontiers) {
    Vector2i idx;
    sdf_map_->posToIndex(frontier, idx);
    auto nbrs = allNeighbors(idx, 2);  // 5x5 neighborhood

    // Find maximum semantic value in local neighborhood
    double sem_value = sdf_map_->value_map_->getValue(idx);
    for (auto nbr : nbrs) sem_value = max(sem_value, sdf_map_->value_map_->getValue(nbr));

    // Find maximum IG value in local neighborhood (if enabled)
    double ig_value = 0.0;
    if (use_ig) {
      ig_value = sdf_map_->ig_value_map_->getIGValue(idx);
      for (auto nbr : nbrs) ig_value = max(ig_value, sdf_map_->ig_value_map_->getIGValue(nbr));
    }

    // Compute total value using dual-value fusion (HSVM/4 to reduce intensity)
    double total_value = use_ig ? (alpha * (sem_value / 4.0) + (1.0 - alpha) * ig_value) : (sem_value / 4.0);

    frontier_values.emplace_back(frontier, total_value);
  }

  // Sort by semantic value (descending), then by distance (ascending)
  auto compareFrontiers = [&cur_pos](
                              const pair<Vector2d, double>& a, const pair<Vector2d, double>& b) {
    if (fabs(a.second - b.second) > 1e-5) {
      return a.second > b.second;  // Higher semantic value first
    }
    else {
      double dist_a = (a.first - cur_pos).norm();
      double dist_b = (b.first - cur_pos).norm();
      return dist_a < dist_b;  // Closer distance first for tie-breaking
    }
  };

  std::sort(frontier_values.begin(), frontier_values.end(), compareFrontiers);

  // Update frontier list with sorted order
  frontiers.clear();
  for (const auto& fv : frontier_values) {
    frontiers.push_back(fv.first);
  }

  // Select first reachable frontier from sorted list
  for (int i = 0; i < (int)frontiers.size(); i++) {
    std::vector<Eigen::Vector2d> tmp_path;
    Eigen::Vector2d tmp_pos;
    if (!searchFrontierPath(cur_pos, frontiers[i], tmp_pos, tmp_path))
      continue;
    next_best_pos = tmp_pos;
    next_best_path = tmp_path;
    break;
  }
}

void ExplorationManager::findClosestFrontierPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  next_best_path.clear();

  // Sort frontiers by Euclidean distance for efficient processing
  std::sort(frontiers.begin(), frontiers.end(), [&cur_pos](const Vector2d& a, const Vector2d& b) {
    return (a - cur_pos).norm() < (b - cur_pos).norm();
  });

  double min_len = std::numeric_limits<double>::max();

  // Find the frontier with shortest actual path length
  for (int i = 0; i < (int)frontiers.size(); i++) {
    // Skip if Euclidean distance already exceeds best path length
    if ((frontiers[i] - cur_pos).norm() >= min_len)
      continue;

    std::vector<Eigen::Vector2d> tmp_path;
    Eigen::Vector2d tmp_pos;

    // Attempt path planning to this frontier
    if (!searchFrontierPath(cur_pos, frontiers[i], tmp_pos, tmp_path))
      continue;

    // Update best solution if this path is shorter
    double len = Astar2D::pathLength(tmp_path);
    if (len < min_len) {
      min_len = len;
      next_best_pos = tmp_pos;
      next_best_path = tmp_path;
    }
  }
}

void ExplorationManager::findTSPTourPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  next_best_path.clear();
  vector<Vector2d> filter_frontiers;
  for (auto frontier : frontiers) {
    Vector2d tmp_pos;
    vector<Vector2d> tmp_path;
    if (searchFrontierPath(cur_pos, frontier, tmp_pos, tmp_path))
      filter_frontiers.push_back(frontier);
  }

  vector<int> indices;
  computeATSPTour(cur_pos, filter_frontiers, indices);
  ed_->tsp_tour_.push_back(cur_pos);
  for (auto idx : indices) ed_->tsp_tour_.push_back(filter_frontiers[idx]);

  if (!indices.empty()) {
    for (auto idx : indices) {
      Vector2d next_bext_frontier = filter_frontiers[idx];
      if (searchFrontierPath(cur_pos, next_bext_frontier, next_best_pos, next_best_path))
        break;
    }
  }
}

double ExplorationManager::computePathCost(const Vector2d& pos1, const Vector2d& pos2)
{
  path_finder_->reset();
  if (path_finder_->astarSearch(pos1, pos2, 0.25, 0.002) == Astar2D::REACH_END)
    return Astar2D::pathLength(path_finder_->getPath());
  return 10000.0;
}

void ExplorationManager::computeATSPCostMatrix(
    const Vector2d& cur_pos, const vector<Vector2d>& frontiers, Eigen::MatrixXd& mat)
{
  int dimen = frontiers.size() + 1;
  mat.resize(dimen, dimen);

  // Agent to frontiers
  for (int i = 1; i < dimen; i++) {
    mat(0, i) = computePathCost(cur_pos, frontiers[i - 1]);
    mat(i, 0) = 0;
  }

  // Costs between frontiers
  for (int i = 1; i < dimen; ++i) {
    for (int j = i + 1; j < dimen; ++j) {
      double cost = computePathCost(frontiers[i - 1], frontiers[j - 1]);
      mat(i, j) = cost;
      mat(j, i) = cost;
    }
  }

  // Diag
  for (int i = 0; i < dimen; ++i) {
    mat(i, i) = 100000.0;
  }
}

void ExplorationManager::computeATSPTour(
    const Vector2d& cur_pos, const vector<Vector2d>& frontiers, vector<int>& indices)
{
  indices.clear();
  if (frontiers.empty()) {
    ROS_ERROR("No frontier to compute tsp!");
    return;
  }
  else if (frontiers.size() == 1) {
    indices.push_back(0);
    return;
  }
  /* change ATSP to lhk3 */
  auto t1 = ros::Time::now();

  // Get cost matrix for current state and clusters
  Eigen::MatrixXd cost_mat;
  computeATSPCostMatrix(cur_pos, frontiers, cost_mat);
  const int dimension = cost_mat.rows();

  double mat_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Initialize ATSP par file
  // Create problem file
  ofstream file(ep_->tsp_dir_ + "/atsp_tour.atsp");
  file << "NAME : amtsp\n";
  file << "TYPE : ATSP\n";
  file << "DIMENSION : " + to_string(dimension) + "\n";
  file << "EDGE_WEIGHT_TYPE : EXPLICIT\n";
  file << "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n";
  file << "EDGE_WEIGHT_SECTION\n";
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      int int_cost = 100 * cost_mat(i, j);
      file << int_cost << " ";
    }
    file << "\n";
  }
  file.close();

  // Create par file
  const int drone_num = 1;
  file.open(ep_->tsp_dir_ + "/atsp_tour.par");
  file << "SPECIAL\n";
  file << "PROBLEM_FILE = " + ep_->tsp_dir_ + "/atsp_tour.atsp\n";
  file << "SALESMEN = " << to_string(drone_num) << "\n";
  file << "MTSP_OBJECTIVE = MINSUM\n";
  file << "RUNS = 1\n";
  file << "TRACE_LEVEL = 0\n";
  file << "TOUR_FILE = " + ep_->tsp_dir_ + "/atsp_tour.tour\n";
  file.close();

  auto par_dir = ep_->tsp_dir_ + "/atsp_tour.atsp";

  lkh_mtsp_solver::SolveMTSP srv;
  srv.request.prob = 1;
  if (!tsp_client_.call(srv)) {
    ROS_ERROR("Fail to solve ATSP.");
    return;
  }

  // Read optimal tour from the tour section of result file
  ifstream res_file(ep_->tsp_dir_ + "/atsp_tour.tour");
  string res;
  while (getline(res_file, res)) {
    // Go to tour section
    if (res.compare("TOUR_SECTION") == 0)
      break;
  }

  // Read path for ATSP formulation
  while (getline(res_file, res)) {
    // Read indices of frontiers in optimal tour
    int id = stoi(res);
    if (id == 1)  // Ignore the current state
      continue;
    if (id == -1)
      break;
    indices.push_back(id - 2);  // Idx of solver-2 == Idx of frontier
  }

  res_file.close();

  // for (auto idx : indices) ROS_WARN("ATSP idx = %d", idx);

  double tsp_time = (ros::Time::now() - t1).toSec();
  ROS_WARN("[ATSP Tour] Cost mat: %lf, TSP: %lf", mat_time, tsp_time);
}

Vector2d ExplorationManager::findNearestObjectPoint(
    const Vector3d& start, const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud)
{
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(object_cloud);
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);

  pcl::PointXYZ cur_pt;
  cur_pt.x = start(0);
  cur_pt.y = start(1);
  cur_pt.z = start(2);

  if (kdtree.nearestKSearch(cur_pt, 1, pointIdxNKNSearch, pointNKNSquaredDistance) <= 0) {
    ROS_ERROR("[Bug] No nearest object point found.");
    return Vector2d(-1000.0, -1000.0);  // Error indicator
  }

  int nearest_idx = pointIdxNKNSearch[0];
  auto nearest_point = object_cloud->points[nearest_idx];
  return Vector2d(nearest_point.x, nearest_point.y);
}

std::vector<Vector2d> ExplorationManager::findCandidateObjectPoints(
    const Vector3d& start, const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
    int max_candidates)
{
  std::vector<Vector2d> candidates;

  if (object_cloud->empty()) {
    return candidates;
  }

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(object_cloud);

  pcl::PointXYZ cur_pt;
  cur_pt.x = start(0);
  cur_pt.y = start(1);
  cur_pt.z = start(2);

  std::vector<int> indices(max_candidates);
  std::vector<float> distances(max_candidates);

  int found = kdtree.nearestKSearch(cur_pt, max_candidates, indices, distances);

  for (int i = 0; i < found; ++i) {
    auto& pt = object_cloud->points[indices[i]];
    candidates.emplace_back(pt.x, pt.y);
  }

  return candidates;
}

bool ExplorationManager::trySearchObjectPathWithDistance(const Vector2d& start2d,
    const Vector2d& object_pose, double distance, double max_search_time,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path,
    const std::string& debug_msg)
{
  // Maximum acceptable distance from final position to object
  // If the reachable point is too far from the object, it's not worth navigating there
  constexpr double MAX_ACCEPTABLE_OBJECT_DISTANCE = 2.0;  // meters

  path_finder_->reset();
  if (path_finder_->astarSearch(start2d, object_pose, distance, max_search_time) ==
      Astar2D::REACH_END) {
    std::vector<Eigen::Vector2d> path = path_finder_->getPath();
    Vector2d tmp_pos(-1000.0, -1000.0);

    // Find valid position along the path (from end to start)
    for (int i = path.size() - 1; i >= 0; i--) {
      if (sdf_map_->getOccupancy(path[i]) != SDFMap2D::OCCUPIED &&
          sdf_map_->getOccupancy(path[i]) != SDFMap2D::UNKNOWN &&
          sdf_map_->getInflateOccupancy(path[i]) != 1) {
        tmp_pos = path[i];
        break;
      }
    }

    // Check if tmp_pos is valid
    if (tmp_pos.x() < -999.0) {
      ROS_WARN("[ObjectPath] No valid position found along path");
      return false;
    }

    // Check distance from reachable position to object
    double dist_to_object = (tmp_pos - object_pose).norm();
    if (dist_to_object > MAX_ACCEPTABLE_OBJECT_DISTANCE) {
      ROS_WARN("[ObjectPath] Reachable point (%.2f, %.2f) too far from object (%.2f, %.2f): %.2fm > %.2fm",
               tmp_pos.x(), tmp_pos.y(), object_pose.x(), object_pose.y(),
               dist_to_object, MAX_ACCEPTABLE_OBJECT_DISTANCE);
      return false;
    }

    // Search path to the valid position
    path_finder_->reset();
    if (path_finder_->astarSearch(start2d, tmp_pos, 0.2, max_search_time) == Astar2D::REACH_END) {
      refined_path = path_finder_->getPath();
      refined_pos = tmp_pos;
      if (!debug_msg.empty()) {
        ROS_WARN("%s (final dist to object: %.2fm)", debug_msg.c_str(), dist_to_object);
      }
      return true;
    }
  }
  return false;
}

bool ExplorationManager::searchObjectPath(const Vector3d& start,
    const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path,
    int object_id)
{
  const double max_search_time = 0.2;  // Maximum planning time per attempt
  Vector2d start2d = Vector2d(start(0), start(1));

  // Get multiple candidate approach points (up to 10 for better coverage)
  std::vector<Vector2d> candidate_points = findCandidateObjectPoints(start, object_cloud, 10);
  if (candidate_points.empty()) {
    ROS_ERROR("[searchObjectPath] No candidate points found in object cloud.");
    return false;
  }

  // Compute object center from point cloud
  Vector2d object_center(0, 0);
  for (const auto& pt : object_cloud->points) {
    object_center += Vector2d(pt.x, pt.y);
  }
  object_center /= static_cast<double>(object_cloud->points.size());

  // Get first detection position and compute detection direction
  Vector2d detection_pos;
  bool has_detection_dir = false;
  Vector2d detection_dir;
  if (object_id >= 0 && object_map2d_->getFirstDetectionPosition(object_id, detection_pos)) {
    detection_dir = (object_center - detection_pos).normalized();
    has_detection_dir = true;
    ROS_INFO("[searchObjectPath] Object %d: detection_pos=(%.2f,%.2f), center=(%.2f,%.2f), dir=(%.2f,%.2f)",
             object_id, detection_pos.x(), detection_pos.y(),
             object_center.x(), object_center.y(), detection_dir.x(), detection_dir.y());
  }

  // Sort candidates by direction preference if we have detection direction
  if (has_detection_dir) {
    // Score each candidate: prefer points on the detection side of the object
    std::vector<std::pair<double, size_t>> scored_candidates;
    for (size_t i = 0; i < candidate_points.size(); ++i) {
      const Vector2d& cand = candidate_points[i];
      // Direction from candidate to object center
      Vector2d cand_to_obj = (object_center - cand).normalized();
      // Dot product: higher score means candidate is on the same side as detection
      // (i.e., approaching from the detection direction)
      double direction_score = cand_to_obj.dot(detection_dir);
      // Also consider distance to current position as secondary factor
      double dist_to_robot = (cand - start2d).norm();
      // Combined score: direction preference (weight 0.7) + distance penalty (weight 0.3)
      double score = 0.7 * direction_score - 0.3 * (dist_to_robot / 10.0);
      scored_candidates.emplace_back(score, i);
    }
    // Sort by score (highest first)
    std::sort(scored_candidates.begin(), scored_candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Reorder candidate_points by score
    std::vector<Vector2d> sorted_candidates;
    for (const auto& sc : scored_candidates) {
      sorted_candidates.push_back(candidate_points[sc.second]);
    }
    candidate_points = std::move(sorted_candidates);
  }

  // Try different safety distances in order of preference
  const std::vector<double> distances = { 0.5, 0.70, 0.85, 1.0, 1.2 };

  // Try each candidate point (now sorted by detection direction preference)
  for (size_t c = 0; c < candidate_points.size(); ++c) {
    const Vector2d& object_pose = candidate_points[c];

    // Check if this approach point has failed before (approach detection check)
    if (object_id >= 0 && object_map2d_->isApproachPointFailed(object_id, object_pose, 0.5)) {
      ROS_INFO("[searchObjectPath] Skipping failed approach point (%.2f, %.2f) for object %d",
               object_pose.x(), object_pose.y(), object_id);
      continue;
    }

    // Attempt path planning with each safety distance
    for (size_t i = 0; i < distances.size(); ++i) {
      std::string debug_msg = "";
      if (c == 0) {
        // Only print for the first (best direction) candidate
        char buf[100];
        snprintf(buf, sizeof(buf), "I'm going to the object! dist = %.2fm!", distances[i]);
        debug_msg = buf;
      }

      if (trySearchObjectPathWithDistance(start2d, object_pose, distances[i], max_search_time,
              refined_pos, refined_path, debug_msg)) {
        if (c > 0) {
          ROS_INFO("[searchObjectPath] Found path using candidate %zu (detection-direction prioritized)", c);
        }
        return true;
      }
    }
  }

  ROS_ERROR("Failed to find object path to any candidate point.");
  return false;
}

double ExplorationManager::computeAdaptiveAlpha(const vector<Vector2d>& frontiers)
{
  // Adaptive Alpha (Paper Equation):
  // alpha(t) = alpha_base + (1 - alpha_base) * max_fi(V_HSVM(fi)) / tau_conf
  //
  // When strong semantic cues are detected at any frontier, alpha increases
  // to favor exploitation; otherwise the agent prioritizes exploration.

  if (!ep_->use_adaptive_alpha_) {
    return ep_->fusion_alpha_;  // Use fixed alpha if adaptive is disabled
  }

  // Find maximum HSVM value across all frontiers
  double max_hsvm = 0.0;
  for (const auto& frontier : frontiers) {
    Vector2i idx;
    sdf_map_->posToIndex(frontier, idx);

    // Get semantic value at frontier position
    double sem_value = sdf_map_->value_map_->getValue(idx);

    // Also check 5x5 neighborhood for maximum
    auto nbrs = allNeighbors(idx, 2);
    for (const auto& nbr : nbrs) {
      if (sdf_map_->getInflateOccupancy(nbr) == 1 ||
          sdf_map_->getOccupancy(nbr) == SDFMap2D::OCCUPIED)
        continue;
      sem_value = std::max(sem_value, sdf_map_->value_map_->getValue(nbr));
    }

    max_hsvm = std::max(max_hsvm, sem_value);
  }

  // Compute adaptive alpha with clamping to [alpha_base, 1.0]
  double alpha = ep_->alpha_base_ + (1.0 - ep_->alpha_base_) * (max_hsvm / ep_->tau_conf_);
  alpha = std::min(alpha, 1.0);  // Clamp to maximum of 1.0

  ROS_DEBUG("[AdaptiveAlpha] max_hsvm=%.3f, alpha=%.3f (base=%.2f, tau=%.2f)",
      max_hsvm, alpha, ep_->alpha_base_, ep_->tau_conf_);

  return alpha;
}

void ExplorationManager::getSortedSemanticFrontiers(const Vector2d& cur_pos,
    const vector<Vector2d>& frontiers, vector<SemanticFrontier>& sem_frontiers)
{
  // Filter and sort frontiers based on total values (semantic + IG) and reachability
  sem_frontiers.clear();

  // Dual-Value Fusion with Adaptive Alpha (Paper Section III-E)
  // alpha(t) = alpha_base + (1 - alpha_base) * max_fi(V_HSVM(fi)) / tau_conf
  double alpha = computeAdaptiveAlpha(frontiers);
  bool use_ig = ep_->use_ig_fusion_ &&
                sdf_map_->use_ig_value_map_ &&
                sdf_map_->ig_value_map_;

  for (auto& frontier : frontiers) {
    SemanticFrontier sem_frontier;
    sem_frontier.position = frontier;

    // Compute semantic value from local neighborhood
    Vector2i idx;
    sdf_map_->posToIndex(frontier, idx);
    auto nbrs = allNeighbors(idx, 2);  // 5x5 grid neighborhood
    double sem_value = sdf_map_->value_map_->getValue(idx);

    // Find maximum semantic value in neighborhood (ignoring occupied cells)
    for (auto& nbr : nbrs) {
      if (sdf_map_->getInflateOccupancy(idx) == 1 ||
          sdf_map_->getOccupancy(idx) == SDFMap2D::OCCUPIED)
        continue;
      sem_value = std::max(sem_value, sdf_map_->value_map_->getValue(nbr));
    }

    // Compute IG value from local neighborhood (if enabled)
    double ig_value = 0.0;
    if (use_ig) {
      ig_value = sdf_map_->ig_value_map_->getIGValue(idx);
      for (auto& nbr : nbrs) {
        if (sdf_map_->getInflateOccupancy(idx) == 1 ||
            sdf_map_->getOccupancy(idx) == SDFMap2D::OCCUPIED)
          continue;
        ig_value = std::max(ig_value, sdf_map_->ig_value_map_->getIGValue(nbr));
      }
    }

    // Compute total value using dual-value fusion (HSVM/4 to reduce intensity)
    double total_value = use_ig ? (alpha * (sem_value / 4.0) + (1.0 - alpha) * ig_value) : (sem_value / 4.0);
    sem_frontier.semantic_value = total_value;  // Store total value

    // Validate reachability and compute path cost
    Vector2d tmp_pos;
    vector<Vector2d> tmp_path;
    if (!searchFrontierPath(cur_pos, frontier, tmp_pos, tmp_path)) {
      // Assign high cost penalty for unreachable frontiers
      sem_frontier.path_length = 1000000;
      sem_frontier.path.clear();
    }
    else {
      sem_frontier.path_length = Astar2D::pathLength(tmp_path);
      sem_frontier.path = tmp_path;
    }

    // Only include frontiers with valid paths
    if (!sem_frontier.path.empty())
      sem_frontiers.push_back(sem_frontier);
  }

  // Sort by semantic value (desc) then by path length (asc)
  std::sort(sem_frontiers.begin(), sem_frontiers.end());
}

void ExplorationManager::calcSemanticFrontierInfo(const vector<SemanticFrontier>& sem_frontiers,
    double& std_dev, double& max_to_mean, double& mean, bool if_print)
{
  // Handle empty frontier list
  if (sem_frontiers.empty()) {
    std::cout << "No semantic frontiers available." << std::endl;
    max_to_mean = 1.0;  // Neutral ratio
    std_dev = 0.0;      // No variation
    return;
  }

  // Compute mean and maximum semantic values
  double sum = 0.0;
  double max_value = 0.0;
  for (const auto& frontier : sem_frontiers) {
    sum += frontier.semantic_value;
    max_value = max(max_value, frontier.semantic_value);
  }
  mean = sum / sem_frontiers.size();

  // Compute standard deviation
  double variance_sum = 0.0;
  for (const auto& frontier : sem_frontiers)
    variance_sum += (frontier.semantic_value - mean) * (frontier.semantic_value - mean);

  max_to_mean = max_value / mean;
  std_dev = std::sqrt(variance_sum / sem_frontiers.size());

  // Print summary statistics
  std::cout << "Mean Value: " << std::fixed << std::setprecision(3) << mean;
  std::cout << " , Standard Deviation: " << std::fixed << std::setprecision(3) << std_dev;
  std::cout << " , Max-to-Mean: " << std::fixed << std::setprecision(3) << max_to_mean << std::endl;

  // Print detailed frontier values if requested
  if (if_print) {
    for (const auto& sem_frontier : sem_frontiers)
      std::cout << "Value: " << std::fixed << std::setprecision(3) << sem_frontier.semantic_value
                << std::endl;
  }
}

// ==================== WTRP (Weighted Traveling Repairman Problem) ====================

void ExplorationManager::computeSoftmaxWeights(const vector<SemanticFrontier>& sem_frontiers,
    double temperature, vector<double>& weights)
{
  weights.clear();
  if (sem_frontiers.empty()) return;

  // Compute softmax: w_i = exp((s_i - s_max) / tau) / sum(exp((s_k - s_max) / tau))
  // Subtract max for numerical stability
  double max_val = sem_frontiers[0].semantic_value;
  for (const auto& sf : sem_frontiers)
    max_val = std::max(max_val, sf.semantic_value);

  double sum_exp = 0.0;
  vector<double> exp_vals;
  for (const auto& sf : sem_frontiers) {
    double exp_v = std::exp((sf.semantic_value - max_val) / temperature);
    exp_vals.push_back(exp_v);
    sum_exp += exp_v;
  }

  for (double ev : exp_vals)
    weights.push_back(ev / sum_exp);
}

void ExplorationManager::solveWTRP(const Eigen::MatrixXd& cost_matrix,
    const vector<double>& weights, vector<int>& best_order, double& best_cost)
{
  // cost_matrix: (N+1) x (N+1), row/col 0 = current position, 1..N = frontiers
  // weights: size N, weights[i] corresponds to frontier i (cost_matrix row/col i+1)
  // Output: best_order contains frontier indices (0-based into the frontiers vector)

  int N = weights.size();
  best_order.clear();
  best_cost = std::numeric_limits<double>::max();

  if (N == 0) return;

  if (N == 1) {
    best_order.push_back(0);
    best_cost = weights[0] * cost_matrix(0, 1);
    return;
  }

  // Build permutation indices [0, 1, ..., N-1]
  vector<int> perm(N);
  std::iota(perm.begin(), perm.end(), 0);

  if (N <= ep_->wtrp_max_brute_force_) {
    // Brute-force: enumerate all permutations for exact solution
    do {
      double total_cost = 0.0;
      double cumulative_time = 0.0;

      // First leg: start (index 0) -> first frontier in permutation
      cumulative_time += cost_matrix(0, perm[0] + 1);
      total_cost += weights[perm[0]] * cumulative_time;

      // Subsequent legs
      for (int k = 1; k < N; ++k) {
        cumulative_time += cost_matrix(perm[k - 1] + 1, perm[k] + 1);
        total_cost += weights[perm[k]] * cumulative_time;
      }

      if (total_cost < best_cost) {
        best_cost = total_cost;
        best_order = perm;
      }
    } while (std::next_permutation(perm.begin(), perm.end()));
  }
  else {
    // Greedy heuristic for large N:
    // At each step, choose the unvisited frontier that minimizes marginal WTRP cost.
    // Marginal cost of choosing j next = w_j * d(cur, j) + d(cur, j) * sum(remaining weights)
    // This accounts for j's own weighted arrival AND the delay imposed on all subsequent nodes.
    vector<bool> visited(N, false);
    int current_node = 0;  // cost_matrix index (0 = start)
    double cumulative_time = 0.0;
    double total_cost = 0.0;

    for (int step = 0; step < N; ++step) {
      int best_next = -1;
      double best_marginal = std::numeric_limits<double>::max();

      // Compute remaining weight sum (excluding candidates being evaluated)
      double total_remaining_weight = 0.0;
      for (int r = 0; r < N; ++r) {
        if (!visited[r]) total_remaining_weight += weights[r];
      }

      for (int j = 0; j < N; ++j) {
        if (visited[j]) continue;

        double travel = cost_matrix(current_node, j + 1);
        // Marginal cost: choosing j delays all remaining nodes (including j itself)
        // = travel * total_remaining_weight
        // This is equivalent to w_j * travel + travel * (total_remaining - w_j)
        double marginal = travel * total_remaining_weight;

        if (marginal < best_marginal) {
          best_marginal = marginal;
          best_next = j;
        }
      }

      visited[best_next] = true;
      best_order.push_back(best_next);
      cumulative_time += cost_matrix(current_node, best_next + 1);
      total_cost += weights[best_next] * cumulative_time;
      current_node = best_next + 1;
    }

    best_cost = total_cost;
  }
}

void ExplorationManager::wtrpExplorePolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  // Step 1: Get frontiers sorted by semantic value (descending)
  vector<SemanticFrontier> sem_frontiers;
  getSortedSemanticFrontiers(cur_pos, frontiers, sem_frontiers);
  if (sem_frontiers.empty())
    return;

  // Step 2: Select top-k frontiers (same as HYBRID mode)
  int k = ep_->top_k_value_;
  int n = std::min(k, (int)sem_frontiers.size());
  vector<SemanticFrontier> top_k_sem;
  vector<Vector2d> top_k_frontiers;
  for (int i = 0; i < n; i++) {
    top_k_sem.push_back(sem_frontiers[i]);
    top_k_frontiers.push_back(sem_frontiers[i].position);
  }

  // Step 3: Filter to reachable frontiers only
  vector<Vector2d> reachable_frontiers;
  vector<SemanticFrontier> reachable_sem;
  for (int i = 0; i < n; i++) {
    Vector2d tmp_pos;
    vector<Vector2d> tmp_path;
    if (searchFrontierPath(cur_pos, top_k_frontiers[i], tmp_pos, tmp_path)) {
      reachable_frontiers.push_back(top_k_frontiers[i]);
      reachable_sem.push_back(top_k_sem[i]);
    }
  }

  if (reachable_frontiers.empty()) {
    ROS_WARN("[WTRP] No reachable frontiers in top-%d", n);
    return;
  }

  if (reachable_frontiers.size() == 1) {
    searchFrontierPath(cur_pos, reachable_frontiers[0], next_best_pos, next_best_path);
    ed_->tsp_tour_.push_back(cur_pos);
    ed_->tsp_tour_.push_back(reachable_frontiers[0]);
    ROS_WARN("[WTRP] Single reachable frontier, going directly");
    return;
  }

  // Step 4: Compute softmax weights from semantic values
  vector<double> weights;
  computeSoftmaxWeights(reachable_sem, ep_->wtrp_temperature_, weights);

  // Step 5: Compute cost matrix (reuse existing ATSP cost matrix function)
  Eigen::MatrixXd cost_matrix;
  computeATSPCostMatrix(cur_pos, reachable_frontiers, cost_matrix);

  // Step 6: Solve WTRP for optimal visitation order
  auto t1 = ros::Time::now();
  vector<int> best_order;
  double best_cost;
  solveWTRP(cost_matrix, weights, best_order, best_cost);
  double wtrp_time = (ros::Time::now() - t1).toSec();

  // Step 7: Log results
  ROS_WARN("[WTRP] Solved with %d frontiers (tau=%.2f), cost=%.4f, time=%.4fs",
      (int)reachable_frontiers.size(), ep_->wtrp_temperature_, best_cost, wtrp_time);
  for (int i = 0; i < (int)best_order.size(); i++) {
    int idx = best_order[i];
    ROS_INFO("[WTRP] Order[%d]: frontier %d, value=%.3f, weight=%.3f",
        i, idx, reachable_sem[idx].semantic_value, weights[idx]);
  }

  // Step 8: Store tour for visualization
  ed_->tsp_tour_.push_back(cur_pos);
  for (auto idx : best_order) ed_->tsp_tour_.push_back(reachable_frontiers[idx]);

  // Step 9: Apply hysteresis to prevent frontier oscillation
  // If we have a previous target that is still valid and reachable, keep it unless
  // the new best target has significantly better cost (controlled by wtrp_hysteresis_ratio_)
  int selected_idx = best_order[0];  // Default: select the best frontier from WTRP
  double selected_cost = 0.0;

  if (has_last_wtrp_target_) {
    // Check if the last target is still in the reachable frontiers
    int last_target_idx = -1;
    for (int i = 0; i < (int)reachable_frontiers.size(); i++) {
      if ((reachable_frontiers[i] - last_wtrp_target_).norm() < 0.5) {
        last_target_idx = i;
        break;
      }
    }

    if (last_target_idx >= 0) {
      // Find the position of last target in the WTRP order
      int last_target_order = -1;
      int new_best_order = 0;
      for (int i = 0; i < (int)best_order.size(); i++) {
        if (best_order[i] == last_target_idx) {
          last_target_order = i;
          break;
        }
      }

      // Calculate weighted costs for comparison
      // Cost = position_in_order * average_weight (simplified comparison)
      double new_best_cost = weights[best_order[0]];
      double last_target_cost = (last_target_order >= 0) ? weights[last_target_idx] : 0.0;

      // Apply hysteresis: only switch if new target is significantly better
      // new_cost * hysteresis_ratio > old_cost means new target needs to be much better
      if (last_target_cost > 0.0 &&
          new_best_cost < last_target_cost / ep_->wtrp_hysteresis_ratio_) {
        // Keep the old target (hysteresis prevents switching)
        selected_idx = last_target_idx;
        ROS_WARN("[WTRP Hysteresis] Keeping previous target (idx=%d, weight=%.3f) vs new best (idx=%d, weight=%.3f), ratio=%.2f",
            last_target_idx, last_target_cost, best_order[0], new_best_cost, ep_->wtrp_hysteresis_ratio_);
      } else {
        ROS_INFO("[WTRP Hysteresis] Switching to new target (idx=%d, weight=%.3f), old (idx=%d, weight=%.3f)",
            best_order[0], new_best_cost, last_target_idx, last_target_cost);
      }
    } else {
      ROS_INFO("[WTRP Hysteresis] Previous target no longer reachable, selecting new target");
    }
  }

  // Navigate to selected frontier
  Vector2d target = reachable_frontiers[selected_idx];
  if (searchFrontierPath(cur_pos, target, next_best_pos, next_best_path)) {
    // Update WTRP target tracking state
    has_last_wtrp_target_ = true;
    last_wtrp_target_ = target;
    last_wtrp_target_value_ = reachable_sem[selected_idx].semantic_value;
  } else {
    // If selected target is not reachable, fall back to first reachable in WTRP order
    for (auto idx : best_order) {
      target = reachable_frontiers[idx];
      if (searchFrontierPath(cur_pos, target, next_best_pos, next_best_path)) {
        has_last_wtrp_target_ = true;
        last_wtrp_target_ = target;
        last_wtrp_target_value_ = reachable_sem[idx].semantic_value;
        break;
      }
    }
  }
}

}  // namespace apexnav_planner
