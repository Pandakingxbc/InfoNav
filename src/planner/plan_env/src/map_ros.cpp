/**
 * @file map_ros.cpp
 * @brief Implementation of ROS interface for 2D SDF mapping system
 *
 * This file implements the MapROS class which provides the ROS interface for
 * the 2D signed distance field mapping system. It handles sensor data processing,
 * object detection integration, and real-time map visualization.
 *
 * @author Zager-Zhang
 */

#include <plan_env/map_ros.h>
#include <plan_env/multi_source_value_map.h>
#include <plan_env/ig_value_map.h>

namespace apexnav_planner {

void MapROS::setMap(SDFMap2D* map)
{
  this->map_ = map;
}

void MapROS::init()
{
  // Load camera intrinsic parameters from ROS parameter server
  node_.param("map_ros/fx", fx_, -1.0);
  node_.param("map_ros/fy", fy_, -1.0);
  node_.param("map_ros/cx", cx_, -1.0);
  node_.param("map_ros/cy", cy_, -1.0);

  // Load depth filtering parameters
  node_.param("map_ros/depth_filter_maxdist", depth_filter_maxdist_, -1.0);
  node_.param("map_ros/depth_filter_mindist", depth_filter_mindist_, -1.0);
  node_.param("map_ros/depth_filter_margin", depth_filter_margin_, -1);
  node_.param("map_ros/filter_min_height", filter_min_height_, 0.5);
  node_.param("map_ros/filter_max_height", filter_max_height_, 0.88);
  node_.param("map_ros/k_depth_scaling_factor", k_depth_scaling_factor_, -1.0);
  node_.param("map_ros/skip_pixel", skip_pixel_, -1);
  node_.param("map_ros/frame_id", frame_id_, string("world"));
  node_.param("map_ros/virtual_ground_height", virtual_ground_height_, -0.28);

  // Handle Habitat simulator vs real-world configuration
  bool is_real_world;
  node_.param("fsm/is_real_world", is_real_world, false);

  if (!is_real_world) {
    // Override depth parameters with Habitat simulator settings
    double habitat_max_depth, habitat_min_depth;
    node_.param("/habitat/simulator/agents/main_agent/sim_sensors/depth_sensor/max_depth",
        habitat_max_depth, -1.0);
    node_.param("/habitat/simulator/agents/main_agent/sim_sensors/depth_sensor/min_depth",
        habitat_min_depth, -1.0);
    if (habitat_max_depth != -1.0 && habitat_min_depth != -1.0) {
      depth_filter_maxdist_ = habitat_max_depth;
      depth_filter_mindist_ = habitat_min_depth;
      ROS_WARN("Using habitat simulator params, set depth_filter_range = [%.2f, %.2f] m",
          habitat_min_depth, habitat_max_depth);
    }
  }

  // Initialize point cloud data structures
  depth_cloud_.reset(new PointCloud3D());
  filtered_depth_cloud2d_.reset(new PointCloud2D());

  // Pre-allocate point cloud vectors for efficiency
  proj_points_.resize(640 * 480 / (skip_pixel_ * skip_pixel_));
  depth_cloud_->points.resize(640 * 480 / (skip_pixel_ * skip_pixel_));
  proj_points_cnt_ = 0;
  depth_image_.reset(new cv::Mat);

  // Initialize state flags
  local_updated_ = false;
  esdf_need_update_ = false;

  // Setup periodic timers for map updates and visualization
  esdf_timer_ = node_.createTimer(ros::Duration(0.1), &MapROS::updateESDFCallback, this);
  vis_timer_ = node_.createTimer(ros::Duration(1.0), &MapROS::visCallback, this);  // Reduced frequency to 1Hz to prevent RViz flickering

  // Setup publishers for map visualization
  occupied_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/occupied", 10);
  unknown_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/unknown", 10);
  free_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/free", 10);
  occupied_inflate_pub_ =
      node_.advertise<sensor_msgs::PointCloud2>("/grid_map/occupied_inflate", 10);

  object_grid_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/occupancy_object", 10);
  esdf_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/esdf", 10);
  update_range_pub_ = node_.advertise<visualization_msgs::Marker>("/grid_map/update_range", 10);
  depth_cloud_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/depth_cloud", 10);
  filtered_depth_cloud_pub_ =
      node_.advertise<sensor_msgs::PointCloud2>("/grid_map/filtered_depth_cloud", 10);
  filtered_object_cloud_pub_ =
      node_.advertise<sensor_msgs::PointCloud2>("/grid_map/filtered_object_cloud", 10);
  all_object_cloud_pub_ =
      node_.advertise<sensor_msgs::PointCloud2>("/grid_map/all_object_cloud", 10);
  over_depth_object_cloud_pub_ =
      node_.advertise<sensor_msgs::PointCloud2>("/grid_map/over_depth_object_cloud", 10);
  value_map_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/value_map", 10);
  ig_map_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/ig_map", 10);
  hsvm_map_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/hsvm_map", 10);
  confidence_map_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/confidence_map", 10);

  // VLM verification support - publisher for object update notifications
  object_update_info_pub_ = node_.advertise<plan_env::ObjectUpdateInfo>("/vlm/object_update_info", 10);

  // Initialize VLM tracking state
  current_nav_target_object_id_ = -1;

  // Setup subscriber for object detection
  detected_object_cloud_sub_ = node_.subscribe(
      "/detector/clouds_with_scores", 10, &MapROS::detectedObjectCloudCallback, this);

  // Setup subscribers for multi-source semantic value map
  semantic_prompts_sub_ = node_.subscribe(
      "/semantic_prompts", 1, &MapROS::semanticPromptsCallback, this);
  semantic_scores_sub_ = node_.subscribe(
      "/semantic_scores", 10, &MapROS::semanticScoresCallback, this);

  // Setup subscriber for episode reset signals
  episode_reset_sub_ = node_.subscribe(
      "/habitat/episode_reset", 10, &MapROS::episodeResetCallback, this);

  // Initialize multi-source semantic mode
  node_.param("value_map/use_multi_source", use_multi_source_semantic_, false);
  if (use_multi_source_semantic_) {
    ROS_INFO("[MapROS] Multi-source semantic value map enabled");
  }

  // Load Dual-Value Fusion parameter (same as exploration_manager)
  node_.param("exploration/fusion_alpha", fusion_alpha_, 0.5);
  ROS_INFO("[MapROS] Dual-Value Fusion alpha=%.2f", fusion_alpha_);

  // Setup synchronized subscribers for depth image and pose data
  depth_sub_.reset(
      new message_filters::Subscriber<sensor_msgs::Image>(node_, "/map_ros/depth", 20));
  pose_sub_.reset(new message_filters::Subscriber<nav_msgs::Odometry>(node_, "/map_ros/pose", 20));

  sync_image_pose_.reset(new message_filters::Synchronizer<MapROS::SyncPolicyImagePose>(
      MapROS::SyncPolicyImagePose(20), *depth_sub_, *pose_sub_));
  sync_image_pose_->setMaxIntervalDuration(ros::Duration(0.01));  // Set maximum temporal offset
  sync_image_pose_->registerCallback(boost::bind(&MapROS::depthPoseCallback, this, _1, _2));

  // Initialize object tracking variables
  continue_over_depth_count_ = -1;
  map_start_time_ = ros::Time::now();
}

void MapROS::visCallback(const ros::TimerEvent& e)
{
  vis_timer_.stop();

  // Publish all visualization topics
  publishOccupied();
  publishInfOccupied();
  publishObjectMap();
  publishUnknown();
  publishFree();
  publishValueMap();
  publishIGMap();
  publishHSVMMap();
  // publishConfidenceMap();
  publishESDFMap();
  // publishUpdateRange();

  vis_timer_.start();
}

void MapROS::semanticPromptsCallback(const plan_env::SemanticPromptsConstPtr& msg)
{
  if (!use_multi_source_semantic_ || !map_->multi_source_value_map_) {
    ROS_WARN_THROTTLE(10.0, "[MapROS] Multi-source semantic mode is disabled");
    return;
  }

  // Store current semantic prompts and target
  current_target_object_ = msg->target_object;
  current_semantic_prompts_.clear();
  current_semantic_scores_.clear();
  current_semantic_scores_.resize(msg->prompts.size(), -1.0);  // Initialize with -1 (not yet received)

  // Initialize multi-source value map with new semantic sources
  map_->multi_source_value_map_->clearSemanticSources();

  for (size_t i = 0; i < msg->prompts.size(); ++i) {
    current_semantic_prompts_.push_back(msg->prompts[i]);
    double weight = (i < msg->weights.size()) ? msg->weights[i] : 1.0;
    map_->multi_source_value_map_->addSemanticSource(msg->prompts[i], weight);
  }

  ROS_INFO("[MapROS] Received %zu semantic prompts for target '%s'",
           msg->prompts.size(), current_target_object_.c_str());

  // Log prompts for debugging
  for (size_t i = 0; i < current_semantic_prompts_.size(); ++i) {
    ROS_DEBUG("  [%zu] %s (weight: %.3f)",
              i, current_semantic_prompts_[i].c_str(),
              map_->multi_source_value_map_->getSemanticSource(i).weight);
  }
}

void MapROS::semanticScoresCallback(const plan_env::SemanticScoresConstPtr& msg)
{
  if (!use_multi_source_semantic_ || !map_->multi_source_value_map_) {
    return;
  }

  // Validate that scores match the number of current prompts
  if (msg->scores.size() != current_semantic_prompts_.size()) {
    ROS_WARN("[MapROS] Received %zu scores but have %zu prompts, ignoring",
             msg->scores.size(), current_semantic_prompts_.size());
    return;
  }

  // Update semantic scores
  current_semantic_scores_ = msg->scores;

  ROS_DEBUG_THROTTLE(2.0, "[MapROS] Updated semantic scores for %zu sources",
                    msg->scores.size());
}

void MapROS::detectedObjectCloudCallback(const plan_env::MultipleMasksWithConfidenceConstPtr& msg)
{
  // Validate message structure consistency
  if (!(msg->confidence_scores.size() == msg->point_clouds.size() &&
          msg->confidence_scores.size() == msg->label_indices.size())) {
    ROS_ERROR("[Bug] The MultipleMasksWithConfidence msg is wrong!!!");
    return;
  }

  auto t1 = ros::Time::now();

  // Check camera orientation - only process when looking down (for better object detection)
  Eigen::Vector3d euler =
      camera_q_.toRotationMatrix().eulerAngles(2, 1, 0);  // ZYX order: yaw, roll, pitch
  if (euler[2] < 0)
    euler[2] += M_PI;
  double camera_pitch = euler[2];
  if (camera_pitch < 1.5)  // Skip if camera not tilted down enough
    return;

  // Backup previous over-depth object cloud for consistency tracking
  auto last_over_depth_cloud =
      std::make_shared<PointCloud3D>(*map_->object_map2d_->over_depth_object_cloud_);
  map_->object_map2d_->over_depth_object_cloud_.reset(new PointCloud3D());

  // Initialize point cloud processing tools and containers
  pcl::VoxelGrid<Point3D> voxel_filter;
  PointCloud3D::Ptr all_object_cloud(new PointCloud3D());
  PointCloud3D::Ptr filtered_all_object_cloud(new PointCloud3D());
  vector<DetectedObject> detected_objects;

  // Process each detected object in the message
  for (int i = 0; i < (int)msg->confidence_scores.size(); i++) {
    auto cloud = msg->point_clouds[i];
    auto confidence_score = msg->confidence_scores[i];
    auto label = msg->label_indices[i];

    // Convert ROS message to PCL point cloud
    PointCloud3D::Ptr single_object_cloud(new PointCloud3D());
    pcl::fromROSMsg(cloud, *single_object_cloud);
    *all_object_cloud += *single_object_cloud;

    // Apply voxel grid downsampling to reduce computational load
    voxel_filter.setInputCloud(single_object_cloud);
    voxel_filter.setLeafSize(0.04f, 0.04f, 0.06f);
    voxel_filter.filter(*single_object_cloud);

    // Filter out points beyond sensor accuracy range (>5m depth is unreliable)
    PointCloud3D::Ptr tmp_object_cloud(new PointCloud3D());
    PointCloud3D::Ptr over_depth_object_cloud(new PointCloud3D());
    for (auto object_pt : single_object_cloud->points) {
      Eigen::Vector3d object_pt3d = Eigen::Vector3d(object_pt.x, object_pt.y, object_pt.z);
      if ((object_pt3d - camera_pos_).norm() > depth_filter_maxdist_ - 0.10) {
        // Store over-depth points for target objects (label == 0) for tracking consistency
        if (label == 0)
          over_depth_object_cloud->points.push_back(object_pt);
        continue;
      }
      tmp_object_cloud->points.push_back(object_pt);
    }
    single_object_cloud = tmp_object_cloud;

    // Skip objects that are entirely beyond valid depth range
    if (single_object_cloud->points.empty()) {
      if (!over_depth_object_cloud->points.empty()) {
        ROS_ERROR("Have all over depth object cloud!!!!");
        *map_->object_map2d_->over_depth_object_cloud_ += *over_depth_object_cloud;
      }
      continue;
    }

    // Apply DBSCAN clustering to remove noise and outliers
    single_object_cloud = dbscan(single_object_cloud, 0.12f, 10);
    if (single_object_cloud == nullptr) {
      ROS_ERROR("After DBSCAN, no point cloud cluster!!");
      continue;
    }

    if (single_object_cloud->points.empty()) {
      ROS_ERROR("Single object point cloud is empty!!!");
      continue;
    }

    // Accumulate filtered object data
    *filtered_all_object_cloud += *single_object_cloud;
    DetectedObject detected_object;
    detected_object.cloud = single_object_cloud;
    detected_object.score = confidence_score;
    detected_object.label = label;
    detected_objects.push_back(detected_object);
  }

  // Maintain consistency in over-depth object tracking
  if (continue_over_depth_count_ == -1 &&
      !map_->object_map2d_->over_depth_object_cloud_->points.empty())
    continue_over_depth_count_ = 0;
  else if (continue_over_depth_count_ <= 4 && continue_over_depth_count_ >= 0) {
    continue_over_depth_count_++;
    *map_->object_map2d_->over_depth_object_cloud_ = *last_over_depth_cloud;
  }
  else {
    continue_over_depth_count_ = -1;
  }

  // Publish visualization point clouds for debugging and monitoring
  publishPointCloud(filtered_object_cloud_pub_, filtered_all_object_cloud);
  publishPointCloud(all_object_cloud_pub_, all_object_cloud);
  publishPointCloud(over_depth_object_cloud_pub_, map_->object_map2d_->over_depth_object_cloud_);

  // Update object map with processed detection results
  *map_->object_map2d_->all_object_clouds_ = *filtered_all_object_cloud;

  // Set current robot position before object detection (for recording detection direction)
  Eigen::Vector2d robot_pos_2d(camera_pos_(0), camera_pos_(1));
  map_->object_map2d_->setCurrentRobotPosition(robot_pos_2d);

  vector<int> detected_object_cluster_ids;
  map_->inputObjectCloud2D(detected_objects, detected_object_cluster_ids);

  // Publish ObjectUpdateInfo for VLM verification for ALL detected objects
  // This allows Python VLM verifier to collect historical frames for any object
  for (size_t i = 0; i < detected_object_cluster_ids.size(); ++i) {
    int object_id = detected_object_cluster_ids[i];

    plan_env::ObjectUpdateInfo update_msg;
    update_msg.header.stamp = ros::Time::now();
    update_msg.header.frame_id = frame_id_;
    update_msg.object_id = object_id;
    update_msg.detection_confidence = detected_objects[i].score;
    update_msg.is_current_target = (object_id == current_nav_target_object_id_);

    // Calculate distance to object
    Eigen::Vector2d object_center;
    if (map_->object_map2d_->getObjectCenter(object_id, object_center)) {
      Eigen::Vector2d robot_pos_2d(camera_pos_(0), camera_pos_(1));
      update_msg.distance_to_object = (object_center - robot_pos_2d).norm();
    } else {
      update_msg.distance_to_object = -1.0;  // Unknown distance
    }

    object_update_info_pub_.publish(update_msg);
    ROS_DEBUG("[MapROS] Published ObjectUpdateInfo for object %d (is_target=%d), confidence=%.3f, dist=%.2f",
              object_id, update_msg.is_current_target, update_msg.detection_confidence, update_msg.distance_to_object);
  }

  // Optional: Log detected object IDs for debugging
  // for (auto object_id : detected_object_cluster_ids)
  //   ROS_INFO("Detected object id is %d", object_id);

  // Extract observation data from depth sensor for objects not detected by vision
  getObservationObjectsCloud(detected_object_cluster_ids);

  double object_map_process_time = (ros::Time::now() - t1).toSec();
  ROS_INFO_THROTTLE(
      10.0, "[Calculating Time] Object Map process time = %.3f s", object_map_process_time);
}

void MapROS::updateESDFCallback(const ros::TimerEvent& /*event*/)
{
  if (!esdf_need_update_)
    return;

  esdf_timer_.stop();

  auto t1 = ros::Time::now();
  map_->updateESDFMap();
  esdf_need_update_ = false;
  double esdf_time = (ros::Time::now() - t1).toSec();
  ROS_INFO_THROTTLE(50.0, "[Calculating Time] ESDF Map process time = %.3f s", esdf_time);

  esdf_timer_.start();
}

void MapROS::depthPoseCallback(
    const sensor_msgs::ImageConstPtr& img, const nav_msgs::OdometryConstPtr& pose)
{
  // Extract camera pose from odometry message
  camera_pos_(0) = pose->pose.pose.position.x;
  camera_pos_(1) = pose->pose.pose.position.y;
  camera_pos_(2) = pose->pose.pose.position.z;
  camera_q_ = Eigen::Quaterniond(pose->pose.pose.orientation.w, pose->pose.pose.orientation.x,
      pose->pose.pose.orientation.y, pose->pose.pose.orientation.z);

  // Calculate camera yaw angle for value map updates
  Eigen::Vector3d euler =
      camera_q_.toRotationMatrix().eulerAngles(2, 1, 0);  // ZYX order: yaw, roll, pitch
  double camera_yaw = euler[0];
  Eigen::Vector2d camera_pos = Eigen::Vector2d(camera_pos_(0), camera_pos_(1));

  // Skip processing if camera is outside map bounds
  if (!map_->isInMap(camera_pos))
    return;

  // Convert depth image format (Habitat publishes Float32, some sensors use 8UC1)
  cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img, img->encoding);
  if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
    (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, k_depth_scaling_factor_);
  if (img->encoding == sensor_msgs::image_encodings::TYPE_8UC1)
    (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, 255.0);
  cv_ptr->image.copyTo(*depth_image_);

  auto t1 = ros::Time::now();

  // Process depth image into 3D point cloud and filter to 2D representation
  processDepthImage();
  filterPointCloudToXY();

  // Update occupancy grid with filtered depth data
  vector<Eigen::Vector2i> free_grids;
  // Dilate free_grids to ensure more complete coverage
  dilateGrids(free_grids, 1);
  map_->inputDepthCloud2D(filtered_depth_cloud2d_, camera_pos_, free_grids);
  double process_time = (ros::Time::now() - t1).toSec();
  ROS_INFO_THROTTLE(50.0, "[Calculating Time] Grid Map process time = %.3f s", process_time);

  t1 = ros::Time::now();
  // Update multi-source semantic value map
  if (use_multi_source_semantic_ && map_->multi_source_value_map_) {
    // Update multi-source semantic value map if scores are available
    bool has_valid_scores = !current_semantic_scores_.empty();
    for (size_t i = 0; i < current_semantic_scores_.size(); ++i) {
      if (current_semantic_scores_[i] >= 0.0) {
        map_->multi_source_value_map_->updateSemanticSource(
            i, camera_pos, camera_yaw, free_grids, current_semantic_scores_[i]);
      }
    }
    if (has_valid_scores) {
      ROS_DEBUG_THROTTLE(5.0, "[MapROS] Updated multi-source value map with %zu sources",
                        current_semantic_scores_.size());
    }
  }

  // Update IG value map sensor state (actual update happens via ROS callback)
  if (map_->use_ig_value_map_ && map_->ig_value_map_) {
    map_->ig_value_map_->updateSensorState(camera_pos, camera_yaw, free_grids);
  }

  double value_map_time = (ros::Time::now() - t1).toSec();
  ROS_INFO_THROTTLE(50.0, "[Calculating Time] Value Map process time = %.3f s", value_map_time);

  // Trigger ESDF update if local map has been updated
  if (local_updated_) {
    map_->clearAndInflateLocalMap();
    esdf_need_update_ = true;
    local_updated_ = false;
  }
}

void MapROS::processDepthImage()
{
  proj_points_cnt_ = 0;

  uint16_t* row_ptr;
  int cols = depth_image_->cols;
  int rows = depth_image_->rows;
  double depth;
  Eigen::Matrix3d camera_r = camera_q_.toRotationMatrix();
  Eigen::Vector3d pt_cur, pt_world;
  const double inv_factor = 1.0 / k_depth_scaling_factor_;

  // Iterate through depth image pixels with margin and skipping for efficiency
  for (int v = depth_filter_margin_; v < rows - depth_filter_margin_; v += skip_pixel_) {
    row_ptr = depth_image_->ptr<uint16_t>(v) + depth_filter_margin_;
    for (int u = depth_filter_margin_; u < cols - depth_filter_margin_; u += skip_pixel_) {
      // Convert pixel depth value to metric distance
      depth = (*row_ptr) * inv_factor * (depth_filter_maxdist_ - depth_filter_mindist_) +
              depth_filter_mindist_;
      row_ptr = row_ptr + skip_pixel_;

      // Apply depth range filtering
      if (depth > depth_filter_maxdist_)
        depth = depth_filter_maxdist_;
      else if (depth < depth_filter_mindist_)
        continue;

      // Project pixel to 3D camera coordinates
      pt_cur(0) = (u - cx_) * depth / fx_;
      pt_cur(1) = (v - cy_) * depth / fy_;
      pt_cur(2) = depth;

      // Transform to world coordinates
      pt_world = camera_r * pt_cur + camera_pos_;
      auto& pt = depth_cloud_->points[proj_points_cnt_++];
      pt.x = pt_world[0];
      pt.y = pt_world[1];
      pt.z = pt_world[2];
    }
  }
  publishPointCloud(depth_cloud_pub_, depth_cloud_);
}

/**
 * @brief Extract undetected objects from depth observation data
 *
 * Identifies objects that appear in depth sensor data but weren't detected by
 * the vision system. Uses bounding box filtering to separate already detected
 * objects from potential undetected ones. Assigns zero confidence to undetected objects.
 *
 * @param filter_object_ids List of already detected object cluster IDs to filter out
 */
void MapROS::getObservationObjectsCloud(const std::vector<int>& filter_object_ids)
{
  // Downsample depth cloud for efficient processing
  PointCloud3D::Ptr filtered_depth_cloud(new PointCloud3D());
  pcl::VoxelGrid<Point3D> voxel_filter;
  voxel_filter.setInputCloud(depth_cloud_);
  voxel_filter.setLeafSize(0.1f, 0.1f, 0.1f);
  voxel_filter.filter(*filtered_depth_cloud);

  // Get object bounding boxes and create filter flags
  vector<Vector3d> bmins, bmaxs;
  map_->object_map2d_->getObjectBoxes(bmins, bmaxs);
  vector<char> filter_object_flag(bmins.size(), 0);
  for (auto filter_object_id : filter_object_ids) filter_object_flag[filter_object_id] = 1;

  // Use CropBox filter to extract points within object bounding boxes
  pcl::CropBox<Point3D> crop_box_filter;
  crop_box_filter.setInputCloud(filtered_depth_cloud);
  vector<pcl::shared_ptr<PointCloud3D>> observation_clouds;

  for (int i = 0; i < (int)bmins.size(); i++) {
    PointCloud3D::Ptr cloud_filtered(new PointCloud3D);
    if (filter_object_flag[i])
      observation_clouds.push_back(cloud_filtered);  // Empty cloud for detected objects
    else {
      // Extract points within bounding box for undetected objects
      double inf = 0.2f;  // Inflation factor for bounding box
      Eigen::Vector4f min_point(bmins[i][0] - inf, bmins[i][1] - inf, bmins[i][2] - inf, 1.0);
      Eigen::Vector4f max_point(bmaxs[i][0] + inf, bmaxs[i][1] + inf, bmaxs[i][2] + inf, 1.0);
      crop_box_filter.setMin(min_point);
      crop_box_filter.setMax(max_point);
      crop_box_filter.filter(*cloud_filtered);
      observation_clouds.push_back(cloud_filtered);
    }
  }

  // DISABLED: Negative sample feedback was causing confidence to drop during navigation,
  // leading to premature target abandonment when object not detected in some frames.
  // map_->object_map2d_->inputObservationObjectsCloud(observation_clouds, 0.0);
}

/**
 * @brief Filter and process 3D point cloud to 2D occupancy grid
 */
void MapROS::filterPointCloudToXY()
{
  // Default ground height assumption (currently set to 0)
  double cur_floor_height = 0.0;
  double virtual_ground = virtual_ground_height_;

  auto t1 = ros::Time::now();
  PointCloud3D::Ptr filtered_cloud_3d(new PointCloud3D());
  PointCloud3D::Ptr down_depth_cloud_3d(new PointCloud3D());
  PointCloud3D::Ptr under_ground_cloud_3d(new PointCloud3D());
  PointCloud2D::Ptr under_ground_cloud_2d(new PointCloud2D());

  // Downsample point cloud for efficient processing
  pcl::VoxelGrid<Point3D> voxel_filter;
  voxel_filter.setInputCloud(depth_cloud_);
  voxel_filter.setLeafSize(0.04f, 0.04f, 0.1f);  // Different resolution for XY vs Z
  voxel_filter.filter(*down_depth_cloud_3d);

  filtered_depth_cloud2d_->clear();

  // Separate points by height categories
  for (int i = 0; i < (int)down_depth_cloud_3d->points.size(); i++) {
    Point3D pt;
    pt.x = down_depth_cloud_3d->points[i].x;
    pt.y = down_depth_cloud_3d->points[i].y;
    pt.z = down_depth_cloud_3d->points[i].z;

    // Points below virtual ground (for virtual ground generation)
    if (down_depth_cloud_3d->points[i].z < cur_floor_height + virtual_ground)
      under_ground_cloud_3d->points.push_back(pt);
    // Points in obstacle height range
    else if (down_depth_cloud_3d->points[i].z > cur_floor_height + filter_min_height_ &&
             down_depth_cloud_3d->points[i].z < cur_floor_height + filter_max_height_)
      filtered_cloud_3d->points.push_back(pt);
  }

  pcl::RadiusOutlierRemoval<Point3D> outrem;

  // Remove outliers from obstacle points (handles noisy depth data from datasets)
  if (!filtered_cloud_3d->points.empty()) {
    outrem.setInputCloud(filtered_cloud_3d);
    outrem.setRadiusSearch(0.3);         // Search radius for neighbors
    outrem.setMinNeighborsInRadius(35);  // Minimum neighbor threshold
    outrem.filter(*filtered_cloud_3d);
  }

  publishPointCloud(filtered_depth_cloud_pub_, filtered_cloud_3d);

  // Project 3D obstacle points to 2D for occupancy mapping
  for (auto pt : filtered_cloud_3d->points) {
    Point2D pt_xy;
    pt_xy.x = pt.x;
    pt_xy.y = pt.y;
    filtered_depth_cloud2d_->points.push_back(pt_xy);
  }

  // Remove outliers from under-ground points (handles noisy depth data)
  if (!under_ground_cloud_3d->points.empty()) {
    outrem.setInputCloud(under_ground_cloud_3d);
    outrem.setRadiusSearch(0.21);        // Smaller search radius for ground points
    outrem.setMinNeighborsInRadius(40);  // Higher neighbor threshold
    outrem.filter(*under_ground_cloud_3d);
  }

  // Add virtual ground points to prevent getting stuck when going downstairs
  Eigen::Vector3d euler =
      camera_q_.toRotationMatrix().eulerAngles(2, 1, 0);  // ZYX order: yaw roll pitch
  if (euler[2] < 0)
    euler[2] += M_PI;
  double camera_pitch = euler[2];

  // When camera is pointing down (pitch > 1.5 rad) and under-ground points exist
  if (camera_pitch > 1.5 && !under_ground_cloud_3d->points.empty()) {
    for (auto pt : under_ground_cloud_3d->points) {
      Eigen::Vector3d pt_pos = Eigen::Vector3d(pt.x, pt.y, pt.z);
      Eigen::Vector2d ground_pos;

      // Interpolate ray from camera to point, finding intersection with virtual ground
      if (interpolateLineAtZ(pt_pos, camera_pos_, cur_floor_height + virtual_ground, ground_pos)) {
        Point2D pt_xy;
        pt_xy.x = ground_pos(0);
        pt_xy.y = ground_pos(1);
        filtered_depth_cloud2d_->points.push_back(pt_xy);
        under_ground_cloud_2d->points.push_back(pt_xy);
      }
    }
    map_->inputVirtualGround(under_ground_cloud_2d);
  }

  double filter_time = (ros::Time::now() - t1).toSec();
  ROS_WARN_COND(filter_time > 0.1, "Filter point cloud time maybe a little long = %.3f ms",
      filter_time * 1000);
}

bool MapROS::interpolateLineAtZ(
    const Eigen::Vector3d& A, const Eigen::Vector3d& B, double target_z, Eigen::Vector2d& P)
{
  // Check if target_z is between A.z and B.z (intersection possible)
  if ((A.z() - target_z) * (B.z() - target_z) > 0)
    return false;  // target_z not within segment bounds

  // Calculate interpolation parameter t (0 = point A, 1 = point B)
  double t = (target_z - A.z()) / (B.z() - A.z());

  // Linear interpolation for X and Y coordinates
  double x = A.x() + t * (B.x() - A.x());
  double y = A.y() + t * (B.y() - A.y());
  P = Eigen::Vector2d(x, y);
  return true;
}

/**
 * @brief DBSCAN clustering algorithm to extract largest point cloud cluster
 *
 * Applies Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
 * to identify and return the largest cluster from a point cloud. This is used
 * to filter noise and extract the main object point cloud, assuming each object
 * consists of a single dominant cluster.
 *
 * @param cloud Input point cloud to cluster
 * @param eps Maximum distance between points in the same cluster (neighborhood radius)
 * @param minPts Minimum number of points required to form a dense region (cluster)
 * @return Pointer to largest cluster point cloud, or nullptr if clustering fails
 */
PointCloud3D::Ptr MapROS::dbscan(const PointCloud3D::Ptr& cloud, double eps, int minPts)
{
  if (cloud->empty()) {
    ROS_ERROR("[DBSCAN] Input cloud is empty!");
    return nullptr;
  }

  // Build KD-tree for efficient neighbor search
  pcl::search::KdTree<Point3D>::Ptr tree(new pcl::search::KdTree<Point3D>);
  tree->setInputCloud(cloud);
  std::vector<pcl::PointIndices> cluster_indices;

  // Use PCL's EuclideanClusterExtraction to implement DBSCAN-like clustering
  pcl::EuclideanClusterExtraction<Point3D> ec;
  ec.setClusterTolerance(eps);                 // Neighborhood radius
  ec.setMinClusterSize(minPts);                // Minimum points per cluster
  ec.setMaxClusterSize(cloud->points.size());  // Maximum cluster size (full cloud)
  ec.setSearchMethod(tree);                    // Set KD-Tree for neighbor search
  ec.setInputCloud(cloud);                     // Input point cloud
  ec.extract(cluster_indices);                 // Extract clustering results

  // Return null if no clusters found
  if (cluster_indices.empty()) {
    ROS_WARN("[DBSCAN] No clusters found!");
    return nullptr;
  }

  // Find the largest cluster by counting points
  int largest_cluster_index = -1;
  size_t max_size = 0;
  for (size_t i = 0; i < cluster_indices.size(); ++i) {
    if (cluster_indices[i].indices.size() > max_size) {
      max_size = cluster_indices[i].indices.size();
      largest_cluster_index = i;
    }
  }

  // Create new point cloud containing only the largest cluster
  PointCloud3D::Ptr largest_cluster(new PointCloud3D);
  for (int idx : cluster_indices[largest_cluster_index].indices)
    largest_cluster->points.push_back(cloud->points[idx]);
  return largest_cluster;
}

void MapROS::resetMapForNewEpisode()
{
  ROS_INFO("[MapROS] Resetting all maps for new episode");

  // Reset SDF map buffers (occupancy, distance, inflation)
  map_->resetAllBuffers();

  // Reset multi-source value map if enabled
  if (use_multi_source_semantic_ && map_->multi_source_value_map_) {
    map_->multi_source_value_map_->resetForNewEpisode();
  }

  // Reset IG value map if enabled
  if (map_->use_ig_value_map_ && map_->ig_value_map_) {
    map_->ig_value_map_->reset();
  }

  // Reset object map to clear accumulated objects
  if (map_->object_map2d_) {
    map_->object_map2d_->clearObjects();
  }

  // Reset tracking state
  continue_over_depth_count_ = -1;
  current_semantic_prompts_.clear();
  current_semantic_scores_.clear();
  current_target_object_.clear();
  current_nav_target_object_id_ = -1;  // Reset VLM target tracking
  local_updated_ = false;
  esdf_need_update_ = false;

  // Reset timing
  map_start_time_ = ros::Time::now();

  ROS_INFO("[MapROS] Episode reset complete - SDF map and Value Maps cleared");
}

void MapROS::episodeResetCallback(const std_msgs::Empty::ConstPtr& msg)
{
  ROS_INFO("[MapROS] Received episode reset signal");
  resetMapForNewEpisode();
}

}  // namespace apexnav_planner